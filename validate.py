"""Validate optical flow estimation performance on standard datasets."""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2 as cv
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import ptlflow
from ptlflow import get_model, get_model_reference
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils import flow_utils
from ptlflow.utils.external.raft import InputPadder
from ptlflow.utils.utils import (
    add_datasets_to_parser, config_logging, get_list_of_available_models_list, tensor_dict_to_numpy, InputScaler)

config_logging()


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        'model', type=str, choices=get_list_of_available_models_list(),
        help='Name of the model to use.')
    parser.add_argument(
        '--datasets', type=str, nargs='+', default=['kitti', 'sintel'],
        help='Names of the datasets to use for the validation. Supported datasets are {{\'kitti\',\'sintel\'}}')
    parser.add_argument(
        '--output_path', type=str, default=str(Path('outputs/validate')),
        help='Path to the directory where the validation results will be saved.')
    parser.add_argument(
        '--prediction_keys', type=str, nargs='+', default=['flows', 'occs', 'mbs', 'confs'],
        help='Keys of the model output dict to retrieve to compute the validations metrics.')
    parser.add_argument(
        '--write_flow', action='store_true',
        help='If set, the estimated flow is saved to disk.')
    parser.add_argument(
        '--write_viz', action='store_true',
        help='If set, an RGB version of the estimated flow is saved to disk.')
    parser.add_argument(
        '--show', action='store_true',
        help='If set, the results are shown on the screen.')
    parser.add_argument(
        '--flow_format', type=str, default='original', choices=['flo', 'png', 'original'],
        help=('The format to use when saving the estimated optical flow. If \'original\', then the format will be the same '
              + 'one the dataset uses for the groundtruth.'))
    parser.add_argument(
        '--max_forward_side', type=int, default=None,
        help=('If max(height, width) of the input image is larger than this value, then the image is downscaled '
              'before the forward and the outputs are bilinearly upscaled to the original resolution.'))
    parser.add_argument(
        '--max_show_side', type=int, default=1000,
        help=('If max(height, width) of the output image is larger than this value, then the image is downscaled '
              'before showing it on the screen.'))
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help=('Maximum number of samples per dataset will be used for calculating the metrics.'))
    return parser


def generate_outputs(
    args: Namespace,
    inputs: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Display on screen and/or save outputs to disk, if required.

    Parameters
    ----------
    args : Namespace
        The arguments with the required values to manage the outputs.
    inputs : Dict[str, torch.Tensor]
        The inputs loaded from the dataset (images, groundtruth).
    preds : Dict[str, torch.Tensor]
        The model predictions (optical flow and others).
    dataloader_name : str
        A string to identify from which dataloader these inputs came from.
    batch_idx : int
        Indicates in which position of the loader this input is.
    metadata : Dict[str, Any], optional
        Metadata about this input, if available.
    """
    inputs = tensor_dict_to_numpy(inputs)
    inputs['flows_viz'] = flow_utils.flow_to_rgb(inputs['flows'])[:, :, ::-1]
    preds = tensor_dict_to_numpy(preds)
    preds['flows_viz'] = flow_utils.flow_to_rgb(preds['flows'])[:, :, ::-1]

    if args.show:
        _show(inputs, preds)

    if args.write_flow or args.write_viz:
        _write_to_file(args, preds, dataloader_name, batch_idx, metadata)


def validate(
    args: Namespace,
    model: BaseModel
) -> pd.DataFrame:
    """Perform the validation.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    model : BaseModel
        The model to be used for validation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the metric results.

    See Also
    --------
    ptlflow.models.base_model.base_model.BaseModel : The parent class of the available models.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataloaders = model.val_dataloader()
    dataloaders = {model.val_dataloader_names[i]: dataloaders[i] for i in range(len(dataloaders))}

    metrics_df = pd.DataFrame()
    metrics_df['model'] = [args.model]
    metrics_df['checkpoint'] = [args.pretrained_ckpt]

    for dataset_name, dl in dataloaders.items():
        metrics_mean = validate_one_dataloader(args, model, dl, dataset_name)
        metrics_df[[f'{dataset_name}-{k}' for k in metrics_mean.keys()]] = metrics_mean.values()
    metrics_df = metrics_df.round(3)
    return metrics_df


def validate_one_dataloader(
    args: Namespace,
    model: BaseModel,
    dataloader: DataLoader,
    dataloader_name: str,
) -> Dict[str, float]:
    """Perform validation for all examples of one dataloader.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    model : BaseModel
        The model to be used for validation.
    dataloader : DataLoader
        The dataloader for the validation.
    dataloader_name : str
        A string to identify this dataloader.

    Returns
    -------
    Dict[str, float]
        The average metric values for this dataloader.
    """
    metrics_sum = {}
    with tqdm(dataloader) as tdl:
        for i, inputs in enumerate(tdl):
            inputs['images_orig'] = inputs['images'].clone()
            inputs, scaler, padder = _prepare_inputs(inputs, model, args.max_forward_side)

            if torch.cuda.is_available():
                inputs['images'] = inputs['images'].cuda()
            preds = model(inputs)

            # Release GPU memory and upscale outputs, if necessary
            inputs['images'] = inputs['images_orig'].clone()
            del inputs['images_orig']

            for k in args.prediction_keys:
                if k in preds:
                    v = padder.unpad(preds[k].detach().cpu())
                    if scaler is not None:
                        v = scaler.unscale(v)
                    preds[k] = v

            metrics = model.val_metrics(preds, inputs)
            for k in metrics.keys():
                metrics[k] = metrics[k].detach().cpu()
                if metrics_sum.get(k) is None:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += metrics[k].item()
            tdl.set_postfix(epe=metrics_sum['val/epe']/(i+1))

            generate_outputs(args, inputs, preds, dataloader_name, i, inputs.get('meta'))

            if args.max_samples is not None and i >= (args.max_samples - 1):
                break

    metrics_mean = {}
    for k, v in metrics_sum.items():
        metrics_mean[k] = v / len(dataloader)
    return metrics_mean


def _prepare_inputs(
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
    max_forward_side: int
) -> Tuple[Dict[str, torch.Tensor], InputScaler, InputPadder]:
    scaler = None
    if max_forward_side is not None and max(inputs['images'].shape[-2:]) > max_forward_side:
        scale_factor = float(max_forward_side) / max(inputs['images'].shape[-2:])
        scaler = InputScaler(inputs['images'].shape, scale_factor=scale_factor)
        inputs['images'] = scaler.scale(inputs['images'])
    padder = InputPadder(inputs['images'].shape, stride=model.output_stride)
    inputs['images'] = padder.pad(inputs['images'])
    return inputs, scaler, padder


def _show(
    inputs: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor]
) -> None:
    for k, v in inputs.items():
        if len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3:
            if max(v.shape[:2]) > args.max_show_side:
                scale_factor = float(args.max_show_side) / max(v.shape[:2])
                v = cv.resize(v, (int(scale_factor*v.shape[1]), int(scale_factor*v.shape[0])))
            cv.imshow(k, v)
    for k, v in preds.items():
        if len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3:
            if max(v.shape[:2]) > args.max_show_side:
                scale_factor = float(args.max_show_side) / max(v.shape[:2])
                v = cv.resize(v, (int(scale_factor*v.shape[1]), int(scale_factor*v.shape[0])))
            cv.imshow('pred_'+k, v)
    cv.waitKey(1)


def _write_to_file(
    args: Namespace,
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    out_flow_dir = Path(args.output_path) / dataloader_name / 'flow'
    out_viz_dir = Path(args.output_path) / dataloader_name / 'viz'

    if metadata is not None:
        img_path = Path(metadata['image_paths'][0][0])
        image_name = img_path.stem
        if 'sintel' in dataloader_name:
            seq_name = img_path.parts[-2]
            out_flow_dir /= seq_name
            out_viz_dir /= seq_name
    else:
        image_name = f'{batch_idx:08d}'

    if args.flow_format != 'original':
        flow_ext = args.flow_format
    else:
        if 'kitti' in dataloader_name or 'hd1k' in dataloader_name:
            flow_ext = 'png'
        elif 'sintel' in dataloader_name:
            flow_ext = 'flo'

    if args.write_flow:
        out_flow_dir.mkdir(parents=True, exist_ok=True)
        flow_utils.flow_write(out_flow_dir / f'{image_name}.{flow_ext}', preds['flows'])
    if args.write_viz:
        out_viz_dir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out_viz_dir / f'{image_name}.png'), preds['flows_viz'])


if __name__ == '__main__':
    parser = _init_parser()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None
    if len(sys.argv) > 1 and sys.argv[1] != '-h' and sys.argv[1] != '--help' and sys.argv[1] != 'all':
        FlowModel = get_model_reference(sys.argv[1])
        parser = FlowModel.add_model_specific_args(parser)

    add_datasets_to_parser(parser, 'datasets.yml')

    args = parser.parse_args()

    if args.val_dataset is None:
        args.val_dataset = 'sintel-final-trainval+sintel-clean-trainval+kitti-2012-trainval+kitti-2015-trainval'
        logging.warning('--val_dataset is not set. It will be set to %s', args.val_dataset)

    if args.model != 'all':
        model_id = args.model
        if args.pretrained_ckpt is not None:
            model_id += f'_{args.pretrained_ckpt}'
        args.output_path = Path(args.output_path) / model_id
        model = get_model(sys.argv[1], args.pretrained_ckpt, args)

        metrics_df = validate(args, model)
        args.output_path.mkdir(parents=True, exist_ok=True)
        metrics_df.T.to_csv(args.output_path / 'metrics.csv', header=False)
    else:
        # Run validation on all models and checkpoints
        metrics_df = pd.DataFrame()

        model_names = ptlflow.models_dict.keys()
        for mname in model_names:
            logging.info(mname)
            model_ref = ptlflow.get_model_reference(mname)

            if hasattr(model_ref, 'pretrained_checkpoints'):
                ckpt_names = model_ref.pretrained_checkpoints.keys()
                for cname in ckpt_names:
                    logging.info(cname)
                    parser_tmp = model_ref.add_model_specific_args(parser)
                    args = parser_tmp.parse_args()

                    args.model = mname
                    args.pretrained_ckpt = cname

                    model_id = args.model
                    if args.pretrained_ckpt is not None:
                        model_id += f'_{args.pretrained_ckpt}'
                    args.output_path = Path(args.output_path) / model_id

                    model = get_model(mname, cname, args)
                    instance_metrics_df = validate(args, model)
                    metrics_df = pd.concat([metrics_df, instance_metrics_df])
                    args.output_path.parent.mkdir(parents=True, exist_ok=True)
                    metrics_df.to_csv(args.output_path.parent / 'metrics_all.csv', index=False)
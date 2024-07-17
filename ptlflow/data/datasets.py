"""Handle common datasets used in optical flow estimation."""

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
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ptlflow.utils import flow_utils
from ptlflow.utils.utils import config_logging

config_logging()

THIS_DIR = Path(__file__).resolve().parent


class BaseFlowDataset(Dataset):
    """Manage optical flow dataset loading.

    This class can be used as the parent for any concrete dataset. It is structured to be able to read most types of inputs
    used in optical flow estimation.

    Classes inheriting from this one should implement the __init__() method and properly load the input paths from the chosen
    dataset. This should be done by populating the lists defined in the attributes below.

    Attributes
    ----------
    img_paths : list[list[str]]
        Paths of the images. Each element of the main list is a list of paths. Typically, the inner list will have two
        elements, corresponding to the paths of two consecutive images, which will be used to estimate the optical flow.
        More than two paths can also be added in case the model is able to use more images for estimating the flow.
    flow_paths : list[list[str]]
        Similar structure to img_paths. However, the inner list must have exactly one element less than img_paths.
        For example, if an entry of img_paths is composed of two paths, then an entry of flow_list should be a list with a
        single path, corresponding to the optical flow from the first image to the second.
    occ_paths : list[list[str]]
        Paths to the occlusion masks, follows the same structure as flow_paths. It can be left empty if not available.
    mb_paths : list[list[str]]
        Paths to the motion boundary masks, follows the same structure as flow_paths. It can be left empty if not available.
    flow_b_paths : list[list[str]]
        The same as flow_paths, but it corresponds to the backward flow. This list must be in the same order as flow_paths.
        For example, flow_b_paths[i] must be backward flow of flow_paths[i]. It can be left empty if backard flows are not
        available.
    occ_b_paths : list[list[str]]
        Backward occlusion mask paths, read occ_paths and flow_b_paths above.
    mb_b_paths : list[list[str]]
        Backward motion boundary mask paths, read mb_paths and flow_b_paths above.
    metadata : list[Any]
        Some metadata for each input. It can include anything. A good recommendation would be to put a dict with the metadata.
    """

    def __init__(
        self,
        dataset_name: str,
        split_name: str = "",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
    ) -> None:
        """Initialize BaseFlowDataset.

        Parameters
        ----------
        dataset_name : str
            A string representing the dataset name. It is just used to be stored as metadata, so it can have any value.
        split_name : str, optional
            A string representing the split of the data. It is just used to be stored as metadata, so it can have any value.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_motion_boundary_mask : bool, default True
            Whether to get motion boundary masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        """
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.transform = transform
        self.max_flow = max_flow
        self.get_valid_mask = get_valid_mask
        self.get_occlusion_mask = get_occlusion_mask
        self.get_motion_boundary_mask = get_motion_boundary_mask
        self.get_backward = get_backward
        self.get_meta = get_meta

        self.img_paths = []
        self.flow_paths = []
        self.occ_paths = []
        self.mb_paths = []
        self.flow_b_paths = []
        self.occ_b_paths = []
        self.mb_b_paths = []
        self.metadata = []

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # noqa: C901
        """Retrieve and return one input.

        Parameters
        ----------
        index : int
            The index of the entry on the input lists.

        Returns
        -------
        Dict[str, torch.Tensor]
            The retrieved input. This dict may contain the following keys, depending on the initialization choices:
            ['images', 'flows', 'mbs', 'occs', 'valids', 'flows_b', 'mbs_b', 'occs_b', 'valids_b', 'meta'].
            Except for 'meta', all the values are 4D tensors with shape NCHW. Notice that N does not correspond to the batch
            size, but rather to the number of images of a given key. For example, typically 'images' will have N=2, and
            'flows' will have N=1, and so on. Therefore, a batch of these inputs will be a 5D tensor BNCHW.
        """
        inputs = {}

        inputs["images"] = [cv2.imread(str(path)) for path in self.img_paths[index]]

        if index < len(self.flow_paths):
            inputs["flows"], valids = self._get_flows_and_valids(self.flow_paths[index])
            if self.get_valid_mask:
                inputs["valids"] = valids

        if self.get_occlusion_mask:
            if index < len(self.occ_paths):
                inputs["occs"] = [
                    cv2.imread(str(path), 0)[:, :, None]
                    for path in self.occ_paths[index]
                ]
            elif self.dataset_name.startswith("KITTI"):
                noc_paths = [
                    str(p).replace("flow_occ", "flow_noc")
                    for p in self.flow_paths[index]
                ]
                _, valids_noc = self._get_flows_and_valids(noc_paths)
                inputs["occs"] = [valids[i] - valids_noc[i] for i in range(len(valids))]
        if self.get_motion_boundary_mask and index < len(self.mb_paths):
            inputs["mbs"] = [
                cv2.imread(str(path), 0)[:, :, None] for path in self.mb_paths[index]
            ]

        if self.get_backward:
            if index < len(self.flow_b_paths):
                inputs["flows_b"], valids_b = self._get_flows_and_valids(
                    self.flow_b_paths[index]
                )
                if self.get_valid_mask:
                    inputs["valids_b"] = valids_b
            if self.get_occlusion_mask and index < len(self.occ_b_paths):
                inputs["occs_b"] = [
                    cv2.imread(str(path), 0)[:, :, None]
                    for path in self.occ_b_paths[index]
                ]
            if self.get_motion_boundary_mask and index < len(self.mb_b_paths):
                inputs["mbs_b"] = [
                    cv2.imread(str(path), 0)[:, :, None]
                    for path in self.mb_b_paths[index]
                ]

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.get_meta:
            inputs["meta"] = {
                "dataset_name": self.dataset_name,
                "split_name": self.split_name,
            }
            if index < len(self.metadata):
                inputs["meta"].update(self.metadata[index])

        return inputs

    def __len__(self) -> int:
        return len(self.img_paths)

    def _get_flows_and_valids(
        self, flow_paths: Sequence[str]
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        flows = []
        valids = []
        for path in flow_paths:
            flow = flow_utils.flow_read(path)

            nan_mask = np.isnan(flow)
            flow[nan_mask] = self.max_flow + 1

            if self.get_valid_mask:
                valid = (np.abs(flow) < self.max_flow).astype(np.uint8) * 255
                valid = np.minimum(valid[:, :, 0], valid[:, :, 1])
                valids.append(valid[:, :, None])

            flow[nan_mask] = 0

            flow = np.clip(flow, -self.max_flow, self.max_flow)
            flows.append(flow)
        return flows, valids

    def _log_status(self) -> None:
        if self.__len__() == 0:
            logging.warning(
                "No samples were found for %s dataset. Be sure to update the dataset path in datasets.yml, "
                "or provide the path by the argument --[dataset_name]_root_dir.",
                self.dataset_name,
            )
        else:
            logging.info(
                "Loading %d samples from %s dataset.", self.__len__(), self.dataset_name
            )

    def _extend_paths_list(
        self,
        paths_list: List[Union[str, Path]],
        sequence_length: int,
        sequence_position: str,
    ):
        if sequence_position == "first":
            begin_pad = 0
            end_pad = sequence_length - 2
        elif sequence_position == "middle":
            begin_pad = sequence_length // 2
            end_pad = int(math.ceil(sequence_length / 2.0)) - 2
        elif sequence_position == "last":
            begin_pad = sequence_length - 2
            end_pad = 0
        else:
            raise ValueError(
                f"Invalid sequence_position. Must be one of ('first', 'middle', 'last'). Received: {sequence_position}"
            )
        for _ in range(begin_pad):
            paths_list.insert(0, paths_list[0])
        for _ in range(end_pad):
            paths_list.append(paths_list[-1])
        return paths_list


class ECCVDataset(BaseFlowDataset):
    """Handle the ECCV Synthetic dataset with burst captures and optical flow files."""

    def __init__(
        self,
        root_dir: str = "/data/TrainingDatasets/ECCV2024/train_synth",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
        dataset_name: str = "ECCVDataset_Synth"
    ) -> None:
        """
        Initialize ECCVDataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the CVPR dataset.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow values above this are clipped.
        get_valid_mask : bool, default True
            Whether to generate valid masks based on flow value clipping.
        """
        super(ECCVDataset, self).__init__(dataset_name=dataset_name)
        self.root_dir = root_dir
        self.transform = transform
        self.max_flow = max_flow
        self.get_valid_mask = get_valid_mask
        self.get_meta = get_meta
        self.dataset_name = dataset_name
        self.split_name = "train"

        # Explore the dataset directory and prepare data paths
        self.img_paths = []
        self.flow_paths = []
        print(self.root_dir, len(list(Path(self.root_dir).glob("*"))))
        for subdir in Path(self.root_dir).glob("*"):
            npz_files = list(subdir.glob("raw_*.npz"))
            npz_files.sort()  # Ensure the files are in the correct order
            
            if len(npz_files) > 1:
                # First file is the reference frame
                # self.img_paths.append([npz_files[0], npz_files[1]])
                # Corresponding flow files (skip the first one as it is the reference)
                # self.flow_paths.append([subdir / f"flow_{i+2}.npz" for i in range(len(npz_files)-1)])
                reference_frame = npz_files[0]
                for idx in range(1, len(npz_files)):
                    self.img_paths.append([reference_frame, npz_files[idx]])
                    self.flow_paths.append([subdir / f"flow_{idx+1}.npz"])

        print(len(self.img_paths), len(self.flow_paths))
        assert len(self.img_paths) == len(self.flow_paths), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        self.metadata = [
            {
                "image_paths": [str(p) for p in paths],
                "is_val": False,
                "misc": "",
                "is_seq_start": True,
            }
            for paths in self.img_paths
        ]

    def __len__(self):
        return len(self.img_paths)

    def bayer_to_rgb(self, raw):
        if len(raw.shape) == 3 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        height, width = raw.shape
        rgb = np.zeros((height//2, width//2, 3), dtype=raw.dtype)
        rgb[:, :, 2] = raw[0::2, 0::2]  # Remember: OpenCV uses BGR by default, change indices for RGB
        rgb[:, :, 1] = (raw[0::2, 1::2] + raw[1::2, 0::2]) / 2
        rgb[:, :, 0] = raw[1::2, 1::2]
    
        return rgb

    def process_raw(self, image_path):
        raw = np.load(image_path)
        img1 = raw["raw"]
        img1 = (img1 / img1.max()).astype(np.float32) # modify if needed
        # img1 = cv2.resize(img1, (512, 384)) # resize if needed
        # img1 = self.bayer_to_rgb(img1) # train with rgb images
        if len(img1.shape) == 2:
            return img1[..., None]
        return img1
        
    def process_flow(self, flow_path):
        flow_npz = np.load(flow_path)
        flow = flow_npz["flow"]
        flow = (flow.astype(np.float32)-2**15)/10
        # flow = cv2.resize(flow, (512, 384)) # resize if needed
        return flow
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Load images
        images = [self.process_raw(path) for path in self.img_paths[index]]

        # Load flows
        flows = [self.process_flow(path) for path in self.flow_paths[index]]
        
        # Create a valid mask based on the max flow value
        # if self.get_valid_mask:
        #     valids = [np.linalg.norm(flow, axis=-1) < self.max_flow for flow in flows]
        
        # Package data
        data = {
            'images': np.stack(images, axis=0),#, dtype=torch.float32),
            'flows': np.stack(flows, axis=0)#, dtype=torch.float32)
        }

        valids_list = []
        if self.get_valid_mask:
            for item in flows:
                # data['valids'] = np.stack(valids, axis=0)#, dtype=torch.bool)
                valids = (np.abs(item) < self.max_flow).astype(np.uint8) * 255
                valids = np.minimum(valids[:, :, 0], valids[:, :, 1])
                valids_list.append(valids[:, :, None])

            data["valids"] = valids_list

        # Apply transform, if any
        if self.transform:
            data = self.transform(data)

        if self.get_meta:
            data["meta"] = {
                "dataset_name": self.dataset_name,
                "split_name": self.split_name,
            }
            if index < len(self.metadata):
                data["meta"].update(self.metadata[index])

        return data

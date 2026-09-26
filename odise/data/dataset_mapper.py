# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Modified by Jiarui Xu
# from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py # noqa
# ------------------------------------------------------------------------------

import copy
import logging
import numpy as np
import os.path as osp
from typing import List, Union
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances
from panopticapi.utils import rgb2id


class COCOPanopticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Read the depth map from "depth_file_name" (if available)
    3. Applies geometric transforms to the image, depth map, and annotation
    4. Find and applies suitable cropping to the image and annotation
    5. Prepare image and annotation to Tensors
    """

    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        segmentation_format: str = "L",
        caption_key: str = "coco_captions",
        depth_folder: str = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.augmentations = T.AugmentationList(augmentations)
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}] Full TransformGens used in training: {self.augmentations}"
        )

        self.img_format = image_format
        self.seg_format = segmentation_format
        self.cap_key = caption_key
        self.is_train = is_train
        self.depth_folder = depth_folder

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        original_image_shape = image.shape[:2]

        # Load depth map if depth_folder is specified
        depth_map = None
        depth_valid = None
        if self.depth_folder is not None:
            # Compute depth file name from image file name
            image_file_name = dataset_dict["file_name"]
            # Get the image id from the file name (e.g., 000000000009.jpg -> 000000000009)
            image_id = osp.splitext(osp.basename(image_file_name))[0]
            depth_file_name = osp.join(self.depth_folder, f"{image_id}.png")
            if osp.exists(depth_file_name):
                depth_map = utils.read_image(depth_file_name, format="L")  # Read as grayscale

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), format=self.seg_format
            )
            if self.seg_format == "L":
                sem_seg_gt = sem_seg_gt.squeeze(2)
        else:
            sem_seg_gt = None

        # Apply augmentations to image first (with sem_seg if available)
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # Apply the same geometric transforms to depth map
        if depth_map is not None:
            # Augmentations are sampled from the RGB image and expect every
            # image-like input to start at the same size.
            depth_valid = np.ones(depth_map.shape[:2], dtype=np.uint8)
            if depth_map.shape[:2] != original_image_shape:
                depth_resize = T.ResizeTransform(
                    depth_map.shape[0],
                    depth_map.shape[1],
                    original_image_shape[0],
                    original_image_shape[1],
                )
                depth_map = depth_resize.apply_image(depth_map)
                depth_valid = depth_resize.apply_segmentation(depth_valid)
            depth_map = transforms.apply_image(depth_map)
            # Segmentation transforms use nearest interpolation.  In
            # particular, FixedSizeCrop pads segmentation with a value other
            # than one, so equality preserves a genuine non-padding mask.
            depth_valid = transforms.apply_segmentation(depth_valid) == 1

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # Process depth map: resize to match image and convert to tensor
        if depth_map is not None:
            # Resize depth map to match augmented image size
            if depth_map.shape[:2] != image_shape:
                import cv2
                depth_map = cv2.resize(
                    depth_map, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR
                )
                depth_valid = cv2.resize(
                    depth_valid.astype(np.uint8),
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            # Ensure depth_map is 3D array (h, w, 1) for consistent processing
            if len(depth_map.shape) == 2:
                depth_map = depth_map[:, :, np.newaxis]
            dataset_dict["depth"] = torch.as_tensor(
                np.ascontiguousarray(depth_map.transpose(2, 0, 1)).astype(np.float32)
            ).float()
            dataset_dict["depth"] = (dataset_dict["depth"] / 255.0).clamp(0.0, 1.0)
            dataset_dict["depth_valid"] = torch.as_tensor(
                np.ascontiguousarray(depth_valid[None]).copy(), dtype=torch.bool
            )
        else:
            # A missing map must not become a valid-looking constant geometry
            # signal.  The model will bypass geometry conditioning for None.
            dataset_dict["depth"] = None
            dataset_dict["depth_valid"] = None

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
        segments_info = dataset_dict["segments_info"]

        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        pan_seg_gt = rgb2id(pan_seg_gt)
        dataset_dict["pan_seg_gt"] = torch.from_numpy(np.ascontiguousarray(pan_seg_gt))

        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()

        if self.cap_key in dataset_dict:
            dataset_dict["captions"] = dataset_dict.pop(self.cap_key)

        dataset_dict["instances"] = instances

        return dataset_dict


class DepthDatasetMapper:
    """Detectron2 inference mapper with an image-aligned monocular depth map.

    This mapper deliberately keeps Detectron2's normal dataset fields intact,
    including ``pan_seg_file_name`` and ``segments_info`` used by the ADE
    panoptic evaluator.  Depth is an auxiliary model input, not an annotation.
    """

    def __init__(
        self,
        is_train: bool = False,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        depth_folder: str,
    ):
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.depth_folder = depth_folder
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}] Augmentations used: {self.augmentations}"
        )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        original_image_shape = image.shape[:2]

        sem_seg_gt = None
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), format="L"
            ).squeeze(2)

        image_id = osp.splitext(osp.basename(dataset_dict["file_name"]))[0]
        depth_file_name = osp.join(self.depth_folder, f"{image_id}.png")
        if not osp.isfile(depth_file_name):
            raise FileNotFoundError(
                f"Depth map not found for {dataset_dict['file_name']}: {depth_file_name}"
            )
        depth = utils.read_image(depth_file_name, format="L")
        if depth.ndim == 3:
            depth = depth.squeeze(2)
        depth_valid = np.ones(depth.shape[:2], dtype=np.uint8)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        # The transform sequence was created from the RGB image. Align the
        # source map before applying it so ResizeTransform assertions cannot
        # fail when a map was generated at another resolution.
        if depth.shape[:2] != original_image_shape:
            depth_resize = T.ResizeTransform(
                depth.shape[0],
                depth.shape[1],
                original_image_shape[0],
                original_image_shape[1],
            )
            depth = depth_resize.apply_image(depth)
            depth_valid = depth_resize.apply_segmentation(depth_valid)
        depth = transforms.apply_image(depth)
        depth_valid = transforms.apply_segmentation(depth_valid) == 1

        image_shape = image.shape[:2]
        if depth.shape[:2] != image_shape:
            # This should only be needed for unusual custom transforms.  Keep
            # the tensor spatially aligned with the image as a final guard.
            depth_resize = T.ResizeTransform(
                depth.shape[0], depth.shape[1], image_shape[0], image_shape[1]
            )
            depth = depth_resize.apply_image(depth)
            depth_valid = depth_resize.apply_segmentation(depth_valid) == 1

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        dataset_dict["depth"] = torch.as_tensor(
            np.ascontiguousarray(depth).copy(), dtype=torch.float32
        ).unsqueeze(0).div(255.0).clamp_(0.0, 1.0)
        dataset_dict["depth_valid"] = torch.as_tensor(
            np.ascontiguousarray(depth_valid[None]).copy(), dtype=torch.bool
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        raise NotImplementedError("DepthDatasetMapper currently supports inference only")

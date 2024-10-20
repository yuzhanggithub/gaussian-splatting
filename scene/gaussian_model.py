#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys

import logging

logging.basicConfig(level=logging.DEBUG)  # Reconfigure logging

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math

from dataclasses import dataclass
from typing import Optional


def assert_sorted(tensor):
    assert torch.all(tensor[:-1] <= tensor[1:]), f"Tensor is not sorted. Size: {tensor.size()}, Values: {tensor}"

@dataclass
class VoxelGaussianArray:
    combined_hash: torch.Tensor
    root_voxel_hash: torch.Tensor  # Shape: (N, 3)
    relative_position_index_hash: torch.Tensor
    nearest_voxel_levels: torch.Tensor  # Shape: (N,)
    voxel_based_translations: torch.Tensor  # Shape: (N, 3)
    voxel_based_scales: torch.Tensor  # Shape: (N,)
    color: torch.Tensor  # Shape: (N, C)
    parent_hashes: torch.Tensor
    children_hashes: torch.Tensor

    def __init__(self):
        # Default empty initialization
        self.combined_hash = torch.empty(0, dtype=torch.long, device="cuda")
        self.root_voxel_hash = torch.empty(0, dtype=torch.long, device="cuda")
        self.relative_position_index_hash = torch.empty(
            0, dtype=torch.long, device="cuda"
        )
        self.nearest_voxel_levels = torch.empty(0, dtype=torch.long, device="cuda")
        self.voxel_based_translations = torch.empty(
            0, dtype=torch.float32, device="cuda"
        )
        self.voxel_based_scales = torch.empty(0, dtype=torch.float32, device="cuda")
        self.color = torch.empty(
            (0, 3), dtype=torch.float32, device="cuda"
        )  # Assuming C=3
        self.parent_hashes = torch.empty(0, dtype=torch.long, device="cuda")
        self.children_hashes = torch.empty((0, 8), dtype=torch.long, device="cuda")

    @classmethod
    def from_tensors(
        cls,
        combined_hash: torch.Tensor,
        root_voxel_hash: torch.Tensor,
        relative_position_index_hash: torch.Tensor,
        nearest_voxel_levels: torch.Tensor,
        voxel_based_translations: torch.Tensor,
        voxel_based_scales: torch.Tensor,
        color: torch.Tensor,
        parent_hashes: torch.Tensor,
        children_hashes: torch.Tensor,
    ):
        # Initialize using the provided tensors
        instance = cls()  # Create an empty instance
        instance.combined_hash = combined_hash
        instance.root_voxel_hash = root_voxel_hash
        instance.relative_position_index_hash = relative_position_index_hash
        instance.nearest_voxel_levels = nearest_voxel_levels
        instance.voxel_based_translations = voxel_based_translations
        instance.voxel_based_scales = voxel_based_scales
        instance.color = color
        instance.parent_hashes = parent_hashes
        instance.children_hashes = children_hashes
        return instance

    def assert_same_dimension(self):
        num_size = self.combined_hash.size(0)
        assert num_size == self.root_voxel_hash.size(
            0
        ), "Mismatch: combined_hash and root_voxel_hash have different sizes."
        assert num_size == self.relative_position_index_hash.size(
            0
        ), "Mismatch: combined_hash and relative_position_index_hash have different sizes."
        assert num_size == self.nearest_voxel_levels.size(
            0
        ), "Mismatch: combined_hash and nearest_voxel_levels have different sizes."
        assert num_size == self.voxel_based_translations.size(
            0
        ), "Mismatch: combined_hash and voxel_based_translations have different sizes."
        assert num_size == self.voxel_based_scales.size(
            0
        ), "Mismatch: combined_hash and voxel_based_scales have different sizes."
        assert num_size == self.color.size(
            0
        ), "Mismatch: combined_hash and color have different sizes."
        assert num_size == self.parent_hashes.size(
            0
        ), "Mismatch: combined_hash and parent_hash have different sizes."
        assert num_size == self.children_hashes.size(
            0
        ), "Mismatch: combined_hash and children_hashes have different sizes."

    def assert_not_empty(self):
        assert self.combined_hash.size(0) != 0
        assert self.root_voxel_hash.size(0) != 0
        assert self.relative_position_index_hash.size(0) != 0
        assert self.nearest_voxel_levels.size(0) != 0
        assert self.voxel_based_translations.size(0) != 0
        assert self.voxel_based_scales.size(0) != 0
        assert self.color.size(0) != 0
        assert self.parent_hashes.size(0) != 0
        assert self.children_hashes.size(0) != 0

    def assert_is_valid(self):
        self.assert_not_empty()
        self.assert_same_dimension()

    def print_hash(self):
        try:
            logging.debug(
                f"combined_hash, {self.combined_hash.size()}: {self.combined_hash}"
            )
        except Exception as e:
            logging.error(f"Error formatting combined_hash: {e}")
            logging.debug(f"combined_hash raw: {self.combined_hash}")

        try:
            logging.debug(
                f"parent_hashes, {self.parent_hashes.size()}: {self.parent_hashes}"
            )
        except Exception as e:
            logging.error(f"Error formatting parent_hashes: {e}")
            logging.debug(f"parent_hashes raw: {self.parent_hashes}")

        try:
            logging.debug(
                f"children_hashes, {self.children_hashes.size()}: {self.children_hashes}"
            )
        except Exception as e:
            logging.error(f"Error formatting children_hashes: {e}")
            logging.debug(f"children_hashes raw: {self.children_hashes}")

    def print_size(self):
        """
        Logs the size of each attribute in the object. If any attribute raises an error during formatting, it logs the error and its raw value.
        """
        attributes = [
            "combined_hash",
            "root_voxel_hash",
            "relative_position_index_hash",
            "nearest_voxel_levels",
            "voxel_based_translations",
            "voxel_based_scales",
            "color",
            "parent_hashes",
            "children_hashes",
        ]
        
        logging.debug(f"========== Print the size ==========")
        for attr in attributes:
            try:
                value = getattr(self, attr)
                logging.debug(f"{attr}, {value.size()}")
            except Exception as e:
                logging.error(f"Error formatting {attr}: {e}")
                logging.debug(f"{attr} raw: {value}")


def get_gaussian_array(gaussians: VoxelGaussianArray, sorted_order):
    return VoxelGaussianArray.from_tensors(
        combined_hash=gaussians.combined_hash[sorted_order],
        root_voxel_hash=gaussians.root_voxel_hash[sorted_order],
        relative_position_index_hash=gaussians.relative_position_index_hash[
            sorted_order
        ],
        nearest_voxel_levels=gaussians.nearest_voxel_levels[sorted_order],
        voxel_based_translations=gaussians.voxel_based_translations[sorted_order],
        voxel_based_scales=gaussians.voxel_based_scales[sorted_order],
        color=gaussians.color[sorted_order],
        parent_hashes=gaussians.parent_hashes[sorted_order],
        children_hashes=gaussians.children_hashes[sorted_order],
    )


def cat_gaussian_array(
    gaussians_0: VoxelGaussianArray, gaussians_1: VoxelGaussianArray
):
    return VoxelGaussianArray.from_tensors(
        combined_hash=torch.cat(
            [gaussians_0.combined_hash, gaussians_1.combined_hash], dim=0
        ),
        root_voxel_hash=torch.cat(
            [gaussians_0.root_voxel_hash, gaussians_1.root_voxel_hash], dim=0
        ),
        relative_position_index_hash=torch.cat(
            [
                gaussians_0.relative_position_index_hash,
                gaussians_1.relative_position_index_hash,
            ],
            dim=0,
        ),
        nearest_voxel_levels=torch.cat(
            [gaussians_0.nearest_voxel_levels, gaussians_1.nearest_voxel_levels], dim=0
        ),
        voxel_based_translations=torch.cat(
            [
                gaussians_0.voxel_based_translations,
                gaussians_1.voxel_based_translations,
            ],
            dim=0,
        ),
        voxel_based_scales=torch.cat(
            [gaussians_0.voxel_based_scales, gaussians_1.voxel_based_scales], dim=0
        ),
        color=torch.cat([gaussians_0.color, gaussians_1.color], dim=0),
        parent_hashes=torch.cat(
            [gaussians_0.parent_hashes, gaussians_1.parent_hashes], dim=0
        ),
        children_hashes=torch.cat(
            [gaussians_0.children_hashes, gaussians_1.children_hashes], dim=0
        ),
    )


class GaussianModel:
    def setup_functions(self):
        """
        定义和初始化一些用于处理3D高斯模型参数的函数。
        """

        # 定义构建3D高斯协方差矩阵的函数
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """
            scaling here is 3d
            """
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 计算实际的协方差矩阵
            actual_covariance = L @ L.transpose(1, 2)
            # 提取对称部分
            symm = strip_symmetric(actual_covariance)
            return symm

        # 初始化一些激活函数
        # 用exp函数确保尺度参数非负
        self.scaling_activation = torch.exp
        # 尺度参数的逆激活函数，用于梯度回传
        self.scaling_inverse_activation = torch.log

        # 协方差矩阵的激活函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 用sigmoid函数确保不透明度在0到1之间
        self.opacity_activation = torch.sigmoid
        # 不透明度的逆激活函数
        self.inverse_opacity_activation = inverse_sigmoid

        # 用于标准化旋转参数的函数
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, input_max_voxel_length, input_max_voxel_level: int):
        """
        初始化3D高斯模型的参数。

        :param sh_degree: 球谐函数的最大次数，用于控制颜色表示的复杂度。
        :param input_max_voxel_length: the max voxel length
        :param input_max_voxel_level: the max level of the voxel
        """

        print("Start to init the gaussian model")

        # 当前激活的球谐次数，初始为0
        self.active_sh_degree = 0
        # 允许的最大球谐次数
        self.max_sh_degree = sh_degree

        # Set up the voxel related property.
        self.max_voxel_length = input_max_voxel_length
        self.max_voxel_level = input_max_voxel_level

        # To store voxel based gaussian information.
        self.voxel_gaussian_array = VoxelGaussianArray()

        # 3D高斯的中心位置（均值）
        self._xyz = torch.empty(0)

        # 第一个球谐系数，用于表示基础颜色
        self._features_dc = torch.empty(0)
        # 其余的球谐系数，用于表示颜色的细节和变化
        self._features_rest = torch.empty(0)

        # 3D高斯的尺度参数，控制高斯的宽度
        self._scaling = torch.empty(0)

        # 3D高斯的旋转参数，用四元数表示
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # ??? Trick?
        # 在2D投影中，每个高斯的最大半径
        self.max_radii2D = torch.empty(0)

        # 用于累积3D高斯中心位置的梯度
        self.xyz_gradient_accum = torch.empty(0)

        # ??? Refer to the viewpoint and add_densification_stats
        self.denom = torch.empty(0)

        # 优化器，用于调整上述参数以改进模型
        self.optimizer = None

        # 设置在训练过程中，用于密集化处理的3D高斯点的比例
        self.percent_dense = 0

        # ??? learning rate? where?
        self.spatial_lr_scale = 0

        # 调用setup_functions来初始化一些处理函数

        self.setup_functions()

        print("Finish init the gaussian model")

    def capture(self):
        return (
            self.active_sh_degree,
            # Capture the voxel related property.
            self.max_voxel_length,
            self.max_voxel_level,
            self.voxel_level,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            # Load the voxel related property.
            self.max_voxel_length,
            self.max_voxel_level,
            self.voxel_level,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        return self._exposure[self.exposure_mapping[image_name]]

    # ??? This is a tensor?
    # Get voxel_level
    @property
    def get_voxel_level(self):
        return self.voxel_level

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling.repeat(1, 3), scaling_modifier, self._rotation
        )

    def get_voxel_length(self, query_level):
        """
        Calculate the voxel length for the given query level.
        """
        # Ensure the query level is a tensor for batch operations
        query_level = torch.clamp(query_level, 0, self.max_voxel_level)
        voxel_length = self.max_voxel_length / (2**query_level)
        return voxel_length

    def get_voxel_sphere_radius(self, query_level):
        """
        Calculate the sphere radius at the voxel for the given query level.
        """
        return 0.5 * self.get_voxel_length(query_level)

    def get_nearest_voxel_level(self, length, without_clamp=False):
        """
        Find the smallest voxel level where the corresponding voxel length is greater than or equal to the given length.
        """
        length = length.squeeze()
        if length.ndim == 0:       
            length = length.unsqueeze(0)
        assert torch.all(length >= 0), f"Assert failed, length:{length}"

        length = torch.clamp(length, max=self.max_voxel_length)

        ratio = self.max_voxel_length / length
        voxel_level = torch.round(torch.log2(ratio))

        if without_clamp:
            return voxel_level.long()
        else:
            return torch.clamp(voxel_level.long(), 0, self.max_voxel_level)

    def get_nearest_voxel_index(self, level, query_translation):

        # Get the voxel length for each level and reshape for broadcasting with the 3D coordinates
        voxel_length = self.get_voxel_length(level)  # Shape: (N,)
        voxel_length = voxel_length[:, None]  # Reshape to (N, 1) for broadcasting

        nearest_index = torch.floor(query_translation / voxel_length).long()

        return nearest_index

    def get_root_voxel_index(self, query_translation):
        nearest_index = torch.floor(query_translation / self.max_voxel_length).long()
        return nearest_index

    def get_nearest_voxel_center(self, level, query_translation):
        """
        Calculate the nearest voxel center based on the provided voxel level and query translation.

        Parameters:
        - level (torch.Tensor): A 1D tensor representing the voxel levels for each query.
        - query_translation (torch.Tensor): A 2D tensor representing the query translation (e.g., xyz coordinates).

        Returns:
        - torch.Tensor: A 2D tensor representing the coordinates of the nearest voxel center.
        """
        # logging.debug(f"-----get_nearest_voxel_center start-----")
        # logging.debug(f"level, {level.size()}")
        # logging.debug(f"query_translation, {query_translation.size()}")
        # Get the voxel length for each level and reshape for broadcasting with the 3D coordinates
        voxel_length = self.get_voxel_length(level)  # Shape: (N,)
        voxel_length = voxel_length[:, None]  # Reshape to (N, 1) for broadcasting
        # logging.debug(f"voxel_length, {voxel_length.size()}")

        # Perform element-wise division and floor to get the nearest center, then scale back
        nearest_center = (
            torch.floor(query_translation / voxel_length) * voxel_length
            + voxel_length / 2
        )

        # logging.debug(f"-----get_nearest_voxel_center end-----")

        return nearest_center

    def get_relative_position_index_hash(
        self, query_level: torch.Tensor, query_translation: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a unique hash based on relative positions across multiple voxel levels.

        Each Gaussian's relative position index is calculated for all applicable levels (from 1 to num_levels).
        The final hash is a weighted sum of these relative indices, ensuring uniqueness.
        For root voxels (level 0), the hash is set to 0.

        Args:
            query_level (torch.Tensor): Tensor of shape (N,) containing the voxel levels.
            query_translation (torch.Tensor): Tensor of shape (N, 3) containing translation vectors.

        Returns:
            torch.Tensor: Tensor of shape (N,) containing the relative position index hash for each translation.
        """
        device = query_level.device

        # Define number of levels
        num_levels = self.max_voxel_level  # e.g., 9
        N = query_translation.size(0)  # Number of queries

        # Create a levels tensor (1 to num_levels) and broadcast it
        levels = torch.arange(1, num_levels + 1, device=device).view(
            -1, 1
        )  # Shape: (num_levels, 1)
        levels_broadcast = levels.repeat(1, N)  # Shape: (num_levels, N)

        # print(f"The size of levels_broadcast is {levels_broadcast.size()}")  # Should print torch.Size([num_levels, N])

        # Broadcast query_level for comparison
        query_level_broadcast = query_level.view(1, -1).repeat(
            num_levels, 1
        )  # Shape: (num_levels, N)

        # Create a mask where query_level >= current level
        cur_voxel_mask = (
            query_level_broadcast >= levels_broadcast
        )  # Shape: (num_levels, N)

        # Compute upper levels
        upper_levels_broadcast = levels_broadcast - 1  # Shape: (num_levels, N)

        # Expand query_translation for batch processing
        query_translation_expanded = query_translation.view(1, N, 3).repeat(
            num_levels, 1, 1
        )  # Shape: (num_levels, N, 3)
        # print(f"The size of query_translation_expanded is {query_translation_expanded.size()}")  # Should print torch.Size([num_levels, N, 3])

        # Flatten the batched tensors
        upper_levels_flattened = upper_levels_broadcast.view(
            -1
        )  # Shape: (num_levels * N,)
        query_translation_flattened = query_translation_expanded.view(
            -1, 3
        )  # Shape: (num_levels * N, 3)

        # Compute voxel_based_translation_upper using the flattened tensors
        voxel_based_translation_upper_flattened = self.get_nearest_voxel_center(
            upper_levels_flattened, query_translation_flattened
        )  # Shape: (num_levels * N, 3)

        # print(f"the voxel_based_translation_upper_flattened is {voxel_based_translation_upper_flattened}")

        # Reshape back to (num_levels, N, 3)
        voxel_based_translation_upper = voxel_based_translation_upper_flattened.view(
            num_levels, N, 3
        )

        # Compute local coordinates within the upper voxel
        local_coords = (
            query_translation_flattened.view(num_levels, N, 3)
            - voxel_based_translation_upper
        )  # Shape: (num_levels, N, 3)

        # Determine bits based on local_coords
        bit_x = (local_coords[:, :, 0] >= 0).long()  # Shape: (num_levels, N)
        bit_y = (local_coords[:, :, 1] >= 0).long()  # Shape: (num_levels, N)
        bit_z = (local_coords[:, :, 2] >= 0).long()  # Shape: (num_levels, N)

        # Combine bits to form relative_index (1 to 8)
        relative_index = (
            (bit_x << 2) | (bit_y << 1) | bit_z
        ) + 1  # Shape: (num_levels, N)

        # Compute weights for each level using exponential weighting: 10^(9 - level)
        weights = torch.pow(10, (9 - levels)).long()  # Shape: (num_levels, 1)

        # Compute weighted_relative_index
        weighted_relative_index = relative_index * weights  # Shape: (num_levels, N)

        # print(f"the cur_voxel_mask is {cur_voxel_mask}")
        # print(f"the relative_index is {relative_index}")
        # print(f"the weights is {weights}")
        # print(f"the weighted_relative_index is {weighted_relative_index}")

        # Apply mask to consider only valid levels
        weighted_relative_index = (
            weighted_relative_index * cur_voxel_mask.long()
        )  # Shape: (num_levels, N)

        # Accumulate the weighted relative indices into the hash
        relative_index_hash = weighted_relative_index.sum(dim=0)  # Shape: (N,)

        return relative_index_hash

    def get_relative_position_index_parent_level(
        self, relative_index_position_hash: torch.Tensor, parent_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts the relative position index for each Gaussian at the specified query_level from the combined hash.

        Args:
            relative_index_position_hash (torch.Tensor): Tensor of shape (N,) containing the combined hash.
            query_level (torch.Tensor): Tensor of shape (N,) containing the voxel levels.

        Returns:
            torch.Tensor: Tensor of shape (N,) containing the relative position index (0-9) for each Gaussian at the query_level.
        """
        # Ensure query_level is within the expected range (1 to 9)
        # If not, clamp to the valid range or handle accordingly
        # Here, we assume query_level is between 1 and 9 inclusive
        assert torch.all(
            (parent_level >= 0) & (parent_level <= 9)
        ), "query_level must be between 1 and 9 inclusive."

        device = relative_index_position_hash.device
        N = relative_index_position_hash.size(0)

        # Define powers of ten from 10^8 to 10^0 for levels 1 to 9
        power_tens = torch.tensor(
            [
                10**8,
                10**7,
                10**6,
                10**5,
                10**4,
                10**3,
                10**2,
                10**1,
                10**0,
            ],
            device=device,
            dtype=torch.long,
        )  # Shape: (9,)

        # Adjust query_level to zero-based index for indexing into power_tens
        query_level_indices = parent_level

        # Gather the corresponding power of ten for each Gaussian based on its query_level
        divisor = power_tens[query_level_indices]  # Shape: (N,)

        # Perform integer division and modulo to extract the relative index
        relative_index = (
            torch.div(relative_index_position_hash, divisor, rounding_mode="trunc") % 10
        )  # Shape: (N,)
        return relative_index

    def get_parent_relative_index_hash(
        self, relative_index_position_hash: torch.Tensor, parent_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts the parent's relative index hash for each Gaussian at the specified parent_level
        by truncating digits below the parent level.

        Args:
            relative_index_position_hash (torch.Tensor): Tensor of shape (N,) containing the combined hash.
            parent_level (torch.Tensor): Tensor of shape (N,) containing the voxel levels.

        Returns:
            torch.Tensor: Tensor of shape (N,) containing the parent's relative index hash.
        """
        # Ensure parent_level is within the expected range (0 to 8)
        assert torch.all(
            (parent_level >= 0) & (parent_level <= 9)
        ), "parent_level must be between 0 and 8 inclusive."

        device = relative_index_position_hash.device

        # Define powers of ten from 10^8 to 10^0
        power_tens = torch.tensor(
            [
                10**9,
                10**8,
                10**7,
                10**6,
                10**5,
                10**4,
                10**3,
                10**2,
                10**1,
            ],
            device=device,
            dtype=torch.long,
        )  # Shape: (9,)

        # Gather the corresponding power of ten for each Gaussian based on its parent_level
        divisor = power_tens[parent_level]  # Shape: (N,)

        # Remove all digits below the parent level by truncation
        parent_relative_index_hash = (
            torch.div(relative_index_position_hash, divisor, rounding_mode="trunc")
            * divisor
        )

        return parent_relative_index_hash

    def get_voxel_index_as_hash(self, voxel_index):
        # TODO(yu zhang): 2*10 may not be enough for large data set.
        # Adjusted base to avoid overflow in int32 format
        base = torch.tensor(
            [2**20, 2**10, 1], device=voxel_index.device, dtype=torch.long
        )

        # Multiply and sum along the last dimension to generate unique hash values
        adjusted_voxel_index = voxel_index + 2**9
        hash_values = (adjusted_voxel_index * base).sum(dim=-1)
        return hash_values

    def get_combined_hash(self, root_voxel_hashes, relative_position_index_hash):
        return root_voxel_hashes * 10**9 + relative_position_index_hash

    def get_root_voxel_hash_and_relative_position_index_hash(self, combined_hash):
        # Extract root voxel hash using torch.div with rounding_mode='floor'
        root_voxel_hash = torch.div(combined_hash, 10**9, rounding_mode="floor")

        # Extract relative position index hash by modulo operation
        relative_position_index_hash = combined_hash % 10**9

        return root_voxel_hash, relative_position_index_hash

    def get_root_gaussian_orders_by_root_voxel_hashes(self, query_root_voxel_hashes):
        # Perform binary search to find insertion indices
        query_hash_orders = torch.searchsorted(
            self.root_voxel_hashes, query_root_voxel_hashes
        )

        # Initialize orders with -1 indicating "not found"
        query_gaussian_orders = -torch.ones_like(
            query_root_voxel_hashes, dtype=torch.long, device="cuda"
        )

        # Create a mask for valid indices where the hash matches
        valid_mask = (query_hash_orders < self.root_voxel_hashes.size(0)) & (
            self.root_voxel_hashes[query_hash_orders] == query_root_voxel_hashes
        )

        # Assign the found indices to the result tensor
        query_gaussian_orders[valid_mask] = query_hash_orders[valid_mask]

        return query_gaussian_orders

    def get_root_gaussian_ids_by_root_voxel_hashes(self, query_root_voxel_hashes):
        # Retrieve the orders (indices) corresponding to the query hashes
        query_orders = self.get_root_gaussian_orders_by_root_voxel_hashes(
            query_root_voxel_hashes
        )

        # Initialize Gaussian IDs with -1 indicating "not found"
        query_gaussian_ids = -torch.ones_like(
            query_root_voxel_hashes, dtype=torch.long, device="cuda"
        )

        # Create a mask for valid orders (where the index is not -1)
        valid_mask = query_orders != -1

        # Assign the Gaussian IDs where the mask is True
        query_gaussian_ids[valid_mask] = self.root_gaussian_ids[
            query_orders[valid_mask]
        ]

        return query_gaussian_ids

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def compute_counts(self, unique_indices, total_elements):
        """
        Computes the counts of each unique key based on their indices.

        Args:
            unique_indices (torch.Tensor): 1D tensor containing indices of unique keys.
            total_elements (int): Total number of elements in the sorted tensor.

        Returns:
            torch.Tensor: 1D tensor containing counts of each unique key.
        """
        if unique_indices.numel() == 0:
            return torch.tensor([], device=unique_indices.device)

        # Compute the differences between consecutive unique indices
        counts = unique_indices[1:] - unique_indices[:-1]

        # Handle the last unique key
        last_count = total_elements - unique_indices[-1]
        counts = torch.cat([counts, last_count.unsqueeze(0)])

        return counts

    def create_unsorted_voxel_gaussian_array(self, raw_lengths, raw_translations):
        array = VoxelGaussianArray()
        array.nearest_voxel_levels = self.get_nearest_voxel_level(raw_lengths)

        # scales are loged
        array.voxel_based_scales = torch.tensor(
            [
                self.get_voxel_sphere_radius(level)
                for level in array.nearest_voxel_levels
            ]
        ).cuda()
        array.voxel_based_scales = torch.log(array.voxel_based_scales)[..., None]

        array.voxel_based_translations = self.get_nearest_voxel_center(
            array.nearest_voxel_levels, raw_translations
        )

        array.relative_position_index_hash = self.get_relative_position_index_hash(
            array.nearest_voxel_levels, raw_translations
        )
        array.root_voxel_hash = self.get_voxel_index_as_hash(
            self.get_root_voxel_index(raw_translations)
        )

        array.combined_hash = self.get_combined_hash(
            array.root_voxel_hash, array.relative_position_index_hash
        )

        assert (
            array.root_voxel_hash.shape
            == array.nearest_voxel_levels.shape
            == array.relative_position_index_hash.shape
        ), f"Shapes mismatch: root_voxel_hash {array.root_voxel_hash.shape}, nearest_voxel_levels {array.nearest_voxel_levels.shape}, relative_position_index {array.relative_position_index.shape}"

        num_gaussians = array.combined_hash.size(0)

        array.color = torch.full((num_gaussians, 3), 0, dtype=torch.long, device="cuda")
        array.parent_hashes = torch.full(
            (num_gaussians,), -1, dtype=torch.long, device="cuda"
        )

        array.children_hashes = torch.full(
            (num_gaussians, 8), -1, dtype=torch.long, device="cuda"
        )

        return array

    def print_summarize_voxel_hash_counts(
        self, grouped_root_voxel_hash_counts, num_bins=10
    ):
        """
        Summarizes the counts of voxel hashes based on specified number of bins.

        Args:
            grouped_root_voxel_hash_counts (torch.Tensor):
                1D tensor on GPU containing counts of each unique voxel hash.
            num_bins (int):
                Number of bins to divide the count range into.

        Prints:
            - Number of voxel hashes in each count range.
            - Maximum count among all voxel hashes.
        """
        # Ensure the input is a 1D tensor
        assert grouped_root_voxel_hash_counts.dim() == 1, "Input tensor must be 1D."

        # Find the maximum count
        max_count = grouped_root_voxel_hash_counts.max().item()

        # Calculate bin size using ceiling to cover all counts
        bin_size = math.ceil(max_count / num_bins)

        print("Voxel Hash Counts Summary:")

        for i in range(num_bins):
            lower = i * bin_size + 1
            upper = (i + 1) * bin_size
            count = (
                (
                    (grouped_root_voxel_hash_counts >= lower)
                    & (grouped_root_voxel_hash_counts <= upper)
                )
                .sum()
                .item()
            )
            print(f"  - Voxel hashes with {lower}-{upper} elements: {count}")

        # Count voxel hashes with counts greater than the last bin
        if max_count > num_bins * bin_size:
            count_ge = (
                (grouped_root_voxel_hash_counts > num_bins * bin_size).sum().item()
            )
            print(f"  - Voxel hashes with > {num_bins * bin_size} elements: {count_ge}")

        # Print the maximum count
        print(f"  - Maximum count among all voxel hashes: {max_count}")

    def sort_gaussians(self, gaussians: VoxelGaussianArray):
        _, sorted_order = torch.sort(gaussians.combined_hash)
        sorted_gaussians = get_gaussian_array(gaussians, sorted_order)
        return sorted_gaussians

    def sort_and_deduplicate_voxels(
        self, gaussians: VoxelGaussianArray
    ) -> VoxelGaussianArray:
        """
        Sorts voxel data based on composite keys, removes duplicates, and summarizes voxel hash counts.

        Args:
            gaussians (VoxelGaussianArray): Input VoxelGaussianArray containing voxel data.

        Returns:
            VoxelGaussianArray: VoxelGaussianArray with unique entries.
        """
        gaussians.assert_is_valid()

        # Sort based on the composite key
        sorted_combined_hash, sorted_order = torch.sort(gaussians.combined_hash)

        # Apply the sorted order to all tensors
        sorted_gaussians = get_gaussian_array(gaussians, sorted_order)

        # Identify unique composite keys
        unique_mask = torch.ones_like(
            sorted_combined_hash, dtype=torch.bool, device="cuda"
        )
        unique_mask[1:] = sorted_combined_hash[1:] != sorted_combined_hash[:-1]

        # Get the indices where unique_mask is True
        unique_indices = unique_mask.nonzero(as_tuple=False).squeeze()

        if unique_indices.dim() == 0:
            unique_indices = unique_indices.unsqueeze(0)

        # Compute counts of each unique key
        total_elements = sorted_combined_hash.size(0)
        counts = self.compute_counts(unique_indices, total_elements)  # Shape: (M,)

        # Retain only unique entries
        unique_gaussians = get_gaussian_array(sorted_gaussians, unique_indices)

        # Summarize the counts
        self.print_summarize_voxel_hash_counts(counts, num_bins=30)

        return unique_gaussians

    def sort_and_merge_voxels(
        self, gaussians: VoxelGaussianArray
    ) -> VoxelGaussianArray:
        """
        Sorts voxel data based on composite keys and merges Gaussians with the same hash by averaging their properties,
        including color as floating-point numbers.

        Args:
            gaussians (VoxelGaussianArray): The input GaussianArray to be sorted and merged.

        Returns:
            VoxelGaussianArray: A new GaussianArray with unique combined_hash values and averaged properties.
        """
        # Ensure the GaussianArray is valid
        gaussians.assert_not_empty()
        gaussians.assert_same_dimension()

        # Sort based on the combined_hash
        sorted_combined_hash, sorted_order = torch.sort(gaussians.combined_hash)
        sorted_gaussians = get_gaussian_array(
            gaussians, sorted_order
        )  # Assuming VoxelGaussianArray supports advanced indexing

        # Identify unique combined_hash values and their start indices
        unique_combined_hash, unique_indices = torch.unique(
            sorted_combined_hash,
            sorted=True,
            return_inverse=False,
            return_counts=False,
            return_index=True,
        )

        # Compute counts of each unique combined_hash
        counts = torch.diff(
            torch.cat(
                [
                    unique_indices,
                    torch.tensor(
                        [sorted_gaussians.combined_hash.size(0)],
                        device=gaussians.combined_hash.device,
                    ),
                ]
            )
        )

        # Number of unique Gaussians
        num_unique = unique_combined_hash.size(0)

        # Initialize tensors to hold averaged properties
        C = gaussians.color.size(
            1
        )  # Number of color channels (e.g., RGB or SH coefficients)
        averaged_gaussians = VoxelGaussianArray(
            combined_hash=unique_combined_hash.clone(),
            root_voxel_hash=gaussians.root_voxel_hash[unique_indices].clone(),
            relative_position_index_hash=gaussians.relative_position_index_hash[
                unique_indices
            ].clone(),
            nearest_voxel_levels=gaussians.nearest_voxel_levels[unique_indices].clone(),
            voxel_based_translations=torch.zeros_like(
                gaussians.voxel_based_translations[unique_indices]
            ),
            voxel_based_scales=torch.zeros_like(
                gaussians.voxel_based_scales[unique_indices]
            ),
            color=torch.zeros(
                (num_unique, C), dtype=gaussians.color.dtype, device="cuda"
            ),
            parent_hash=gaussians.parent_hashes[unique_indices].clone(),
            children_hashes=gaussians.children_hashes[unique_indices].clone(),
        )

        # Prepare indices for scatter operations
        # Repeat unique_indices according to counts to align with sorted_gaussians
        repeat_unique_indices = unique_indices.repeat_interleave(counts)

        # Ensure repeat_unique_indices is of type long for scatter operations
        repeat_unique_indices = repeat_unique_indices.to(torch.long)

        summed_translations = torch.zeros_like(
            averaged_gaussians.voxel_based_translations, dtype=torch.float32
        )
        summed_scales = torch.zeros_like(
            averaged_gaussians.voxel_based_scales, dtype=torch.float32
        )

        # Scatter add translations and scales
        summed_translations.scatter_add_(
            0,
            repeat_unique_indices.unsqueeze(1).expand(-1, 3),
            sorted_gaussians.voxel_based_translations,
        )
        summed_scales.scatter_add_(
            0,
            repeat_unique_indices.unsqueeze(1).expand(-1, 1),
            sorted_gaussians.voxel_based_scales.unsqueeze(1),
        )

        # Compute averages
        counts_float = counts.float().unsqueeze(1)  # Shape: (num_unique, 1)
        averaged_gaussians.voxel_based_translations = summed_translations / counts_float
        averaged_gaussians.voxel_based_scales = (
            summed_scales.squeeze(1) / counts.float()
        )

        # Average color (floating-point)
        # Assuming color is of shape (N, C) and is floating-point
        summed_colors = torch.zeros((num_unique, C), dtype=torch.float32, device="cuda")
        summed_colors.scatter_add_(
            0, repeat_unique_indices.unsqueeze(1).expand(-1, C), sorted_gaussians.color
        )
        averaged_colors = summed_colors / counts_float
        averaged_gaussians.color = averaged_colors

        return averaged_gaussians

    def remove_gaussians_and_descendants(
        self, gaussians: VoxelGaussianArray, indices_to_remove: torch.Tensor
    ) -> VoxelGaussianArray:
        """
        Deletes Gaussians with specified indices and all their descendants.

        Args:
            gaussians (VoxelGaussianArray): The input array of Gaussians.
            indices_to_remove (torch.Tensor): Tensor of indices to remove.

        Returns:
            VoxelGaussianArray: A new array with the specified Gaussians and their descendants removed.
        """
        # Validate input GaussianArray
        gaussians.assert_is_valid()

        # If no indices to remove, return the original array
        if indices_to_remove.size(0) == 0:
            return gaussians

        # Filter valid parent hashes
        cur_parent_hashes = gaussians.parent_hashes[indices_to_remove]
        valid_parent_mask = cur_parent_hashes != -1  # Identify valid parent hashes

        # Apply the mask to filter indices_to_remove for valid parents
        valid_indices_to_remove = indices_to_remove[valid_parent_mask]

        # Get the valid parent hashes
        cur_parent_hashes = cur_parent_hashes[valid_parent_mask]

        # Find the indices of the valid parent hashes in the combined hash array
        cur_parent_indices = torch.searchsorted(
            gaussians.combined_hash, cur_parent_hashes
        )

        # Compute the relative position index for the valid entries
        cur_relative_position_index = self.get_relative_position_index_parent_level(
            gaussians.relative_position_index_hash[valid_indices_to_remove],
            gaussians.nearest_voxel_levels[valid_indices_to_remove] - 1,
        )
        gaussians.children_hashes[cur_parent_indices, cur_relative_position_index] = -1

        # Initialize tensors to track all indices to remove
        all_combined_indices_to_remove = indices_to_remove.clone()
        cur_combined_indices_to_remove = indices_to_remove.clone()

        # Iterate through levels to find descendants
        for level in range(self.max_voxel_level + 1):  # Include max level
            logging.debug(f"---------- Current Level ---------: {level}")

            # Get the child hashes for current indices
            cur_children_hash_to_remove = gaussians.children_hashes[
                cur_combined_indices_to_remove
            ]
            logging.debug(f"cur_children_hash_to_remove: {cur_children_hash_to_remove}")

            # Flatten and filter out invalid (-1) hashes
            cur_children_hash_to_remove = cur_children_hash_to_remove[
                cur_children_hash_to_remove != -1
            ]
            logging.debug(f"cur_children_hash_to_remove: {cur_children_hash_to_remove}")

            # If no children to remove, break the loop
            if cur_children_hash_to_remove.size(0) == 0:
                break

            # Find the indices of the child hashes in the combined hash array
            children_indices_to_remove = torch.searchsorted(
                gaussians.combined_hash, cur_children_hash_to_remove
            )

            # Append the new indices to the full list
            all_combined_indices_to_remove = torch.cat(
                (all_combined_indices_to_remove, children_indices_to_remove)
            )

            logging.debug(f"cur_indices: {children_indices_to_remove}")

            # Update the current combined indices to process further
            cur_combined_indices_to_remove = children_indices_to_remove

        # Create a mask for all indices to remove
        remove_mask = torch.zeros(
            gaussians.combined_hash.size(0),
            dtype=torch.bool,
            device=gaussians.combined_hash.device,
        )
        remove_mask[all_combined_indices_to_remove] = True

        # Return the Gaussians that are not in the remove list
        return get_gaussian_array(gaussians, ~remove_mask)

    def build_parent_child_relationships(self, unique_gaussians: VoxelGaussianArray):
        logging.debug(f"range: {range(self.max_voxel_level, 0, -1)}")
        unique_gaussians.print_hash()
        unique_gaussians.assert_is_valid()

        # Iterate from max_voxel_level down to 1
        for level in range(self.max_voxel_level, 0, -1):
            logging.debug(f"---------- --------- current level ---------: {level}")

            # Step 1: Identify children at the current level
            children_mask = unique_gaussians.nearest_voxel_levels == level
            children_indices = children_mask.nonzero(as_tuple=False).squeeze()

            logging.debug(
                f"children_indices, {children_indices.size()}, : {children_indices}"
            )

            if children_indices.numel() == 0:
                logging.debug(f"No children found at level {level}. Skipping.")
                continue  # No gaussians at this level

            # Step 2: Compute required parent hashes for these children
            required_parent_levels = (
                unique_gaussians.nearest_voxel_levels[children_indices] - 1
            )
            required_parent_relative_index_hash = self.get_parent_relative_index_hash(
                unique_gaussians.relative_position_index_hash[children_indices],
                required_parent_levels,
            )
            required_parent_combined_hash = self.get_combined_hash(
                unique_gaussians.root_voxel_hash[children_indices],
                required_parent_relative_index_hash,
            )
            assert_sorted(required_parent_combined_hash)
            logging.debug(
                f"required_parent_relative_index_hash, {required_parent_relative_index_hash.size()}, : {required_parent_relative_index_hash}"
            )
            logging.debug(
                f"required_parent_combined_hash, {required_parent_combined_hash.size()}, : {required_parent_combined_hash}"
            )

            # Step 3: Assignt the parent_hashes
            unique_gaussians.parent_hashes[
                children_indices
            ] = required_parent_combined_hash

            # Step 4: Find unique required parent hashes and map children to them
            (
                unique_required_parent_combined_hash,
                children_to_unique_required_indices,
            ) = torch.unique(required_parent_combined_hash, return_inverse=True)
            logging.debug(
                f"children_to_unique_required_indices, {children_to_unique_required_indices.size()}, : {children_to_unique_required_indices}"
            )
            logging.debug(
                f"unique_required_parent_combined_hash, {unique_required_parent_combined_hash.size()}, : {unique_required_parent_combined_hash}"
            )
            assert_sorted(unique_required_parent_combined_hash)

            # Step 5: Identify existing parents at the required level (level - 1)
            existing_parent_mask = unique_gaussians.nearest_voxel_levels == (level - 1)
            existing_parent_combined_hash = unique_gaussians.combined_hash[
                existing_parent_mask
            ]
            assert_sorted(existing_parent_combined_hash)
            existing_parent_indices = existing_parent_mask.nonzero(as_tuple=True)[0]
            logging.debug(
                f"existing_parent_combined_hash, {existing_parent_combined_hash.size()}: {existing_parent_combined_hash}"
            )
            logging.debug(
                f"existing_parent_indices, {existing_parent_indices.size()}: {existing_parent_indices}"
            )

            # Step 6: Check which required parents already exist
            # Use searchsorted to find insertion indices
            found_unique_required_in_existing_indices = torch.searchsorted(
                existing_parent_combined_hash, unique_required_parent_combined_hash
            )

            # Create a mask to check if the found indices actually match the required hashes
            found_unique_required_in_existing_mask = (
                found_unique_required_in_existing_indices
                < existing_parent_combined_hash.size(0)
            ) & (
                existing_parent_combined_hash[
                    found_unique_required_in_existing_indices
                ]
                == unique_required_parent_combined_hash
            )

            logging.debug(
                
                f"found_unique_required_in_existing_mask, {found_unique_required_in_existing_mask.size()}: {found_unique_required_in_existing_mask}"
            )

            # Find which children map to these existing parents
            children_with_existing_parents_mask = (
                found_unique_required_in_existing_mask[
                    children_to_unique_required_indices
                ]
            )
            children_with_existing_parents = children_indices[
                children_with_existing_parents_mask
            ]
            children_without_existing_parents = children_indices[
                ~children_with_existing_parents_mask
            ]
            assert_sorted(children_with_existing_parents)
            assert_sorted(children_without_existing_parents)
            logging.debug(
                f"children_with_existing_parents, {children_with_existing_parents.size()}: {children_with_existing_parents}"
            )
            logging.debug(
                f"children_without_existing_parents, {children_without_existing_parents.size()}: {children_without_existing_parents}"
            )

            # Step 6: Assign existing parents to corresponding children
            if children_with_existing_parents_mask.any():
                # Indices of unique_required_parent_combined_hash that exist
                found_existing_indices = (
                    found_unique_required_in_existing_indices[
                        found_unique_required_in_existing_mask
                    ]
                )
                # Get the corresponding parent indices in unique_gaussians
                found_parent_indices = existing_parent_indices[
                    found_existing_indices
                ]
                logging.debug(
                    f"found_parent_indices, {found_parent_indices.size()}: {found_parent_indices}"
                )
                logging.debug(
                    f"found_existing_indices, {found_existing_indices.size()}: {found_existing_indices}"
                )

                # Assign children to parents based on relative position index
                existing_rel_pos = (
                    self.get_relative_position_index_parent_level(
                        unique_gaussians.relative_position_index_hash[
                            children_with_existing_parents
                        ],
                        unique_gaussians.nearest_voxel_levels[
                            children_with_existing_parents
                        ]
                        - 1,
                    )
                    - 1
                )  # Adjust to 0-based index
                logging.debug(
                    f"existing_rel_pos, {existing_rel_pos.size()}: {existing_rel_pos}"
                )

                assert (
                    (existing_rel_pos >= 0) & (existing_rel_pos < 8)
                ).all(), "relative_position_index out of bounds"

                _, children_to_unique_existing = torch.unique(
                    children_to_unique_required_indices[
                        children_with_existing_parents_mask
                    ],
                    dim=0,
                    return_inverse=True,
                )
                _, children_to_unique_new_indices = torch.unique(
                    children_to_unique_required_indices[
                        ~children_with_existing_parents_mask
                    ],
                    dim=0,
                    return_inverse=True,
                )
                assert_sorted(children_to_unique_existing)
                assert_sorted(children_to_unique_new_indices)

                unique_gaussians.children_hashes[
                    found_parent_indices[children_to_unique_existing],
                    existing_rel_pos,
                ] = unique_gaussians.combined_hash[children_with_existing_parents]

                logging.debug(
                    f"Assigned {children_with_existing_parents.size(0)} children to existing parents."
                )

                logging.debug(f"------- Existing ---------A summary here --------")
                unique_gaussians.print_hash()

            # Step 7: Handle children without existing parents
            if children_without_existing_parents.numel() != 0:
                # Step 8: Identify unique new parent hashes required
                unique_new_parent_combined_hash = unique_required_parent_combined_hash[
                    ~found_unique_required_in_existing_mask
                ]

                num_new_parents = unique_new_parent_combined_hash.size(0)
                logging.debug(f"Number of new parents to create: {num_new_parents}")

                # Step 10: Aggregate attributes for new parents
                sum_translations = torch.zeros(
                    (num_new_parents, 3),
                    device=children_without_existing_parents.device,
                )
                sum_scales = torch.zeros(
                    (num_new_parents,), device=children_without_existing_parents.device
                )
                sum_colors = torch.zeros(
                    (num_new_parents, unique_gaussians.color.size(1)),
                    device=children_without_existing_parents.device,
                )

                # Aggregate translations
                sum_translations.index_add_(
                    0,
                    children_to_unique_new_indices,
                    unique_gaussians.voxel_based_translations[
                        children_without_existing_parents
                    ],
                )

                # Aggregate scales
                sum_scales.index_add_(
                    0,
                    children_to_unique_new_indices,
                    unique_gaussians.voxel_based_scales[
                        children_without_existing_parents
                    ],
                )

                # Aggregate colors
                sum_colors.index_add_(
                    0,
                    children_to_unique_new_indices,
                    unique_gaussians.color[children_without_existing_parents].float(),
                )

                # Compute averages
                counts_new_parents = (
                    torch.bincount(children_to_unique_new_indices, minlength=num_new_parents)
                    .unsqueeze(1)
                    .float()
                )
                avg_translations = sum_translations / counts_new_parents
                avg_scales = (sum_scales / counts_new_parents.squeeze())[...,None]
                avg_colors = (sum_colors / counts_new_parents).round().long()

                logging.debug(f"avg_translations, {avg_translations.size()}: {avg_translations}")
                logging.debug(f"avg_scales, {avg_scales.size()}: {avg_scales}")
                logging.debug(f"avg_colors, {avg_colors.size()}: {avg_colors}")

                # Step 11: Decode new parent combined hashes to get root_voxel_hash and relative_position_index_hash
                (
                    unique_new_parent_root_voxel_hash,
                    unique_new_parent_relative_position_index_hash,
                ) = self.get_root_voxel_hash_and_relative_position_index_hash(
                    unique_new_parent_combined_hash
                )

                # Step 12: Create new parent gaussians
                new_parent_gaussians = VoxelGaussianArray.from_tensors(
                    combined_hash=unique_new_parent_combined_hash,
                    root_voxel_hash=unique_new_parent_root_voxel_hash,
                    relative_position_index_hash=unique_new_parent_relative_position_index_hash,
                    nearest_voxel_levels=torch.full(
                        (num_new_parents,),
                        level - 1,
                        dtype=torch.long,
                        device=unique_gaussians.combined_hash.device,
                    ),
                    voxel_based_translations=avg_translations,
                    voxel_based_scales=avg_scales,
                    color=avg_colors,
                    parent_hashes=torch.full(
                        (num_new_parents,),
                        -1,
                        dtype=torch.long,
                        device=unique_gaussians.combined_hash.device,
                    ),
                    children_hashes=torch.full(
                        (num_new_parents, 8),
                        -1,
                        dtype=torch.long,
                        device=unique_gaussians.combined_hash.device,
                    ),
                )

                # Step 13: Append new parents to unique_gaussians
                unique_gaussians = cat_gaussian_array(
                    unique_gaussians, new_parent_gaussians
                )

                # Step 14: Assign children to new parents
                # Find the indices of new parents in the updated unique_gaussians
                original_N = unique_gaussians.combined_hash.size(0) - num_new_parents
                new_parent_indices = torch.arange(
                    original_N,
                    original_N + num_new_parents,
                    device=unique_gaussians.combined_hash.device,
                )
                logging.debug(f"new_parent_indices, {new_parent_indices.size()}: {new_parent_indices}")

                # Determine relative positions for children
                new_rel_pos = (
                    self.get_relative_position_index_parent_level(
                        unique_gaussians.relative_position_index_hash[
                            children_without_existing_parents
                        ],
                        unique_gaussians.nearest_voxel_levels[
                            children_without_existing_parents
                        ]
                        - 1,
                    )
                    - 1
                )  # Adjust to 0-based index

                assert (
                    (new_rel_pos >= 0) & (new_rel_pos < 8)
                ).all(), "relative_position_index out of bounds"

                # Assign children to their respective new parents
                unique_gaussians.children_hashes[
                    new_parent_indices[children_to_unique_new_indices], new_rel_pos
                ] = unique_gaussians.combined_hash[children_without_existing_parents]

                logging.debug(
                    f"Assigned {children_without_existing_parents.size(0)} children to new parents."
                )

                # Step 15: sort
                unique_gaussians = self.sort_gaussians(unique_gaussians)

                logging.debug(f"------- Level {level} Processing Complete ---------")
                unique_gaussians.print_hash()

        # Step 16: print
        logging.debug("------- Final Sorted Gaussians ---------")
        unique_gaussians.print_hash()
        return unique_gaussians

    def create_from_pcd(
        self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float
    ):
        """
        从点云数据初始化模型参数。

        :param pcd: 点云数据，包含点的位置和颜色。
        :param spatial_lr_scale: 空间学习率缩放因子，影响位置参数的学习率。
        """
        print("Start create from pcd")

        self.spatial_lr_scale = spatial_lr_scale
        # 将点云的位置和颜色数据从numpy数组转换为PyTorch张量，并传送到CUDA设备上
        fused_translation = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # distCUDA2 is in sample knn
        # 计算点云中每个点到其最近的k个点的平均距离的平方，用于确定高斯的尺度参数
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )

        # Set the voxel based scales
        dist = torch.sqrt(dist2)
        self.voxel_gaussian_array = self.create_unsorted_voxel_gaussian_array(
            2 * dist, fused_translation
        )
        self.voxel_gaussian_array.color = fused_color

        self.voxel_gaussian_array.assert_is_valid()

        self.voxel_gaussian_array = self.sort_and_deduplicate_voxels(
            self.voxel_gaussian_array
        )

        self.voxel_gaussian_array.assert_is_valid()

        self.voxel_gaussian_array = self.build_parent_child_relationships(
            self.voxel_gaussian_array
        )

        num_of_gaussians = self.voxel_gaussian_array.voxel_based_translations.shape[0]
        print("Number of points at initialisation : ", num_of_gaussians)

        # quaternion store as wxyz
        rots = torch.zeros((num_of_gaussians, 4), device="cuda")
        rots[:, 0] = 1

        # 初始化每个点的不透明度为0.1（通过inverse_sigmoid转换）
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((num_of_gaussians, 1), dtype=torch.float, device="cuda")
        )

        # 初始化存储球谐系数的张量，每个颜色通道有(max_sh_degree + 1) ** 2个球谐系数
        features = (
            torch.zeros(
                (
                    self.voxel_gaussian_array.color.shape[0],
                    3,
                    (self.max_sh_degree + 1) ** 2,
                )
            )
            .float()
            .cuda()
        )
        # set color SH to only 1 color DC part.
        features[:, :3, 0] = self.voxel_gaussian_array.color
        features[:, 3:, 1:] = 0.0

        # 将以上计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(
            self.voxel_gaussian_array.voxel_based_translations.requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            self.voxel_gaussian_array.voxel_based_scales.requires_grad_(True)
        )

        # Disable roation to be learnable
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._rotation = rots.clone().detach()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {
            cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)
        }

        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        logging.debug(f"self._xyz, {self._xyz.size()}: {self._xyz}")  
        logging.debug(f"self._scaling, {self._scaling.size()}: {self._scaling}")

    def print_scaling(self):
        logging.debug(f"----------print_scaling start----------")
        logging.debug(f"self._xyz, {self._xyz.size()}: {self._xyz}")  
        logging.debug(f"self._scaling, {self._scaling.size()}: {self._scaling}")
        self.voxel_gaussian_array.print_size()
        logging.debug(f"----------print_scaling end----------")

    def training_setup(self, training_args):
        """
        设置训练参数，包括初始化用于累积梯度的变量，配置优化器，以及创建学习率调度器

        :param training_args: 包含训练相关参数的对象。
        """

        # 设置在训练过程中，用于密集化处理的3D高斯点的比例
        self.percent_dense = training_args.percent_dense

        # 初始化用于累积3D高斯中心点位置梯度的张量，用于之后判断是否需要对3D高斯进行克隆或切分
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 配置各参数的优化器，包括指定参数、学习率和参数名称
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        # 创建优化器，这里使用Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # 创建学习率调度器，用于对中心点位置的学习率进行调整
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.exposure_optimizer.param_groups:
            param_group["lr"] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            # 找到名为"xyz"的参数组，即3D高斯分布中心位置的参数
            if param_group["name"] == "xyz":
                # ??? lr is only based on xyz? Maybe also exposure, depth, etc.
                # 使用xyz_scheduler_args函数（一个根据迭代次数返回学习率的调度函数）计算当前迭代次数的学习率
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        # TODO(yu) not sure if works
        l = ["voxel_level", "x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        voxel_levels = self.voxel_level.detach().cpu().numpy()
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (voxel_levels, xyz, normals, f_dc, f_rest, opacities, scale, rotation),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        重置不透明度参数。这个方法将所有的不透明度值设置为一个较小的值(但不是0),以避免在训练过程中因为不透明度过低而导致的问题。
        """
        # Set a low opacities. more solid? hold for now?
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        print("Function load_ply called")

        plydata = PlyData.read(path)

        # TODO(yu): not sure if works
        voxel_level = np.asarray(plydata.elements[0]["voxel_level"])[..., np.newaxis]

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        将指定的参数张量替换到优化器中，这主要用于更新模型的某些参数（例如不透明度）并确保优化器使用新的参数值。

        :param tensor: 新的参数张量。
        :param name: 参数的名称，用于在优化器的参数组中定位该参数。
        :return: 包含已更新参数的字典。
        """
        # python copy the parameter to l and pass to Adam
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        删除不符合要求的3D高斯分布在优化器中对应的参数

        :param mask: 一个布尔张量,表示需要保留的3D高斯分布。
        :return: 更新后的可优化张量字典。
        """
        # python copy the parameter to l and pass to Adam
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # 更新优化器状态
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 删除旧状态并更新参数
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        删除不符合要求的3D高斯分布。

        :param mask: 一个布尔张量,表示需要删除的3D高斯分布。
        """
        # 生成有效点的掩码并更新优化器中的参数
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新各参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 更新累积梯度和其他相关张量
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新的参数张量添加到优化器的参数组中
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        """
        将新生成的3D高斯分布的属性添加到模型的参数中。
        """

        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        # TODO(yu): not sure if we should reset all tensors' xyz and scaling?
        # 将字典中的张量连接（concatenate）成可优化的张量。这个方法的具体实现可能是将字典中的每个张量进行堆叠，以便于在优化器中进行处理。
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densification_postfix(self, gaussians: VoxelGaussianArray):
        """
        将新生成的3D高斯分布的属性添加到模型的参数中。
        """

        d = {
            "xyz": gaussians.voxel_based_translations,
            "f_dc": gaussians.new_features_dc,
            "f_rest": gaussians.new_features_rest,
            "opacity": gaussians.new_opacities,
            "scaling": gaussians.new_scaling,
            "rotation": gaussians.new_rotation,
        }

        # TODO(yu): not sure if we should reset all tensors' xyz and scaling?
        # 将字典中的张量连接（concatenate）成可优化的张量。这个方法的具体实现可能是将字典中的每个张量进行堆叠，以便于在优化器中进行处理。
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        对那些梯度超过一定阈值且尺度大于一定阈值的3D高斯进行分割操作。
        这意味着这些高斯可能过于庞大，覆盖了过多的空间区域，需要分割成更小的部分以提升细节。
        """

        # Voxel process after densification
        exp_scaling = torch.exp(self._scaling)
        exp_scaling = exp_scaling.squeeze(1)
        new_nearest_voxel_levels = self.get_nearest_voxel_level(2 * exp_scaling)

        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # Padded grad should have same size/demension of the point.
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_scaling.squeeze(1) > self.percent_dense * scene_extent,
        )

        # With voxel restriction
        voxel_level_mask = (
            torch.tensor(new_nearest_voxel_levels).cuda() < self.max_voxel_level
        )

        torch.cuda.empty_cache()

        selected_pts_mask = torch.logical_and(selected_pts_mask, voxel_level_mask)

        # 计算新高斯分布的属性
        # 尺度
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        # 均值（新分布的中心点）
        means = torch.zeros((stds.size(0), 3), device="cuda")
        # 随机采样新的位置
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        # 计算新的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_xyz = self.adjust_translations(self.get_xyz[selected_pts_mask], new_xyz)

        sys.exit("Debug exit")
        # 调整尺度并保持其他属性
        # 将原始点的特征重复 N 次。
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # TODO
        # 将分割得到的新高斯分布的属性添加到模型中
        # self.try_merge_new_points()

        # ???why also delete here?
        # 删除原有过大的高斯分布
        # TODO(yu zhang): i guess this would not work anymore?
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def try_merge_new_points(
        self,
        new_combined_hash,
        new_scaling,
        new_rotation,
        new_features_dc,
        new_features_rest,
        new_opacity,
    ):
        """
        Attempts to merge new Gaussians into the existing VoxelGaussianArray.

        Args:
            new_combined_hash (torch.Tensor): Combined hashes of new Gaussians. Shape: (M,)
            new_scaling (torch.Tensor): Scaling of new Gaussians. Shape: (M, 1)
            new_rotation (torch.Tensor): Rotation matrices of new Gaussians. Shape: (M, 3)
            new_features_dc (torch.Tensor): DC features of new Gaussians. Shape: (M, 1, 1)
            new_features_rest (torch.Tensor): REST features of new Gaussians. Shape: (M, 1, 1)
            new_opacity (torch.Tensor): Opacity of new Gaussians. Shape: (M, 1)
        """
        # Ensure all tensors have the same number of Gaussians
        M = new_combined_hash.size(0)
        assert new_scaling.size(0) == M and new_rotation.size(0) == M
        assert new_features_dc.size(0) == M and new_features_rest.size(0) == M
        assert (
            new_opacity.size(0) == M
        ), "All new Gaussian tensors must have the same first dimension."

        # Compute root_voxel_hash and relative_position_index_hash from combined_hash
        (
            new_root_voxel_hash,
            new_relative_position_index_hash,
        ) = self.decode_combined_hash(new_combined_hash)

        # Compute relative_position_index for new Gaussians
        new_relative_position_index = self.compute_relative_position_index(
            new_relative_position_index_hash
        )

        # Create a temporary VoxelGaussianArray for new Gaussians
        new_gaussians = self.VoxelGaussianArray(
            combined_hash=new_combined_hash,  # Shape: (M,)
            root_voxel_hash=new_root_voxel_hash,  # Shape: (M, 3)
            relative_position_index_hash=new_relative_position_index_hash,  # Shape: (M,)
            nearest_voxel_levels=(
                self.get_nearest_voxel_level(new_scaling.squeeze(1)) + 1
            ).long(),  # Shape: (M,)
            relative_position_index=new_relative_position_index,  # Shape: (M,)
            voxel_based_translations=self.adjust_translations(
                new_scaling.repeat(1, 3) * new_rotation[:, :3]
            ),  # Adjust translations as needed
            voxel_based_scales=new_scaling,  # Shape: (M, 1)
            color=self.compute_color(
                new_features_dc, new_features_rest
            ),  # Shape: (M, C)
            parent_hash=torch.full(
                (M,), -1, dtype=torch.long, device="cuda"
            ),  # Initialize to -1
            children_hashes=torch.full(
                (M, 8), -1, dtype=torch.long, device="cuda"
            ),  # Initialize to -1
        )

        # Find existing Gaussians with the same combined_hash
        # Expand existing combined_hash for comparison
        existing_combined_hash = self.combined_hash  # Shape: (N,)
        new_hash = new_gaussians.combined_hash  # Shape: (M,)

        # Create a mask where combined_hash matches
        # To handle multiple matches, we'll use broadcasting
        # However, for efficiency, use a sorted approach or hashing
        # Here, we'll assume combined_hash is sorted; if not, sort it first
        sorted_existing_hash, sorted_indices = torch.sort(existing_combined_hash)
        sorted_unique_hash, unique_indices = torch.unique(
            sorted_existing_hash, return_inverse=False, return_counts=False
        )

        # Use torch.searchsorted to find matches
        required_parent_in_existing_indices = torch.searchsorted(
            sorted_existing_hash, new_hash
        )

        # Check if the inserted indices have matching hashes
        matches = (
            required_parent_in_existing_indices < sorted_existing_hash.size(0)
        ) & (sorted_existing_hash[required_parent_in_existing_indices] == new_hash)

        # Indices in existing_gaussians that match
        existing_match_indices = sorted_indices[
            required_parent_in_existing_indices[matches]
        ]

        # Indices in new_gaussians that have matches
        new_match_indices = matches.nonzero(as_tuple=False).squeeze()

        # Merge properties for matched Gaussians
        if new_match_indices.numel() > 0:
            # Existing Gaussians to merge with
            existing_gaussians_to_merge = existing_match_indices

            # Corresponding new Gaussians
            new_gaussians_to_merge = new_match_indices

            # Compute the number of matches
            num_matches = new_gaussians_to_merge.size(0)

            # Average translations and colors
            self.voxel_based_translations[existing_gaussians_to_merge] = (
                self.voxel_based_translations[existing_gaussians_to_merge]
                + new_gaussians.voxel_based_translations[new_gaussians_to_merge]
            ) / 2.0
            self.color[existing_gaussians_to_merge] = (
                self.color[existing_gaussians_to_merge].float()
                + new_gaussians.color[new_gaussians_to_merge].float()
            ) / 2.0
            self.color[existing_gaussians_to_merge] = (
                self.color[existing_gaussians_to_merge].round().long()
            )

            # Optionally, update other properties as needed (e.g., scaling, rotation)
            self.voxel_based_scales[existing_gaussians_to_merge] = (
                self.voxel_based_scales[existing_gaussians_to_merge]
                + new_gaussians.voxel_based_scales[new_gaussians_to_merge]
            ) / 2.0

            # Optionally, handle children_hashes if applicable
            # For simplicity, assume children_hashes remain unchanged

        # Handle new Gaussians that do not have existing matches
        new_no_match_mask = ~matches
        new_no_match_indices = new_no_match_mask.nonzero(as_tuple=False).squeeze()

        if new_no_match_indices.numel() > 0:
            # Extract Gaussians that need to be added
            gaussians_to_add = new_gaussians[new_no_match_indices]

            # Add these Gaussians to the existing VoxelGaussianArray
            self = self.cat_gaussian_array(self, gaussians_to_add)

            # Update parent-child relationships for the newly added Gaussians
            self.build_parent_child_relationships(self)

            self.densification_postfix(gaussians_to_add)

        # Optional: Clean up GPU memory
        torch.cuda.empty_cache()

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        对那些梯度超过一定阈值且尺度小于一定阈值的3D高斯进行克隆操作。
        这意味着这些高斯在空间中可能表示的细节不足，需要通过克隆来增加细节。
        """

        # Extract points that satisfy the gradient condition
        # ??? Needs to look deeper?
        # 建一个掩码，标记满足梯度条件的点。具体来说，对于每个点，计算其梯度的L2范数，如果大于等于指定的梯度阈值，则标记为True，否则标记为False。
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        # 在上述掩码的基础上，进一步过滤掉那些缩放（scaling）大于一定百分比（self.percent_dense）的场景范围（scene_extent）的点。这样可以确保新添加的点不会太远离原始数据。
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        # 提取这些点的属性
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 将克隆得到的新高斯分布的属性添加到模型中
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def voxelize_translation_and_scaling(self):
        exp_scaling = torch.exp(self._scaling)
        exp_scaling = exp_scaling.squeeze(1)  # Shape becomes [135041]

        # Use PyTorch for batch processing
        nearest_voxel_levels = self.get_nearest_voxel_level(2 * exp_scaling)

        # Calculate voxel-based scales using tensor operations
        voxel_based_scales = self.get_voxel_sphere_radius(
            nearest_voxel_levels
        )  # Batch processing
        # No need to repeat(1, 3)
        voxel_based_scales = torch.log(voxel_based_scales)[..., None]

        # Calculate voxel-based translations using batch-wise operation
        voxel_based_translation = self.get_nearest_voxel_center(
            nearest_voxel_levels, self._xyz
        )

        # Update the values of _xyz and _scaling in-place without resetting gradients
        self._xyz.data.copy_(voxel_based_translation)
        self._scaling.data.copy_(voxel_based_scales)

    def loss_voxelize_translation_and_scaling(self):
        # print
        # self.voxel_gaussian_array.print_size()

        exp_scaling = torch.exp(self._scaling)
        exp_scaling = exp_scaling.squeeze(1)
        # logging.debug(f"self._scaling, {self._scaling.size()}: {self._scaling}")
        
        # Use PyTorch for batch processing
        nearest_voxel_levels = self.get_nearest_voxel_level(2 * exp_scaling)

        # Get voxel-based scales and translations
        voxel_based_scales = self.get_voxel_sphere_radius(
            nearest_voxel_levels
        )  # Batch processing
        voxel_based_translation = self.get_nearest_voxel_center(
            nearest_voxel_levels, self._xyz
        )
        voxel_lengths = self.get_voxel_length(nearest_voxel_levels)

        # 1) Translation Loss
        # Calculate the distance between the original xyz and nearest voxel center
        translation_diff = self._xyz - voxel_based_translation
        translation_loss = torch.norm(translation_diff, dim=-1) / voxel_lengths

        # 2) Scaling Loss
        # Calculate the difference between the original scaling and the voxel-based scaling (radius)
        scaling_diff = torch.abs(exp_scaling - voxel_based_scales)
        scaling_loss = scaling_diff / voxel_based_scales

        # Average the losses
        loss_voxelize_translation = translation_loss.mean()
        loss_voxelize_scaling = scaling_loss.mean()

        # logging.debug(f"loss_voxelize_translation, {loss_voxelize_translation.size()}: {loss_voxelize_translation}")
        # logging.debug(f"loss_voxelize_scaling, {loss_voxelize_scaling.size()}: {loss_voxelize_scaling}")

        return loss_voxelize_translation, loss_voxelize_scaling

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        对3D高斯分布进行密集化和修剪的操作

        :param max_grad: 梯度的最大阈值，用于判断是否需要克隆或分割。
        :param min_opacity: 不透明度的最小阈值,低于此值的3D高斯将被删除。
        :param extent: 场景的尺寸范围，用于评估高斯分布的大小是否合适。
        :param max_screen_size: 最大屏幕尺寸阈值，用于修剪过大的高斯分布。
        """

        print("number of gaussians before densify_and_prune: ", self._xyz.shape[0])

        # 计算3D高斯中心的累积梯度并修正NaN值
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 根据梯度和尺寸阈值进行克隆或分割操作
        # self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 根据梯度和尺寸阈值进行克隆或分割操作
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        # Try delete very small gaussians?
        exp_scaling = torch.exp(self._scaling)
        exp_scaling = exp_scaling.squeeze(1)  # Shape becomes [135041]
        nearest_voxel_levels = self.get_nearest_voxel_level(
            2 * exp_scaling, without_clamp=True
        )
        high_voxel_levels = nearest_voxel_levels > self.max_voxel_level
        prune_mask = torch.logical_or(prune_mask, high_voxel_levels)

        # Try remove duplicate voxels
        nearest_voxel_index = self.get_nearest_voxel_index(
            nearest_voxel_levels, self._xyz
        )

        if torch.is_floating_point(nearest_voxel_levels):
            nearest_voxel_levels_int = torch.round(nearest_voxel_levels).to(torch.long)
        else:
            nearest_voxel_levels_int = nearest_voxel_levels.to(
                torch.long
            )  # If it's already int, just convert

        # Do the same for nearest_voxel_index if necessary
        if torch.is_floating_point(nearest_voxel_index):
            nearest_voxel_index_int = torch.round(nearest_voxel_index).to(torch.long)
        else:
            nearest_voxel_index_int = nearest_voxel_index.to(torch.long)

        combined_voxel_data = torch.cat(
            [nearest_voxel_levels_int.unsqueeze(1), nearest_voxel_index_int], dim=1
        )

        unique_voxels, inverse_indices = torch.unique(
            combined_voxel_data, dim=0, return_inverse=True
        )

        occurrences = torch.bincount(inverse_indices)

        duplicate_voxel_mask = occurrences[inverse_indices] > 1

        prune_mask = torch.logical_or(prune_mask, duplicate_voxel_mask)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # ???why 0.1 extent
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        # self.voxelize_translation_and_scaling()

        torch.cuda.empty_cache()

        print("number of gaussians after densify_and_prune: ", self._xyz.shape[0])

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # ??? local gradient may be very large/small, average gradient will more smooth.
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

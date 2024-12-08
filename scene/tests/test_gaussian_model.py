import pytest
import torch
from scene.gaussian_model import GaussianModel

def test_get_voxel_length_batch():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    query_levels = torch.tensor([-1, 0, 1, 2, 3, 4], dtype=torch.long, device='cuda')
    voxel_lengths = gaussians.get_voxel_length(query_levels)
    expected_lengths = torch.tensor([2.0, 2.0, 1.0, 0.5, 0.25, 0.25], dtype=torch.float32, device='cuda')
    assert torch.allclose(voxel_lengths, expected_lengths, atol=1e-6)

def test_get_voxel_sphere_radius_batch():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    ratio = 0.5
    query_levels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device='cuda')
    expected_voxel_lengths = ratio * gaussians.get_voxel_length(query_levels)
    voxel_sphere_radii = gaussians.get_voxel_sphere_radius(query_levels)
    torch.testing.assert_close(
        voxel_sphere_radii,
        expected_voxel_lengths,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
        msg="Voxel sphere radii do not match expected values."
    )

def test_get_nearest_voxel_level_batch():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    inputs = torch.tensor([100, 2.2, 2, 1.8, 1.2, 0.8, 0.25, 0.2], dtype=torch.float32, device='cuda')
    expected_levels = torch.tensor([0, 0, 0, 0, 1, 1, 3, 3], dtype=torch.int32, device='cuda')
    nearest_levels = gaussians.get_nearest_voxel_level(inputs)
    torch.testing.assert_close(
        nearest_levels,
        expected_levels,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
        msg=f"Nearest voxel levels do not match expected values.\nActual: {nearest_levels}\nExpected: {expected_levels}"
    )

def test_get_nearest_voxel_index_batch():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    query_translations = torch.tensor([
        [-2.8, -2.8, -2.8],
        [-2.0, -2.0, -2.0],
        [-0.8, -0.8, -0.8],
        [0.0, 0.0, 0.0],
        [0.8, 0.8, 0.8],
        [2.0, 2.0, 2.0],
        [2.8, 2.8, 2.8],
        [4.0, 4.0, 4.0],
        [4.8, 4.8, 4.8]
    ], dtype=torch.float32, device='cuda')
    
    query_levels = torch.zeros(query_translations.size(0), dtype=torch.long, device='cuda')
    
    expected_results = torch.tensor([
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0]
    ], dtype=torch.float32, device='cuda')

    voxel_translations = gaussians.get_nearest_voxel_index(query_levels, query_translations)
    
    torch.testing.assert_close(
        voxel_translations,
        expected_results,
        rtol=1e-5,      # Relative tolerance
        atol=1e-6,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg="Voxel translations do not match expected results."
    )

def test_get_nearest_voxel_center():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    query_translations = torch.tensor([
        [-2.8, -2.8, -2.8],
        [-2.0, -2.0, -2.0],
        [-0.8, -0.8, -0.8],
        [0.0, 0.0, 0.0],
        [0.8, 0.8, 0.8],
        [2.0, 2.0, 2.0],
        [2.8, 2.8, 2.8],
        [4.0, 4.0, 4.0],
        [4.8, 4.8, 4.8]
    ], dtype=torch.float32, device='cuda')
    
    query_levels = torch.zeros(query_translations.size(0), dtype=torch.long, device='cuda')
    
    expected_results = torch.tensor([
        [-2.0, -2.0, -2.0],
        [-2.0, -2.0, -2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0]
    ], dtype=torch.float32, device='cuda')

    voxel_center = gaussians.get_nearest_voxel_center(query_levels, query_translations)
    
    torch.testing.assert_close(
        voxel_center,
        expected_results,
        rtol=1e-5,      # Relative tolerance
        atol=1e-6,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg="Voxel translations do not match expected results."
    )

def test_get_relative_position_index_hash():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    query_translations = torch.tensor([
        [2.8, 2.8, 2.8]
    ], dtype=torch.float32, device='cuda')

    query_levels = torch.tensor([
        3
    ], dtype=torch.float32, device='cuda')
    
    expected_results = torch.tensor([
        0
    ], dtype=torch.float32, device='cuda')
    
    relative_position_index_hash = gaussians.get_relative_position_index_hash(query_levels, query_translations)
    
    torch.testing.assert_close(
        relative_position_index_hash,
        expected_results,
        rtol=1e-5,      # Relative tolerance
        atol=1e-6,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"relative_position_index_hash do not match expected values.\nActual: {relative_position_index_hash}\nExpected: {expected_results}"
    )

def test_get_relative_position_index_hash_batch():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    query_translations = torch.tensor([
        [-2.8, -2.8, -2.8],
        [-2.8, -2.8, -2.8],
        [-2.8, -2.8, -2.8],
        [-2.8, -2.8, -2.8],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.0, -0.0, -0.0],
        [-0.0, -0.0, -0.0],
        [-0.0, -0.0, -0.0],
        [-0.0, -0.0, -0.0],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8]
    ], dtype=torch.float32, device='cuda')

    query_levels = torch.tensor([
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3
    ], dtype=torch.float32, device='cuda')
    
    expected_results = torch.tensor([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ], dtype=torch.float32, device='cuda')
    
    relative_position_index_hash = gaussians.get_relative_position_index_hash(query_levels, query_translations)
    
    torch.testing.assert_close(
        relative_position_index_hash,
        expected_results,
        rtol=1e-5,      # Relative tolerance
        atol=1e-6,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"relative_position_index_hash do not match expected values.\nActual: {relative_position_index_hash}\nExpected: {expected_results}"
    )


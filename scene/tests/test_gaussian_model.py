import pytest
import torch
from scene.gaussian_model import GaussianModel, get_gaussian_array, logging

def test_get_gaussian_arrary():
    raw_translations = torch.tensor([
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32, device='cuda')

    raw_lengths = torch.tensor([
        0.5, 1, 2
    ], dtype=torch.float32, device='cuda')

    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)

    gaussian_array = gaussians.create_unsorted_voxel_gaussian_array(raw_lengths, raw_translations)

    gaussian_array.assert_is_valid()

    expected_scalings = torch.tensor([
        0.5, 0.25
    ], dtype=torch.float32, device='cuda')

    new_gaussian_array = get_gaussian_array(gaussian_array, [1, 0])

    torch.testing.assert_close(
        torch.exp(new_gaussian_array.voxel_based_scales),
        expected_scalings,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
        msg=f"Voxel sphere radii do not match expected values. \nActual: {torch.exp(new_gaussian_array.voxel_based_scales)}\nExpected: {expected_scalings}"
    )
    

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
    expected_levels = torch.tensor([0, 0, 0, 0, 1, 1, 3, 3], dtype=torch.long, device='cuda')
    nearest_levels = gaussians.get_nearest_voxel_level(inputs)
    torch.testing.assert_close(
        nearest_levels,
        expected_levels,
        rtol=0,
        atol=0,
        equal_nan=True,
        msg=f"Nearest voxel levels do not match expected values.\nActual: {nearest_levels}\nExpected: {expected_levels}"
    )

def test_get_nearest_voxel_index_batch():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    query_translations = torch.tensor([
        [-2.8, -2.8, -2.8],
        [-2.1, -2.1, -2.1],
        [-0.8, -0.8, -0.8],
        [-0.0, -0.0, -0.0],
        [0.0, 0.0, 0.0],
        [0.8, 0.8, 0.8],
        [2.0, 2.0, 2.0],
        [2.8, 2.8, 2.8],
        [4.0, 4.0, 4.0],
        [4.8, 4.8, 4.8]
    ], dtype=torch.float32, device='cuda')
    
    query_levels = torch.zeros(query_translations.size(0), dtype=torch.long, device='cuda')
    
    expected_results = torch.tensor([
        [-2, -2, -2],
        [-2, -2, -2],
        [-1, -1, -1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2]
    ], dtype=torch.long, device='cuda')

    voxel_index = gaussians.get_nearest_voxel_index(query_levels, query_translations)
    
    torch.testing.assert_close(
        voxel_index,
        expected_results,
        rtol=0,      # Relative tolerance
        atol=0,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"Voxel translations do not match expected results. \n Actual: {voxel_index} \n expected: {expected_results}"
    )

def test_get_nearest_voxel_center():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    query_translations = torch.tensor([
        [-2.8, -2.8, -2.8],
        [-2.1, -2.1, -2.1],
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
        [-3.0, -3.0, -3.0],
        [-3.0, -3.0, -3.0],
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0],
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0]
    ], dtype=torch.float32, device='cuda')

    voxel_center = gaussians.get_nearest_voxel_center(query_levels, query_translations)
    
    torch.testing.assert_close(
        voxel_center,
        expected_results,
        rtol=1e-5,      # Relative tolerance
        atol=1e-6,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"Voxel translations do not match expected results. \n Actual: {voxel_center} \n expected: {expected_results}"
    )

def test_get_nearest_voxel_center2():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    query_translations = torch.tensor([
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8]
    ], dtype=torch.float32, device='cuda')
    
    query_levels = torch.tensor([
        0, 1, 2, 3
    ], dtype=torch.int32, device='cuda')
    
    expected_results = torch.tensor([
        [3.0, 3.0, 3.0],
        [2.5, 2.5, 2.5],
        [2.75, 2.75, 2.75],
        [2.875, 2.875, 2.875]
    ], dtype=torch.float32, device='cuda')

    voxel_center = gaussians.get_nearest_voxel_center(query_levels, query_translations)
    
    torch.testing.assert_close(
        voxel_center,
        expected_results,
        rtol=1e-5,      # Relative tolerance
        atol=1e-6,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"Voxel translations do not match expected results. \n Actual: {voxel_center} \n expected: {expected_results}"
    )

def test_get_relative_position_index_hash():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    query_translations = torch.tensor([
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8]
    ], dtype=torch.float32, device='cuda')

    query_levels = torch.tensor([
        0, 1, 2, 3
    ], dtype=torch.long, device='cuda')
    
    expected_results = torch.tensor([
        0, 100000000, 180000000, 188000000
    ], dtype=torch.long, device='cuda')
    
    relative_position_index_hash = gaussians.get_relative_position_index_hash(query_levels, query_translations)
    
    torch.testing.assert_close(
        relative_position_index_hash,
        expected_results,
        rtol=0,      # Relative tolerance
        atol=0,      # Absolute tolerance
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
        [-0.01, -0.01, -0.01],
        [-0.01, -0.01, -0.01],
        [-0.01, -0.01, -0.01],
        [-0.01, -0.01, -0.01],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [2.8, 2.8, 2.8]
    ], dtype=torch.float32, device='cuda')

    query_levels = torch.tensor([
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3
    ], dtype=torch.float32, device='cuda')
    
    expected_results = torch.tensor([
        0, 800000000, 810000000, 811000000, #
        0, 100000000, 110000000, 111000000, #
        0, 100000000, 110000000, 111000000, #
        0, 800000000, 880000000, 888000000, #
        0, 100000000, 180000000, 188000000  #
    ], dtype=torch.long, device='cuda')
    
    relative_position_index_hash = gaussians.get_relative_position_index_hash(query_levels, query_translations)
    
    torch.testing.assert_close(
        relative_position_index_hash,
        expected_results,
        rtol=0,      # Relative tolerance
        atol=0,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"relative_position_index_hash do not match expected values.\nActual: {relative_position_index_hash}\nExpected: {expected_results}"
    )

def test_get_relative_position_index_parent_level_batch_1():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    relative_index_position_hash = torch.tensor([
            0, 0, 0, 0, #
            800000000, 800000000, 800000000, 800000000, #
            810000000, 810000000, 810000000, 810000000, #
            811000000, 811000000, 811000000, 811000000  #
        ], dtype=torch.long, device='cuda')

    query_levels = torch.tensor([
        0, 1, 2, 3, #
        0, 1, 2, 3, #
        0, 1, 2, 3, #
        0, 1, 2, 3, #
    ], dtype=torch.long, device='cuda')
    
    expected_relative_position_index = torch.tensor([
        0, 0, 0, 0, #
        8, 0, 0, 0, #
        8, 1, 0, 0, #
        8, 1, 1, 0, #
    ], dtype=torch.long, device='cuda')
    
    
    relative_position_index = gaussians.get_relative_position_index_parent_level(relative_index_position_hash, query_levels)
    
    torch.testing.assert_close(
        relative_position_index,
        expected_relative_position_index,
        rtol=0,      # Relative tolerance
        atol=0,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"relative_position_index_hash do not match expected values.\nActual: {relative_position_index}\nExpected: {expected_relative_position_index}"
    )

def test_get_parent_relative_index_hash_batch_1():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
    
    relative_index_position_hash = torch.tensor([
            0, 0, 0, 0, #
            811000000, 811000000, 811000000, 811000000,  #
            123456788, 123456788, 123456788, 123456788
        ], dtype=torch.long, device='cuda')

    query_levels = torch.tensor([
        0, 1, 2, 3, #
        0, 1, 2, 3, #
        0, 1, 2, 3  #
    ], dtype=torch.long, device='cuda')
    
    expected_parent_relative_index_hash = torch.tensor([
        0, 0, 0, 0, #
        0, 800000000, 810000000, 811000000, #
        0, 100000000, 120000000, 123000000
    ], dtype=torch.long, device='cuda')
    
    
    parent_relative_index_hash = gaussians.get_parent_relative_index_hash(relative_index_position_hash, query_levels)
    
    torch.testing.assert_close(
        parent_relative_index_hash,
        expected_parent_relative_index_hash,
        rtol=0,      # Relative tolerance
        atol=0,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"parent_relative_index_hash do not match expected values.\nActual: {parent_relative_index_hash}\nExpected: {expected_parent_relative_index_hash}"
    )


def test_get_parent_relative_index_hash_batch_2():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=8)
    
    relative_index_position_hash = torch.tensor([
            123456788, 123456788, 123456788, 123456788, 123456788, 
            123456788, 123456788, 123456788, 123456788
        ], dtype=torch.long, device='cuda')

    query_levels = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ], dtype=torch.long, device='cuda')
    
    expected_parent_relative_index_hash = torch.tensor([
        0, 100000000, 120000000, 123000000, 123400000, 123450000, 123456000, 123456700, 123456780
    ], dtype=torch.long, device='cuda')
    
    
    parent_relative_index_hash = gaussians.get_parent_relative_index_hash(relative_index_position_hash, query_levels)
    
    torch.testing.assert_close(
        parent_relative_index_hash,
        expected_parent_relative_index_hash,
        rtol=0,      # Relative tolerance
        atol=0,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"parent_relative_index_hash do not match expected values.\nActual: {parent_relative_index_hash}\nExpected: {expected_parent_relative_index_hash}"
    )


def test_get_parent_relative_index_hash_batch_3():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=8)
    
    relative_index_position_hash = torch.tensor([
            123456700, 123456700, 123456700, 123456700, 123456700, 
            123456700, 123456700, 123456700, 123456700
        ], dtype=torch.long, device='cuda')

    query_levels = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ], dtype=torch.long, device='cuda')
    
    expected_parent_relative_index_hash = torch.tensor([
        0, 100000000, 120000000, 123000000, 123400000, 123450000, 123456000, 123456700, 123456700
    ], dtype=torch.long, device='cuda')
    
    
    parent_relative_index_hash = gaussians.get_parent_relative_index_hash(relative_index_position_hash, query_levels)
    
    logging.debug(f"parent_relative_index_hash{parent_relative_index_hash}")
    logging.debug(f"expected_parent_relative_index_hash{expected_parent_relative_index_hash}")

    torch.testing.assert_close(
        parent_relative_index_hash,
        expected_parent_relative_index_hash,
        rtol=0,      # Relative tolerance
        atol=0,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"parent_relative_index_hash do not match expected values.\nActual: {parent_relative_index_hash}\nExpected: {expected_parent_relative_index_hash}"
    )


def test_get_voxel_index_as_hash_batch():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)
        
    voxel_index = torch.tensor([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=torch.long, device='cuda')

    # 2^20 == 1048576, 2^10 = 1024
    expected_hash = torch.tensor([
            536346111, #
            537395712, #
            538445313  #
        ], dtype=torch.long, device='cuda')

    voxel_hash = gaussians.get_voxel_index_as_hash(voxel_index)
    
    torch.testing.assert_close(
        voxel_hash,
        expected_hash,
        rtol=0,      # Relative tolerance
        atol=0,      # Absolute tolerance
        equal_nan=True, # Whether to compare NaNs as equal
        msg=f"Voxel translations do not match expected results. \n Actual: {voxel_hash} \n expected: {expected_hash}"
    )

# All existing
def test_build_parent_child_relationships_batch_1():
    raw_translations = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
    ], dtype=torch.float32, device='cuda')

    raw_lengths = torch.tensor([
        2, 1, 1, 0.5, 0.5
    ], dtype=torch.float32, device='cuda')

    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)

    gaussian_array = gaussians.create_unsorted_voxel_gaussian_array(raw_lengths, raw_translations)

    gaussian_array = gaussians.sort_gaussians(gaussian_array)

    gaussians.build_parent_child_relationships(gaussian_array)

def test_build_parent_child_relationships_batch_2():
    logging.debug(
                f"test_build_parent_child_relationships_batch_2"
            )
    raw_translations = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
        [-0.25, -0.25, -0.25],
        [-0.75, -0.75, -0.75],
    ], dtype=torch.float32, device='cuda')

    raw_lengths = torch.tensor([
        2, 1, 1, 0.5, 0.5, 0.5, 0.5
    ], dtype=torch.float32, device='cuda')

    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)

    gaussian_array = gaussians.create_unsorted_voxel_gaussian_array(raw_lengths, raw_translations)

    gaussian_array = gaussians.sort_gaussians(gaussian_array)

    gaussians.build_parent_child_relationships(gaussian_array)

def test_remove_gaussians_and_descendants_1():
    logging.debug(
                f"remove_gaussians_and_descendants_1"
            )
    raw_translations = torch.tensor([
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
        [-0.25, -0.25, -0.25],
        [-0.75, -0.75, -0.75],
    ], dtype=torch.float32, device='cuda')

    raw_lengths = torch.tensor([
        2, 1, 0.5, 0.5, 0.5, 0.5
    ], dtype=torch.float32, device='cuda')

    indices_to_remove = torch.tensor([
        0
    ], dtype=torch.long, device='cuda')

    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=2, input_max_voxel_level=3)

    gaussian_array = gaussians.create_unsorted_voxel_gaussian_array(raw_lengths, raw_translations)

    gaussian_array = gaussians.sort_gaussians(gaussian_array)

    gaussian_array = gaussians.build_parent_child_relationships(gaussian_array)

    logging.debug(f"------ before remove_gaussians_and_descendants: ")

    gaussian_array = gaussians.remove_gaussians_and_descendants(gaussian_array, indices_to_remove)

    logging.debug(f"------ after remove_gaussians_and_descendants: ")
    gaussian_array.print_hash()
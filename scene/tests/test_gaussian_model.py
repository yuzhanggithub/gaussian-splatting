import pytest
from scene.gaussian_model import GaussianModel

def test_get_voxel_length():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=1, input_max_voxel_level=3)
    assert gaussians.get_voxel_length(0) == pytest.approx(1)
    assert gaussians.get_voxel_length(1) == pytest.approx(0.5)
    assert gaussians.get_voxel_length(2) == pytest.approx(0.25)
    assert gaussians.get_voxel_length(3) == pytest.approx(0.125)
    assert gaussians.get_voxel_length(4) == pytest.approx(0.125)

def test_get_voxel_length():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=1, input_max_voxel_level=3)
    ratio = 0.5
    assert  gaussians.get_voxel_sphere_radius(0) == ratio * gaussians.get_voxel_length(0)
    assert  gaussians.get_voxel_sphere_radius(1) == ratio * gaussians.get_voxel_length(1)
    assert  gaussians.get_voxel_sphere_radius(2) == ratio * gaussians.get_voxel_length(2)
    assert  gaussians.get_voxel_sphere_radius(3) == ratio * gaussians.get_voxel_length(3)
    assert  gaussians.get_voxel_sphere_radius(4) == ratio * gaussians.get_voxel_length(4)

def test_get_nearest_voxel_level():
    gaussians = GaussianModel(sh_degree=3, input_max_voxel_length=1, input_max_voxel_level=3)
    assert gaussians.get_nearest_voxel_level(100) == pytest.approx(0)
    assert gaussians.get_nearest_voxel_level(1.1) == pytest.approx(0)
    assert gaussians.get_nearest_voxel_level(1) == pytest.approx(0)
    assert gaussians.get_nearest_voxel_level(0.9) == pytest.approx(0)
    assert gaussians.get_nearest_voxel_level(0.6) == pytest.approx(0)
    assert gaussians.get_nearest_voxel_level(0.4) == pytest.approx(1)
    assert gaussians.get_nearest_voxel_level(0.125) == pytest.approx(3)
    assert gaussians.get_nearest_voxel_level(0.1) == pytest.approx(3)
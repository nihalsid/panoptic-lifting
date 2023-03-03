# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch

from util.camera import _radial_and_tangential_undistort


def create_grid(height, width):
    xs = torch.linspace(0, width - 1, width)
    ys = torch.linspace(0, height - 1, height)
    i, j = torch.meshgrid(xs, ys, indexing='ij')
    return i.t(), j.t()


def get_ray_directions(height, width, focal_length):
    i, j = create_grid(height, width)
    directions = torch.stack([
        (i - width / 2) / focal_length,
        -(j - height / 2) / focal_length,
        -torch.ones_like(i)
    ], -1)
    return directions


def get_ray_directions_with_intrinsics(height, width, intrinsics):
    i, j = create_grid(height, width)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    directions = torch.stack([
        (i - cx) / fx, (j - cy) / fy, torch.ones_like(i)
    ], -1)
    return directions


def get_ray_directions_with_intrinsics_undistorted(height, width, intrinsics, distortion_params):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    i, j = create_grid(height, width)
    x = (i - cx) / fx
    y = (j - cy) / fy
    x, y = _radial_and_tangential_undistort(x, y, distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3])
    directions = torch.stack([
        x, y, torch.ones_like(i)
    ], -1)
    return directions


def get_rays(directions, cam2world):
    rays_d = directions @ cam2world[:3, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = cam2world[:3, 3].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(height, width, focal_length, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (width / (2. * focal_length)) * ox_oz
    o1 = -1. / (height / (2. * focal_length)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (width / (2. * focal_length)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (height / (2. * focal_length)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def rays_intersect_sphere(rays_o, rays_d, r=1):
    """
    Solve for t such that a=ro+trd with ||a||=r
    Quad -> r^2 = ||ro||^2 + 2t (ro.rd) + t^2||rd||^2
    -> t = (-b +- sqrt(b^2 - 4ac))/(2a) with
       a = ||rd||^2
       b = 2(ro.rd)
       c = ||ro||^2 - r^2
       => (forward intersection) t= (sqrt(D) - (ro.rd))/||rd||^2
       with D = (ro.rd)^2 - (r^2 - ||ro||^2)
    """
    odotd = torch.sum(rays_o * rays_d, 1)
    d_norm_sq = torch.sum(rays_d ** 2, 1)
    o_norm_sq = torch.sum(rays_o ** 2, 1)
    determinant = odotd ** 2 + (r ** 2 - o_norm_sq) * d_norm_sq
    assert torch.all(
        determinant >= 0
    ), "Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!"
    return (torch.sqrt(determinant) - odotd) / d_norm_sq

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np
import torch
from einops import repeat

from util.transforms import trs_comp, dot


def frustum_world_bounds(dims, intrinsics, cam2worlds, max_depth, form='bbox'):
    """Compute bounds defined by the frustum provided cameras
    Args:
        dims (N,2): heights,widths of cameras
        intrinsics (N,3,3): intrinsics (unnormalized, hence HW required)
        cam2worlds (N,4,4): camera to world transformations
        max_depth (float): depth of all frustums
        form (str): bbox: convex bounding box, sphere: convex bounding sphere
    """
    # unproject corner points
    h_img_corners = torch.Tensor([[0, 1, 1], [1, 0, 1], [1, 1, 1]])
    intrinsics_inv = torch.linalg.inv(intrinsics[:, [1, 0, 2]])  # K in WH -> convert to HW
    k = len(h_img_corners)
    n = len(dims)
    rep_HWds = repeat(torch.cat([dims, torch.ones((n, 1))], 1), "n c -> n k c", k=k)
    skel_pts = rep_HWds * repeat(h_img_corners, "k c -> n k c", n=n)  # (N,K,(hwd))
    corners_cam_a = torch.einsum("nkij,nkj->nki", repeat(intrinsics_inv, "n x y -> n k x y", k=k), skel_pts) * max_depth
    corners_cam_b = torch.einsum("nkij,nkj->nki", repeat(intrinsics_inv, "n x y -> n k x y", k=k), skel_pts) * 0.01
    # nihalsid: adding corner with max depth, and corners with min depth
    corners_cam = torch.cat([corners_cam_a, corners_cam_b], 0)
    corners_cam_h = torch.cat([corners_cam, torch.ones(corners_cam.shape[0], corners_cam.shape[1], 1)], -1)

    corners_world_h = torch.einsum("nij,nkj->nki", cam2worlds.repeat(2, 1, 1), corners_cam_h)
    corners_world_flat = corners_world_h.reshape(-1, 4)[:, :3]
    if form == 'bbox':
        bounds = torch.stack([corners_world_flat.min(0).values, corners_world_flat.max(0).values])
        return bounds
    elif form == 'sphere':
        corners_world_center = torch.mean(corners_world_flat, 0)
        sphere_radius = torch.max(torch.norm((corners_world_flat - corners_world_center), dim=1))

        # todo: remove visualization
        ##############################
        # from util.misc import visualize_points
        # import trimesh
        # visualize_points(corners_world_flat, "runs/world_flat.obj")
        # sphere = trimesh.creation.icosphere(radius=sphere_radius.numpy())
        # sphere.apply_translation(corners_world_center)
        # sphere.export("runs/world_bounds_sphere.obj")
        ##############################

        return corners_world_center, sphere_radius
    else:
        raise Exception("Not implemented yet: Ellipsoid for example")


def compute_world2normscene(dims, intrinsics, cam2worlds, max_depth, rescale_factor=1.0):
    """Compute transform converting world to a normalized space enclosing all
    cameras frustums (given depth) into a unit sphere
    Note: max_depth=0 -> camera positions only are contained (like NeRF++ does it)

    Args:
        dims (N,2): heights,widths of cameras
        intrinsics (N,3,3): intrinsics (unnormalized, hence HW required)
        cam2worlds (N,4,4): camera to world transformations
        max_depth (float): depth of all frustums
        rescale_factor (float)>=1.0: factor to scale the world space even further so no camera is too close to the unit sphere surface
    """
    assert rescale_factor >= 1.0, "prevent cameras outside of unit sphere"

    sphere_center, sphere_radius = frustum_world_bounds(dims, intrinsics, cam2worlds, max_depth, 'sphere')  # sphere containing frustums
    world2nscene = trs_comp(-sphere_center / (rescale_factor * sphere_radius), torch.eye(3), 1 / (rescale_factor * sphere_radius))

    return world2nscene


def depth_to_distance(depth, intrinsics):
    uv = np.stack(np.meshgrid(
        np.arange(depth.shape[1]),
        np.arange(depth.shape[0])
    ), -1).reshape(-1, 2)
    depth = depth.reshape(-1)
    uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)
    return depth * np.linalg.norm((np.linalg.inv(intrinsics) @ uvh.T).T, axis=1)


def distance_to_depth(K, dist, uv=None):
    if uv is None and len(dist.shape) >= 2:
        # create mesh grid according to d
        uv = np.stack(np.meshgrid(np.arange(dist.shape[1]), np.arange(dist.shape[0])), -1)
        uv = uv.reshape(-1, 2)
        dist = dist.reshape(-1)
        if not isinstance(dist, np.ndarray):
            uv = torch.from_numpy(uv).to(dist)
    if isinstance(dist, np.ndarray):
        # z * np.sqrt(x_temp**2+y_temp**2+z_temp**2) = dist
        uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)
        temp_point = dot(np.linalg.inv(K), uvh)
        z = dist / np.linalg.norm(temp_point, axis=1)

    else:
        uvh = torch.cat([uv, torch.ones(len(uv), 1).to(uv)], -1)
        temp_point = dot(torch.inverse(K), uvh)
        z = dist / torch.linalg.norm(temp_point, dim=1)
    return z


def unproject_2d_3d(cam2world, intrinsics, depth, dims):
    uv = np.stack(np.meshgrid(np.arange(dims[0]), np.arange(dims[1])), -1)
    uv = torch.from_numpy(uv.reshape(-1, 2))
    uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)
    cam_point = (torch.linalg.inv(intrinsics) @ torch.from_numpy(uvh).float().T).T * depth[:, None]
    world_point = (cam2world[:3, :3] @ cam_point.T).T + cam2world[:3, 3]
    return world_point


def project_3d_2d(cam2world, K, world_point, with_dist=False, discrete=True, round=True):
    if isinstance(world_point, np.ndarray):
        cam_point = dot(np.linalg.inv(cam2world), world_point)
        point_dist = np.sqrt((cam_point ** 2).sum(-1))
        img_point = dot(K, cam_point)
        uv_point = img_point[:, :2] / img_point[:, 2][:, None]
        if discrete:
            if round:
                uv_point = np.round(uv_point)
            uv_point = uv_point.astype(np.int)
        if with_dist:
            return uv_point, img_point[:, 2], point_dist
        return uv_point
    else:
        cam_point = dot(torch.inverse(cam2world), world_point)
        point_dist = (cam_point ** 2).sum(-1).sqrt()
        img_point = dot(K, cam_point)
        uv_point = img_point[:, :2] / img_point[:, 2][:, None]
        if discrete:
            if round:
                uv_point = torch.round(uv_point)
                uv_point = uv_point.int()
        if with_dist:
            return uv_point, img_point[:, 2], point_dist

        return uv_point


def auto_orient_poses(poses, method="up"):
    """Orients and centers the poses. We provide two methods for orientation: pca and up.
    pca: Orient the poses so that the principal component of the points is aligned with the axes.
        This method works well when all of the cameras are in the same plane.
    up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    Args:
        poses: The poses to orient.
        method: The method to use for orientation. Either "pca" or "up".
    Returns:
        The oriented poses.
    borrowed from from nerfstudio
    """
    translation = poses[..., :3, 3]

    mean_translation = torch.mean(translation, dim=0)
    translation = translation - mean_translation

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation.T @ translation)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -mean_translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(axis=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method == "up":
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -mean_translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

    return oriented_poses


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.
    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    borrowed from from nerfstudio
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))


def _compute_residual_and_jacobian(x, y, xd, yd, k1=0.0, k2=0.0, k3=0.0, k4=0.0, p1=0.0, p2=0.0):
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = (k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4)))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(xd, yd, k1=0, k2=0, k3=0, k4=0, p1=0, p2=0, eps=1e-9, max_iterations=10):
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    xd = xd.numpy()
    yd = yd.numpy()
    x = np.copy(xd)
    y = np.copy(yd)

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(
            np.abs(denominator) > eps, x_numerator / denominator,
            np.zeros_like(denominator))
        step_y = np.where(
            np.abs(denominator) > eps, y_numerator / denominator,
            np.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return torch.from_numpy(x), torch.from_numpy(y)

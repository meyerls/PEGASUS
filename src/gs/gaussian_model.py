#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
import torch
import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement
import einops
from einops import einsum

import open3d as o3d
from scipy.spatial.transform import Rotation
from e3nn import o3
import sphecerix

import sys
sys.path.append("../../gaussian-splatting")


from simple_knn._C import distCUDA2

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.sh_utils import SH2RGB, RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class GaussianModelBase:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
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
        (self.active_sh_degree,
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
         self.spatial_lr_scale) = model_args
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
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # Computes the mean distance to the 3 nearest neighbour for every point in the point cloud!
        # From the paper: We estimate the initial covariance matrix as an isotropic gaussian with
        #                 axes equal to the mean of the distance to the closest three points.
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # Scale of the isotropic gaussian
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # Set initial rotation of gaussians in quaternions
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Why do we use inverse sigmoid here? TODO: find out!
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, clean_pcd: bool = False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        if clean_pcd:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            cl, ind = pcd.remove_radius_outlier(16, 0.03)
            # o3d.visualization.draw_geometries([cl])
            indecies = ind
        else:
            indecies = list(range(0, xyz.shape[0]))

        xyz = xyz[indecies]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis][indecies]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])[indecies]
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])[indecies]
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])[indecies]

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])[indecies]
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])[indecies]

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])[indecies]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

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
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        # std of splats
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        # init mean value for all 3d gauss dist with (0,0,0)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        # sample points according to stds and mean just computed
        samples = torch.normal(mean=means, std=stds)
        # take identical rotation from original splats
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        # compute new position to be added
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # we just added N new points to the gaussian point cloud. One of this points is the original point. So we
        # simply delete the original one
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1


class GaussianModel(GaussianModelBase):
    def __init__(self, sh_degree):
        super().__init__(sh_degree)

    def get_point_cloud(self):
        features_dc = self._features_dc.clone()
        f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        base_color = SH2RGB(f_dc).clip(0, 1)  # ToDo: is this correct?

        xyz = self._xyz.clone()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(base_color)

        return pcd

    def save_ply(self, path):
        super().save_ply(path=path)

        pcd = self.get_point_cloud()
        o3d.io.write_point_cloud(os.path.join(os.path.dirname(path), 'point_cloud_o3d.ply'), pcd)

    def apply_translation_on_xyz(self, t):
        self._xyz = self._xyz + t.to('cuda').type(torch.float32)

    def apply_rotation_on_xyz(self, R, origin=False):
        new_xyz = self._xyz
        if not origin:
            mean_xyz = torch.mean(new_xyz, 0)
            new_xyz = new_xyz - mean_xyz
            new_xyz = (R @ new_xyz.T).T
            self._xyz = new_xyz + mean_xyz
        elif origin:
            self._xyz = (R @ new_xyz.T).T

    def apply_transformation_on_xyz(self, T):
        self.apply_rotation_on_xyz(R=T[:3, :3])
        self.apply_translation_on_xyz(t=T[:3, 3])

    def apply_rotation_on_splats(self, R):
        q = torch.from_numpy(Rotation.from_quat(self._rotation.detach().cpu().numpy()).as_quat()).to(
            'cuda').type(torch.float32)
        splat_rotation_matrix = build_rotation(q)
        splat_rotated = R @ splat_rotation_matrix
        splat_rotated_quat = np.roll(Rotation.from_matrix(splat_rotated.detach().cpu().numpy()).as_quat(), 1, axis=-1)
        self._rotation = torch.from_numpy(splat_rotated_quat).to('cuda').type(torch.float32)

    def apply_rotation_on_sh(self, R):
        with torch.no_grad():
            P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
            permuted_rotation_matrix = np.linalg.inv(P) @ R.cpu().numpy() @ P
            rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))

            # Construction coefficient
            D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
            D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
            D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

            one_degree_shs = self._features_rest[:, [0, 1, 2]]
            one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            one_degree_shs = einsum(
                D_1.to(R.device).to(torch.float),
                one_degree_shs,
                "... i j, ... j -> ... i",
            )
            one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            self._features_rest[:, [0, 1, 2]] = one_degree_shs

            two_degree_shs = self._features_rest[:, [3, 4, 5, 6, 7]]
            two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            two_degree_shs = einsum(
                D_2.to(R.device).to(torch.float),
                two_degree_shs,
                "... i j, ... j -> ... i",
            )
            two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            self._features_rest[:, [3, 4, 5, 6, 7]] = two_degree_shs

            three_degree_shs = self._features_rest[:, [8, 9, 10, 11, 12, 13, 14]]
            three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            three_degree_shs = einsum(
                D_3.to(R.device).to(torch.float),
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
            three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            self._features_rest[:, [8, 9, 10, 11, 12, 13, 14]] = three_degree_shs

    def apply_rotation_on_sh_old(self, R):
        Robj = Rotation.from_matrix(R.cpu())
        # D_0 = sphecerix.tesseral_wigner_D(1, Robj)

        # dc = self._features_dc
        # Y_0 = dc.cpu().numpy()
        # Y_0_rotated = D_0 @ Y_0
        with torch.no_grad():
            aby = torch.asarray(Robj.as_euler('yzy')).type(torch.float32) # alpha , beta, gamma  # o3._rotation.matrix_to_angles(R.cpu())

            # D_1 = torch.from_numpy(sphecerix.tesseral_wigner_D(1, Robj)).to('cuda').type(torch.float32)
            D_1 = o3.wigner_D(1, aby[2], aby[1], aby[0]).to('cuda').type(torch.float32)
            Y_1 = self._features_rest[:, [0, 1, 2]]
            Y_1_rotated = torch.matmul(D_1, Y_1)  # torch.matmul(D_1, Y_1)  # torch.matmul(R, Y_1)
            # Y_1_rotated.requires_grad = True
            self._features_rest[:, [0, 1, 2]] = Y_1_rotated

            # D_2 = torch.from_numpy(sphecerix.tesseral_wigner_D(2, Robj)).to('cuda').type(torch.float32)
            D_2 = o3.wigner_D(2, aby[2], aby[1], aby[0]).to('cuda').type(torch.float32)
            Y_2 = self._features_rest[:, [3, 4, 5, 6, 7]]
            Y_2_rotated = torch.matmul(D_2, Y_2)
            # Y_2_rotated.requires_grad = True
            self._features_rest[:, [3, 4, 5, 6, 7]] = Y_2_rotated

            # D_3 = torch.from_numpy(sphecerix.tesseral_wigner_D(3, Robj)).to('cuda').type(torch.float32)
            D_3 = o3.wigner_D(3, aby[2], aby[1], aby[0]).to('cuda').type(torch.float32)
            Y_3 = self._features_rest[:, [8, 9, 10, 11, 12, 13, 14]]
            Y_3_rotated = torch.matmul(D_3, Y_3)
            # Y_3_rotated.requires_grad = True
            self._features_rest[:, [8, 9, 10, 11, 12, 13, 14]] = Y_3_rotated

    def apply_transformation(self, T):
        self.apply_transformation_on_xyz(T=T)
        self.apply_rotation_on_splats(R=T[:3, :3])
        self.apply_rotation_on_sh(R=T[:3, :3])

    def merge_gaussians(self, gaussian):

        self._xyz = torch.vstack((self._xyz, gaussian._xyz))
        self._features_dc = torch.vstack((self._features_dc, gaussian._features_dc))
        self._features_rest = torch.vstack((self._features_rest, gaussian._features_rest))
        self._opacity = torch.vstack((self._opacity, gaussian._opacity))
        self._scaling = torch.vstack((self._scaling, gaussian._scaling))
        self._rotation = torch.vstack((self._rotation, gaussian._rotation))

        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        # self.denom = self.denom[valid_points_mask]
        # = self.max_radii2D[valid_points_mask]

    def mask_points(self, mask):
        if self.optimizer:
            optimizable_tensors = self._prune_optimizer(mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
        else:
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._opacity = self._opacity[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]

        if self.xyz_gradient_accum.__len__() != 0:
            self.xyz_gradient_accum = self.xyz_gradient_accum[mask]

        if self.denom.__len__() != 0:
            self.denom = self.denom[mask]

        if self.max_radii2D.__len__() != 0:
            self.max_radii2D = self.max_radii2D[mask]

    def translate_selected_points(self, mask, t):
        translation = torch.zeros(self._xyz.shape, device=t.device)
        translation[mask] = + t
        self._xyz = self._xyz + translation

        # self.denom = self.denom[mask]
        # self.max_radii2D = self.max_radii2D[mask]

    def denoise_point_cloud(self, nb_points=16, radius=0.05, debug=False):
        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                              zoom=0.3412,
                                              front=[0.4257, -0.2125, -0.8795],
                                              lookat=[2.6172, 2.0475, 1.532],
                                              up=[-0.0694, -0.9768, 0.2024])

        pcd = self.get_point_cloud()
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        mask = np.full(self.get_xyz.shape[0], False)
        mask[ind] = True
        self.mask_points(torch.asarray(mask).to(self.get_xyz.device))

        if debug:
            display_inlier_outlier(pcd, ind)

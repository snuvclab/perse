# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import numpy as np
from .lbs import *
import pickle
from utils.hutils import is_dist_avail_and_initialized, save_on_master, get_rank, is_main_process


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)



class FLAME(nn.Module):
    def __init__(self, conf, flame_model_path, lmk_embedding_path, n_shape, n_exp, canonical_expression, canonical_pose): 
        super(FLAME, self).__init__()
        if is_main_process():
            print("creating the FLAME Decoder")
        self.dtype = torch.float32
        with open(flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)
        # begin: for plotting landmarks, to make sure that camera is correct. This is so hard to debug otherwise...
        lmk_embeddings = np.load(lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long();
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long())
        self.register_buffer('lmk_bary_coords',
                             torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long())
        self.register_buffer('full_lmk_bary_coords',
                             torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype))

        neck_kin_chain = []; NECK_IDX=1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

        # end: for plotting landmarks, to make sure that camera is correct. This is so hard to debug otherwise...

        factor = 4

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template) * factor, dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs * factor)
        canonical_expression = canonical_expression[..., :n_exp]
        if is_main_process():
            print("Canonical expression: ", canonical_expression)
        # self.v_template = self.v_template + torch.einsum('bl,mkl->bmk', [shape_params.cpu(), self.shapedirs[:, :, :n_shape]]).squeeze(0)          # NOTE shape_params 삭제


        # NOTE buffer로 등록시켜서 checkpoint에 자동으로 넣는 방법. learnable은 아니라고 한다. nn.Parameter만 learnable하다고 하고.
        # 2023.09.22. 22:10 
        # learning은 안되지만 pytorch lightning에서 cuda로 자동으로 넘겨주기 때문에 register로 등록해야할 거 같다.
        # model state로 등록한다고 해서 특별히 문제가 될 사항은 보이지 않는다.
        # NOTE custom ####################################
        canonical_pose_buffer = torch.zeros(1, 15).float()
        canonical_pose_buffer[:, 6] = canonical_pose
        self.register_buffer('canonical_pose', canonical_pose_buffer)
        self.register_buffer('canonical_exp', canonical_expression.float())
        self.register_buffer('canonical_beta', torch.zeros(1, n_shape).float())       # NOTE n_shape: 100
        ###################################################

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs) * factor, dtype=self.dtype))
        # 
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        self.n_shape = n_shape
        self.n_exp = n_exp

        # NOTE custom code: register_buffer를 이용해서 cuda 할당이 자동으로 되도록 수정해본다.
        canonical_verts, canonical_pose_feature, canonical_transformations = self(expression_params=self.canonical_exp, full_pose=self.canonical_pose, shape_params=self.canonical_beta)          # NOTE shape_params 추가
        canonical_verts = canonical_verts.squeeze(0)
        self.register_buffer('canonical_verts', canonical_verts)
        self.register_buffer('canonical_pose_feature', canonical_pose_feature)
        if conf.get_bool('model.deformer_network.ghostbone'):
            # NOTE cuda였으나 type_as로 변경
            # self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().type_as(self.FLAMEServer.canonical_transformations), self.FLAMEServer.canonical_transformations], 1)
            canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float(), canonical_transformations], 1)
        self.register_buffer('canonical_transformations', canonical_transformations)

    # FLAME mesh morphing
    def forward(self, expression_params, full_pose, shape_params):                                              
        """
            Input:
                expression_params: N X number of expression parameters
                full_pose: N X number of pose parameters (15)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = expression_params.shape[0]
        betas = torch.cat([shape_params.to(expression_params.device), expression_params[:, :self.n_exp]], dim=1)         
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, pose_feature, transformations = lbs(betas, full_pose, template_vertices,
                                                      self.shapedirs, self.posedirs,
                                                      self.J_regressor, self.parents,
                                                      self.lbs_weights, dtype=self.dtype)

        return vertices, pose_feature, transformations

    def forward_pts(self, pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32, mask=None, beta_shapedirs=None, shapes=None):
        # NOTE pnts_c는 pnts_flame이다.
        assert len(pnts_c.shape) == 2
        if mask is not None:
            pnts_c = pnts_c[mask]
            betas = betas[mask]
            transformations = transformations[mask]
            pose_feature = pose_feature[mask]
        num_points = pnts_c.shape[0]
        if shapedirs.shape[-1] > self.n_exp:
            canonical_exp = torch.cat([self.canonical_exp, torch.zeros(1, shapedirs.shape[-1] - self.n_exp).to(self.canonical_exp.device)], dim=1)            # NOTE .cuda()였으나 type_as로 바꿈
        else:
            canonical_exp = self.canonical_exp
        pnts_c_original = inverse_pts_shapecancel(pnts_p=pnts_c, 
                                                  betas=canonical_exp.expand(num_points, -1), 
                                                  transformations=self.canonical_transformations.expand(num_points, -1, -1, -1), 
                                                  shapes=self.canonical_beta.expand(num_points, -1),
                                                  pose_feature=self.canonical_pose_feature.expand(num_points, -1), 
                                                  shapedirs=shapedirs, 
                                                  posedirs=posedirs, 
                                                  beta_shapedirs=beta_shapedirs, 
                                                  lbs_weights=lbs_weights, 
                                                  dtype=dtype)
        pnts_p = forward_pts_shapecancel(pnts_c=pnts_c_original, 
                                         betas=betas, 
                                         transformations=transformations, 
                                         shapes=shapes,
                                         pose_feature=pose_feature, 
                                         shapedirs=shapedirs, 
                                         posedirs=posedirs, 
                                         beta_shapedirs=beta_shapedirs,
                                         lbs_weights=lbs_weights, 
                                         dtype=dtype)
        return pnts_p

    # below is just for plotting flame landmarks, code borrowed from DECA: https://github.com/yfeng95/DECA

    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks
            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:dd2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(vertices.device) * num_verts        # NOTE .to(device=vertices.device)였으나 type_as로 바꿈

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                         self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def find_landmarks(self, vertices, full_pose):

        batch_size = vertices.shape[0]

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(full_pose, 
                                                                                        self.dynamic_lmk_faces_idx,
                                                                                        self.dynamic_lmk_bary_coords,
                                                                                        self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor,
                                         lmk_faces_idx,
                                         lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(bz, 1),
                                         self.full_lmk_bary_coords.repeat(bz, 1, 1))
        return landmarks2d, landmarks3d



# class FLAME_ShapeCancel(nn.Module):
#     """
#     NOTE
#     * 23.10.01. 17:48 FLAME_lightning코드를 기반으로 수정했음. 
#     * beta cancel을 적용시켜서 변형시키는게 목표다. 
#     * 기존 코드에서 뭐가 바뀌었는지 전부 표시해두었다. 
#     """
#     # def __init__(self, conf, flame_model_path, lmk_embedding_path, n_shape, n_exp, shape_params, canonical_expression, canonical_pose):
#     def __init__(self, conf, flame_model_path, lmk_embedding_path, n_shape, n_exp, canonical_expression, canonical_pose):                   # NOTE shape_params 삭제
#         super(FLAME_ShapeCancel, self).__init__()
#         print("creating the FLAME Decoder")
#         self.dtype = torch.float32
#         with open(flame_model_path, 'rb') as f:
#             ss = pickle.load(f, encoding='latin1')
#             flame_model = Struct(**ss)
#         # begin: for plotting landmarks, to make sure that camera is correct. This is so hard to debug otherwise...
#         lmk_embeddings = np.load(lmk_embedding_path, allow_pickle=True, encoding='latin1')
#         lmk_embeddings = lmk_embeddings[()]
#         parents = to_tensor(to_np(flame_model.kintree_table[0])).long();
#         parents[0] = -1
#         self.register_buffer('parents', parents)

#         self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long())
#         self.register_buffer('lmk_bary_coords',
#                              torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype))
#         self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long())
#         self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype))
#         self.register_buffer('full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long())
#         self.register_buffer('full_lmk_bary_coords',
#                              torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype))

#         neck_kin_chain = []; NECK_IDX=1
#         curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
#         while curr_idx != -1:
#             neck_kin_chain.append(curr_idx)
#             curr_idx = self.parents[curr_idx]
#         self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

#         # end: for plotting landmarks, to make sure that camera is correct. This is so hard to debug otherwise...

#         factor = 4

#         self.dtype = torch.float32
#         self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
#         # The vertices of the template model
#         self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template) * factor, dtype=self.dtype))
#         # The shape components and expression
#         shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
#         shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+n_exp]], 2)
#         self.register_buffer('shapedirs', shapedirs * factor)
#         canonical_expression = canonical_expression[..., :n_exp]
#         print("Canonical expression: ", canonical_expression)
#         # self.v_template = self.v_template + torch.einsum('bl,mkl->bmk', [shape_params.cpu(), self.shapedirs[:, :, :n_shape]]).squeeze(0)          # NOTE shape_params 삭제

#         # NOTE custom ####################################
#         # self.canonical_pose = torch.zeros(1, 15).float()            # NOTE .cuda() 삭제
#         # self.canonical_pose[:, 6] = canonical_pose

#         # self.canonical_exp = canonical_expression.float()           # NOTE .cuda() 삭제
#         ###################################################

#         # NOTE buffer로 등록시켜서 checkpoint에 자동으로 넣는 방법. learnable은 아니라고 한다. nn.Parameter만 learnable하다고 하고.
#         # 2023.09.22. 22:10 
#         # learning은 안되지만 pytorch lightning에서 cuda로 자동으로 넘겨주기 때문에 register로 등록해야할 거 같다.
#         # model state로 등록한다고 해서 특별히 문제가 될 사항은 보이지 않는다.
#         # NOTE custom ####################################
#         canonical_pose_buffer = torch.zeros(1, 15).float()
#         canonical_pose_buffer[:, 6] = canonical_pose
#         self.register_buffer('canonical_pose', canonical_pose_buffer)
#         self.register_buffer('canonical_exp', canonical_expression.float())
#         self.register_buffer('canonical_beta', torch.zeros(1, n_shape).float())       # NOTE n_shape: 100
#         ###################################################

#         # The pose components
#         num_pose_basis = flame_model.posedirs.shape[-1]
#         posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
#         self.register_buffer('posedirs', to_tensor(to_np(posedirs) * factor, dtype=self.dtype))
#         # 
#         self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
#         parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
#         self.register_buffer('parents', parents)
#         self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

#         default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
#         self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
#                                                          requires_grad=False))
#         default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
#         self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
#                                                           requires_grad=False))

#         self.n_shape = n_shape
#         self.n_exp = n_exp

#         # NOTE custom code: register_buffer를 이용해서 cuda 할당이 자동으로 되도록 수정해본다.
#         canonical_verts, canonical_pose_feature, canonical_transformations = self(expression_params=self.canonical_exp, full_pose=self.canonical_pose, shape_params=self.canonical_beta)          # NOTE shape_params 추가
#         canonical_verts = canonical_verts.squeeze(0)
#         self.register_buffer('canonical_verts', canonical_verts)
#         self.register_buffer('canonical_pose_feature', canonical_pose_feature)
#         if conf.get_bool('model.deformer_network.ghostbone'):
#             # NOTE cuda였으나 type_as로 변경
#             # self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().type_as(self.FLAMEServer.canonical_transformations), self.FLAMEServer.canonical_transformations], 1)
#             canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float(), canonical_transformations], 1)
#         self.register_buffer('canonical_transformations', canonical_transformations)


#     # FLAME mesh morphing
#     # def forward(self, expression_params, full_pose):
#     def forward(self, expression_params, full_pose, shape_params):                                                              # NOTE shape_params 추가
#         """
#             Input:
#                 expression_params: N X number of expression parameters
#                 full_pose: N X number of pose parameters (15)
#             return:d
#                 vertices: N X V X 3
#                 landmarks: N X number of landmarks X 3
#         """
#         batch_size = expression_params.shape[0]
#         # NOTE 원래는 .to(expression_params.device)였으나 lightning에서 type_as로 하라고 해서 이렇게 쓴다.
#         betas = torch.cat([shape_params.to(expression_params.device), expression_params[:, :self.n_exp]], dim=1)                  # NOTE shape_params 추가
#         template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
#         vertices, pose_feature, transformations = lbs_batch.lbs(betas, full_pose, template_vertices,
#                                                       self.shapedirs, self.posedirs,
#                                                       self.J_regressor, self.parents,
#                                                       self.lbs_weights, dtype=self.dtype)

#         return vertices, pose_feature, transformations


#     def forward_pts(self, pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32, mask=None, beta_shapedirs=None, shapes=None):
#         # NOTE pnts_c는 pnts_flame이다.
#         # assert len(pnts_c.shape) == 2
#         if mask is not None:
#             pnts_c = pnts_c[mask]
#             betas = betas[mask]
#             transformations = transformations[mask]
#             pose_feature = pose_feature[mask]
#         num_points = pnts_c.shape[-1]
#         batch_size = pnts_c.shape[0]
#         if shapedirs.shape[2] > self.n_exp:         # torch.cat([self.canonical_exp, torch.zeros(1, shapedirs.shape[-1] - self.n_exp).cuda()], dim=1)
#             canonical_exp = torch.cat([self.canonical_exp, torch.zeros(1, shapedirs.shape[1] - self.n_exp).to(self.canonical_exp.device)], dim=1)            # NOTE .cuda()였으나 type_as로 바꿈
#         else:
#             canonical_exp = self.canonical_exp
        
#         pnts_c_original = lbs_batch.inverse_pts_shapecancel(pnts_p=pnts_c, 
#                                                   betas=canonical_exp[..., None].expand(batch_size, -1, num_points), 
#                                                   transformations=self.canonical_transformations[..., None].expand(batch_size, -1, -1, -1, num_points), 
#                                                   shapes=self.canonical_beta[..., None].expand(batch_size, -1, num_points),
#                                                   pose_feature=self.canonical_pose_feature[..., None].expand(batch_size, -1, num_points), 
#                                                   shapedirs=shapedirs, 
#                                                   posedirs=posedirs, 
#                                                   beta_shapedirs=beta_shapedirs, 
#                                                   lbs_weights=lbs_weights, 
#                                                   dtype=dtype)
#         pnts_p = lbs_batch.forward_pts_shapecancel(pnts_c=pnts_c_original, 
#                                          betas=betas, 
#                                          transformations=transformations, 
#                                          shapes=shapes,
#                                          pose_feature=pose_feature, 
#                                          shapedirs=shapedirs, 
#                                          posedirs=posedirs, 
#                                          beta_shapedirs=beta_shapedirs,
#                                          lbs_weights=lbs_weights, 
#                                          dtype=dtype)
        
#         # pnts_c_original_list = []
#         # for idx in range(num_points):
#         #     pnts_c_original_no_batch = lbs.inverse_pts_shapecancel(pnts_p=pnts_c[..., idx], 
#         #                                                 betas=canonical_exp.expand(1, -1), 
#         #                                                 transformations=self.canonical_transformations, 
#         #                                                 shapes=self.canonical_beta,
#         #                                                 pose_feature=self.canonical_pose_feature, 
#         #                                                 shapedirs=shapedirs[..., idx], 
#         #                                                 posedirs=posedirs[..., idx], 
#         #                                                 beta_shapedirs=beta_shapedirs[..., idx], 
#         #                                                 lbs_weights=lbs_weights[..., idx], 
#         #                                                 dtype=dtype)
#         #     pnts_c_original_list.append(pnts_c_original_no_batch.unsqueeze(-1))
#         # pnts_c_original_list = torch.cat(pnts_c_original_list, dim=-1)
        
#         # pnts_p_no_batch_list = []
#         # for idx in range(num_points):
#         #     pnts_p_no_batch = lbs.forward_pts_shapecancel(pnts_c=pnts_c_original[..., idx], 
#         #                                                 betas=betas[..., idx], 
#         #                                                 transformations=transformations[..., idx], 
#         #                                                 shapes=shapes[..., idx],
#         #                                                 pose_feature=pose_feature[..., idx], 
#         #                                                 shapedirs=shapedirs[..., idx], 
#         #                                                 posedirs=posedirs[..., idx], 
#         #                                                 beta_shapedirs=beta_shapedirs[..., idx],
#         #                                                 lbs_weights=lbs_weights[..., idx], 
#         #                                                 dtype=dtype)
#         #     pnts_p_no_batch_list.append(pnts_p_no_batch.unsqueeze(-1))
#         # pnts_p_no_batch_list = torch.cat(pnts_p_no_batch_list, dim=-1)
#         # pnts_c_original[:, 2, :] - pnts_c_original_list[:, 2, :]
#         # pnts_p[:, 2, :] - pnts_p_no_batch_list[:, 2, :]
#         return pnts_p

#     # below is just for plotting flame landmarks, code borrowed from DECA: https://github.com/yfeng95/DECA

#     def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
#                                           dynamic_lmk_b_coords,
#                                           neck_kin_chain, dtype=torch.float32):
#         """
#             Selects the face contour depending on the reletive position of the head
#             Input:
#                 vertices: N X num_of_vertices X 3
#                 pose: N X full pose
#                 dynamic_lmk_faces_idx: The list of contour face indexes
#                 dynamic_lmk_b_coords: The list of contour barycentric weights
#                 neck_kin_chain: The tree to consider for the relative rotation
#                 dtype: Data type
#             return:
#                 The contour face indexes and the corresponding barycentric weights
#         """

#         batch_size = pose.shape[0]

#         aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
#                                      neck_kin_chain)
#         rot_mats = lbs_batch.batch_rodrigues(
#             aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

#         rel_rot_mat = torch.eye(3, device=pose.device,
#                                 dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
#         for idx in range(len(neck_kin_chain)):
#             rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

#         y_rot_angle = torch.round(
#             torch.clamp(lbs_batch.rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
#                         max=39)).to(dtype=torch.long)

#         neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
#         mask = y_rot_angle.lt(-39).to(dtype=torch.long)
#         neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
#         y_rot_angle = (neg_mask * neg_vals +
#                        (1 - neg_mask) * y_rot_angle)

#         dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
#                                                0, y_rot_angle)
#         dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
#                                               0, y_rot_angle)
#         return dyn_lmk_faces_idx, dyn_lmk_b_coords

#     def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
#         """
#             Calculates landmarks by barycentric interpolation
#             Input:
#                 vertices: torch.tensor NxVx3, dtype = torch.float32
#                     The tensor of input vertices
#                 faces: torch.tensor (N*F)x3, dtype = torch.long
#                     The faces of the mesh
#                 lmk_faces_idx: torch.tensor N X L, dtype = torch.long
#                     The tensor with the indices of the faces used to calculate the
#                     landmarks.
#                 lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
#                     The tensor of barycentric coordinates that are used to interpolate
#                     the landmarks
#             Returns:
#                 landmarks: torch.tensor NxLx3, dtype = torch.float32
#                     The coordinates of the landmarks for each mesh in the batch
#         """
#         # Extract the indices of the vertices for each face
#         # NxLx3
#         batch_size, num_verts = vertices.shape[:dd2]
#         lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
#             1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

#         lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(vertices.device) * num_verts        # NOTE .to(device=vertices.device)였으나 type_as로 바꿈

#         lmk_vertices = vertices.view(-1, 3)[lmk_faces]
#         landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
#         return landmarks

#     def seletec_3d68(self, vertices):
#         landmarks3d = lbs_batch.vertices2landmarks(vertices, self.faces_tensor,
#                                          self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
#                                          self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
#         return landmarks3d

#     def find_landmarks(self, vertices, full_pose):

#         batch_size = vertices.shape[0]

#         lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
#         lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

#         dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(full_pose, 
#                                                                                         self.dynamic_lmk_faces_idx,
#                                                                                         self.dynamic_lmk_bary_coords,
#                                                                                         self.neck_kin_chain, dtype=self.dtype)
#         lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
#         lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

#         landmarks2d = lbs_batch.vertices2landmarks(vertices, self.faces_tensor,
#                                          lmk_faces_idx,
#                                          lmk_bary_coords)
#         bz = vertices.shape[0]
#         landmarks3d = lbs_batch.vertices2landmarks(vertices, self.faces_tensor,
#                                          self.full_lmk_faces_idx.repeat(bz, 1),
#                                          self.full_lmk_bary_coords.repeat(bz, 1, 1))
#         return landmarks2d, landmarks3d

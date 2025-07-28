import math
from functools import partial
import torch
import torch.nn as nn
from pytorch3d.ops import knn_points
from functorch import jacfwd, vmap
from utils import general as utils
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import clip
import torch.nn.functional as F

print_flushed = partial(print, flush=True)

class Model(nn.Module):
    def __init__(self, conf, img_res, canonical_expression, latent_code_dim, device):
        super().__init__()
        self.device = device
        canonical_pose = conf.get_float('dataset.canonical_pose', default=0.2)
        self.latent_code_dim = latent_code_dim
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')
        self.lora_finetuning = conf.get_bool('train.lora.lora_finetuning')

        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
                                                                                flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose).to(self.device)                       

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(conf=conf,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(conf=conf,
                                                                                         FLAMEServer=self.FLAMEServer,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.deformer_network'))
        
        self.gaussian_deformer_network = utils.get_class(conf.get_string('model.gaussian_class'))(conf=conf,
                                                                                                  latent_code_dim=self.latent_code_dim,
                                                                                                  **conf.get_config('model.gaussian_network'))

        self.ghostbone = self.deformer_network.ghostbone

        self.normal = True if conf.get_float('loss.normal_weight') > 0 and self.training else False
        self.target_training = conf.get_bool('train.target_training', default=False)
        
        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(**conf.get_config('model.point_cloud')).to(self.device)

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = conf.get_bool('dataset.use_background', default=False)

        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().to(self.device)
            self.background = nn.Parameter(init_background)
        else:
            self.background = torch.ones(img_res[0] * img_res[1], 3).float().to(self.device)

        self.visible_points = torch.zeros(n_points).bool().to(self.device)

        self.enable_prune = conf.get_bool('train.enable_prune')
        self.num_views = 100


        self.radius = 0.15 * (0.75 ** math.log2(n_points / 100))
        self.scale_ac = torch.sigmoid
        self.rotations_ac = torch.nn.functional.normalize
        self.opacity_ac = torch.sigmoid
        self.color_ac = torch.sigmoid

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)               # freeze by optimizer.
        self.clip_model = clip_model.requires_grad_(False)
        self.clip_preprocess = clip_preprocess
        self.clip_regression_network = utils.get_class(conf.get_string('model.clip_regression_class'))(conf=conf,
                                                                                                       clip_model=self.clip_model,
                                                                                                       clip_preprocess=self.clip_preprocess,
                                                                                                       **conf.get_config('model.clip_regression_network'))
        self.diffmorpher_weight = conf.get_float('loss.diffmorpher_weight')
        self.accumulation_steps = conf.get_int('train.accumulation_steps')
        self.diffmorpher_target_attrs = conf.get_list('train.diffmorpher_target_attrs')

    def _compute_canonical_normals_and_feature_vectors(self, p, condition):
        geometry_output, scales, rotations, opacity = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
        feature_rgb_vector = geometry_output
        feature_scale_vector = scales
        feature_rotation_vector = rotations
        feature_opacity_vector = opacity
        feature_vector = torch.concat([feature_rgb_vector, feature_rotation_vector, feature_scale_vector, feature_opacity_vector], dim=1)
        if not self.training:
            self._output['pnts_albedo'] = feature_rgb_vector

        return feature_vector
    
    def _render(self, 
                world_view_transform, 
                full_proj_transform, 
                camera_center, 
                tanfovx, 
                tanfovy,
                bg_color, 
                image_h, 
                image_w, 
                xyz, 
                color, 
                scales, 
                rotations, 
                opacity):

        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        raster_settings = GaussianRasterizationSettings(
            image_height=image_h,
            image_width=image_w,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=3,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        render_image, radii = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color,
            opacities=opacity,
            scales=scales+self.radius,
            rotations=rotations,
            cov3D_precomp=None)

        n_points = self.pc.points.shape[0]
        id = torch.arange(start=0, end=n_points, step=1).cuda()
        visible_points = id[opacity.reshape(-1) >= self.prune_thresh]
        visible_points = visible_points[visible_points != -1]
        return render_image, visible_points

    def latent_factory(self, 
                       model_input, 
                       scene_latent_code_sample):
        
        # sub_dir = model_input['sub_dir']
        category = model_input['category']
        # dataset_train_subdir = model_input['dataset_train_subdir']
        category_dict = model_input['category_dict']
        source_category_dict = model_input['source_category_dict']

        source_scene_latent_codes = model_input['source_scene_latent_codes']
        zero_latent_codes = model_input['zero_latent_codes']

        # scene_latent_code_sample = self.clip_regression_network(model_input['reference_image'], model_input['text_prompt'])
        if self.training and category != 'source':
            self._output['scene_latent_code_sample'] = scene_latent_code_sample         # mean이 0이 되겠금.

        # category_idx = category_dict[category]
        # category_one_hot = F.one_hot(torch.tensor([category_idx]), num_classes=len(category_dict)).to(self.device)
        # scene_latent_code_sample = torch.cat((category_one_hot, scene_latent_code_sample), dim=1)
        # batch_scene_latent, scene_latent_dim = scene_latent_code_sample.shape
        if category == 'source':
            category_idx = category_dict['hair']
        else:
            category_idx = category_dict[category]
        category_one_hot = F.one_hot(torch.tensor([category_idx]), num_classes=len(category_dict)).to(self.device)
        if category == 'source':
            scene_latent_code_sample = zero_latent_codes(torch.tensor([0]).to(self.device))
        scene_latent_code_sample = torch.cat((category_one_hot, scene_latent_code_sample), dim=1)
        batch_scene_latent, scene_latent_dim = scene_latent_code_sample.shape

        # step 3. initial input latent code를 만든다. [B, 32*len(category_dict)]
        input_latent_codes = torch.zeros(batch_scene_latent, scene_latent_dim*len(category_dict))
        
        # step 4. zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
        for i, v in category_dict.items():
            category_one_hot = F.one_hot(torch.tensor(v), num_classes=len(category_dict)).to(self.device)
            start_idx = v*scene_latent_dim
            end_idx = (v+1)*scene_latent_dim
            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_one_hot, zero_latent_codes(torch.tensor(0).to(self.device))), dim=0)
                
        # step 5. source에 관한 latent code들을 넣어준다.
        for i, (k, v) in enumerate(source_category_dict.items()):
            category_one_hot = F.one_hot(torch.tensor(v), num_classes=len(category_dict)).to(self.device)
            source_start_idx = v*scene_latent_dim
            source_end_idx = (v+1)*scene_latent_dim
            source_scene_latent_code_sample = source_scene_latent_codes(torch.tensor(i).to(self.device))
            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_one_hot, source_scene_latent_code_sample), dim=0)
        
        # step 6. 마지막으로 db human의 latent code를 넣어준다.
        # for i in range(len(index_list)):
        #     start_idx = index_list[i]*scene_latent_dim
        #     end_idx = (index_list[i]+1)*scene_latent_dim
        #     input_latent_codes[i, start_idx:end_idx] = scene_latent_code_sample[i]

        if category != 'source':
            start_idx = category_idx*scene_latent_dim
            end_idx = (category_idx+1)*scene_latent_dim
            input_latent_codes[0, start_idx:end_idx] = scene_latent_code_sample[0]

        # Add to the model_input dictionary
        return input_latent_codes.to(self.device)
    
    def interpolation(self, x1, x2, ratio, interp_type='linear'):
        if interp_type == 'linear':
            interp = x1 + ratio * (x2 - x1)
        return interp
    
    def forward(self, input):
        self._output = {}
        # intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]                               # NOTE BetaCancel
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, 
                                                                full_pose=flame_pose, 
                                                                shape_params=shape_params)              # NOTE BetaCancel

        if self.ghostbone:
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)        

        # NOTE add gaussian splatting-related
        world_view_transform = input["world_view_transform"].clone()
        full_proj_transform = input["full_proj_transform"].clone()
        camera_center = input["camera_center"].clone()
        tanfovx = input["tanfovx"]
        tanfovy = input["tanfovy"]
        bg_color = input["bg_color"].clone()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        if self.optimize_scene_latent_code:
            network_condition = dict()
            # network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            input['category'] = input['sub_dir'][0].split('_')[1]           # 0으로 하면 gafni sequence, 1로 하면 source_human2 sequence.
            if self.lora_finetuning:
                network_condition['lora_weight'] = input['lora_weight']    

            if self.training:
                # diffmorpher related
                if self.diffmorpher_weight > 0 and (input['category'] in self.diffmorpher_target_attrs) and input['dataset_type'] == ['diffmorpher']:
                    ref_image = input['reference_image'][0][input['sub_dir'][0]]
                    ref_text = input['reference_text'][0][input['sub_dir'][0]]
                    scene_latent_code_sample = self.clip_regression_network(ref_image, ref_text, network_condition)

                    ref_image = input['reference_image'][0][input['sub_dir_another_pair'][0]]
                    ref_text = input['reference_text'][0][input['sub_dir_another_pair'][0]]
                    scene_latent_code_sample_another_pair = self.clip_regression_network(ref_image, ref_text, network_condition)
                    assert (input['sub_dir'][0] != input['sub_dir_another_pair'][0]) and (input['sub_dir'][0].split('_')[1] == input['sub_dir_another_pair'][0].split('_')[1]), 'same sub_dir is not allowed.'
                    scene_latent_code_sample = self.interpolation(x1=scene_latent_code_sample, 
                                                                  x2=scene_latent_code_sample_another_pair, 
                                                                  ratio=input['interpolation_ratio'], 
                                                                  interp_type='linear')
                    network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                    # ref_image = input['reference_image'][0][input['sub_dir'][0]]
                    # ref_text = input['reference_text'][0][input['sub_dir'][0]]
                    # ref_image_another_pair = input['reference_image'][0][input['sub_dir_another_pair'][0]]
                    # ref_text_another_pair = input['reference_text'][0][input['sub_dir_another_pair'][0]]
                    # assert (input['sub_dir'][0] != input['sub_dir_another_pair'][0]) and (input['sub_dir'][0].split('_')[1] == input['sub_dir_another_pair'][0].split('_')[1]), 'same sub_dir is not allowed.'
                    
                    # network_condition['alpha'] = input['interpolation_ratio']
                    # scene_latent_code_sample = self.clip_regression_network([ref_image, ref_image_another_pair], [ref_text, ref_text_another_pair], network_condition)
                    # network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                # default training
                else:
                    if input['category'] != 'source':
                        ref_image = input['reference_image'][0][input['sub_dir'][0]]
                        ref_text = input['reference_text'][0][input['sub_dir'][0]]
                        scene_latent_code_sample = self.clip_regression_network(ref_image, ref_text, network_condition)
                        if self.lora_finetuning and 'lora_index' in input and input['lora_index'] > 0:
                            # scene_latent_code_sample = scene_latent_code_sample + input['random_noise']
                            scene_latent_code_sample = input['random_latent_code']
                        network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                    else:
                        network_condition['scene_latent'] = self.latent_factory(input, None).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                # if input['data_index'] == 0 and not self.lora_finetuning and not (self.diffmorpher_weight > 0):   # latent kl divergence 연산을 위해. lora와 diffmorpher finetuing할때는 스킵한다.
                #     scene_latent_code_dict = {}
                #     for ref_image_key, ref_image, ref_text in zip(input['reference_image'][0].keys(), input['reference_image'][0].values(), input['reference_text'][0].values()):
                #         scene_latent_code_sample = self.clip_regression_network(ref_image, ref_text, network_condition)
                #         category = ref_image_key.split('_')[1]
                #         assert category != 'source', 'source category is not allowed.'
                #         if category not in scene_latent_code_dict:
                #             scene_latent_code_dict[category] = {}
                #         scene_latent_code_dict[category][ref_image_key] = scene_latent_code_sample
                #     self._output['scene_latent_code_dict'] = scene_latent_code_dict         # kl divergence를 위해 넘김.
                #     # 그냥 scene latent 연산하는 과정.
                #     if input['category'] != 'source':
                #         network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_dict[input['category']][input['sub_dir'][0]]).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                #     else:
                #         network_condition['scene_latent'] = self.latent_factory(input, None).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                # else:
                #     if self.diffmorpher_weight > 0 and (input['category'] in self.diffmorpher_target_attrs) and input['dataset_type'] == ['diffmorpher']:
                #         ref_image = input['reference_image'][0][input['sub_dir'][0]]
                #         ref_text = input['reference_text'][0][input['sub_dir'][0]]
                #         scene_latent_code_sample = self.clip_regression_network(ref_image, ref_text, network_condition)

                #         ref_image = input['reference_image'][0][input['sub_dir_another_pair'][0]]
                #         ref_text = input['reference_text'][0][input['sub_dir_another_pair'][0]]
                #         scene_latent_code_sample_another_pair = self.clip_regression_network(ref_image, ref_text, network_condition)
                #         assert (input['sub_dir'][0] != input['sub_dir_another_pair'][0]) and (input['sub_dir'][0].split('_')[1] == input['sub_dir_another_pair'][0].split('_')[1]), 'same sub_dir is not allowed.'
                #         scene_latent_code_sample = self.interpolation(x1=scene_latent_code_sample, 
                #                                                       x2=scene_latent_code_sample_another_pair, 
                #                                                       ratio=input['interpolation_ratio'], 
                #                                                       interp_type='linear')
                #         network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                #     else:
                #         if input['category'] != 'source':
                #             ref_image = input['reference_image'][0][input['sub_dir'][0]]
                #             ref_text = input['reference_text'][0][input['sub_dir'][0]]
                #             scene_latent_code_sample = self.clip_regression_network(ref_image, ref_text, network_condition)
                #             if self.lora_finetuning and 'lora_index' in input and input['lora_index'] > 0:
                #                 scene_latent_code_sample = scene_latent_code_sample + input['random_noise']
                #             network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                #         else:
                #             network_condition['scene_latent'] = self.latent_factory(input, None).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            else:           # inference code
                if input.get('rendering_type') == ['interpolation']:
                    input['category'] = input['sub_dir_a'][0].split('_')[1]
                    ref_image_a = input['reference_image'][0][input['sub_dir_a'][0]]
                    ref_text_a = input['reference_text'][0][input['sub_dir_a'][0]]
                    ref_image_b = input['reference_image'][0][input['sub_dir_b'][0]]
                    ref_text_b = input['reference_text'][0][input['sub_dir_b'][0]]
                    scene_latent_code_sample_a = self.clip_regression_network(ref_image_a, ref_text_a, network_condition)
                    scene_latent_code_sample_b = self.clip_regression_network(ref_image_b, ref_text_b, network_condition)
                    scene_latent_code_sample = self.interpolation(x1=scene_latent_code_sample_a,
                                                                  x2=scene_latent_code_sample_b,
                                                                  ratio=input['interpolation_ratio'],
                                                                  interp_type='linear')
                    network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                elif input.get('rendering_type') == ['same_time']:
                    input['category'] = input['sub_dir'][0].split('_')[1]
                    if input['category'] == 'source':
                        if input['category'] in list(input['source_category_dict'].keys()):
                            scene_latent_code_sample_a = input['source_scene_latent_codes'](torch.tensor([list(input['source_category_dict'].keys()).index(input['category'])]).to(self.device))
                        else:
                            scene_latent_code_sample_a = input['zero_latent_codes'](torch.tensor([0]).to(self.device))
                        
                    ref_image = input['reference_image'][0][input['sub_dir'][0]]
                    ref_text = input['reference_text'][0][input['sub_dir'][0]]
                    scene_latent_code_sample_b = self.clip_regression_network(ref_image, ref_text, network_condition)
                    
                    scene_latent_code_sample = self.interpolation(x1=scene_latent_code_sample_a,
                                                                  x2=scene_latent_code_sample_b,
                                                                  ratio=input['interpolation_ratio'],
                                                                  interp_type='linear')
                    network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                elif input.get('rendering_type') == ['zero_shot']:
                    input['category'] = input['category_zero_shot']
                    ref_image = input['image_zero_shot']
                    ref_text = input['text_zero_shot']
                    scene_latent_code_sample = self.clip_regression_network(ref_image, ref_text, network_condition)
                    network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                elif input.get('rendering_type') == ['random_sampling']:
                    input['category'] = input['category_random_sampling']
                    scene_latent_code_sample = input['random_latent_code']
                    network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                else:
                    ref_image = input['reference_image'][0][input['sub_dir'][0]]
                    ref_text = input['reference_text'][0][input['sub_dir'][0]]
                    scene_latent_code_sample = self.clip_regression_network(ref_image, ref_text, network_condition)
                    if self.lora_finetuning and 'random_latent_code' in input:
                        scene_latent_code_sample = input['random_latent_code']
                    network_condition['scene_latent'] = self.latent_factory(input, scene_latent_code_sample).unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)

        else:
            network_condition = None

        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc

        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]        # NOTE FLAME의 beta blendshapes basis를 FLAME canonical space의 beta blendshaps basis로 바꿔주는 과정.

        feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)   

        transformed_points, rgb_points, scale_vals, rotation_vals, opacity_vals = self.get_rbg_value_functorch(pnts_c=self.pc.points,
                                                                                                               feature_vectors=feature_vector,
                                                                                                               pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                               betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                               transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                               cond=network_condition,
                                                                                                               shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                               gt_beta_shapedirs=gt_beta_shapedirs)

        transformed_points = transformed_points.reshape(batch_size, n_points, 3)

        # NOTE add gaussian splatting-related
        scale = scale_vals.reshape(transformed_points.shape[0], -1, 3)
        rotation = rotation_vals.reshape(transformed_points.shape[0], -1, 4)
        opacity = opacity_vals.reshape(transformed_points.shape[0], -1, 1)
        offset = transformed_points.squeeze().detach() - pnts_c_flame.detach()
        offset_scale, offset_rotation, offset_opacity, offset_color = self.gaussian_deformer_network(offset, network_condition)
        scale = scale + offset_scale
        rotation = rotation + offset_rotation
        opacity = opacity + offset_opacity

        rgb_points = rgb_points.reshape(batch_size, n_points, 3)

        # NOTE add gaussian splatting-related
        rgb_points = rgb_points + offset_color
        rgb_points = self.color_ac(rgb_points)
        scale = self.scale_ac(scale)
        #1024
        # scale = scale * 0.01
        # 512
        # scale = scale * 0.025
        # 768
        scale = scale * 0.0175
        rotation = self.rotations_ac(rotation)
        opacity = self.opacity_ac(opacity)

        rendering_list = []
        for idx in range(transformed_points.shape[0]):
            world_view_transform_i = world_view_transform[idx]
            full_proj_transform_i = full_proj_transform[idx]
            camera_center_i = camera_center[idx]
            tanfovx_i = tanfovx[idx]
            tanfovy_i = tanfovy[idx]
            bg_color_i = bg_color[idx]
            image_h_i = self.img_res[0]
            image_w_i = self.img_res[1]
            # image_h_i = 1024
            # image_w_i = 1024
            xyz_i = transformed_points[idx]
            color_i = rgb_points[idx]
            scales_i = scale[idx]
            rotations_i = rotation[idx]
            opacity_i = opacity[idx]

            if False:
                color_i = torch.rand_like(color_i) * 0.7
                opacity_i = (opacity_i >= 0.5).float()
                
            image, visible_points = self._render(world_view_transform_i, 
                                                 full_proj_transform_i, 
                                                 camera_center_i, 
                                                 tanfovx_i, 
                                                 tanfovy_i,
                                                 bg_color_i, 
                                                 image_h_i, 
                                                 image_w_i, 
                                                 xyz_i, 
                                                 color_i, 
                                                 scales_i, 
                                                 rotations_i, 
                                                 opacity_i)
            if self.training:
                self.visible_points[visible_points] = True
            rendering_list.append(image.unsqueeze(0))
        rgb_values = torch.concat(rendering_list, dim=0).permute(0, 2, 3, 1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights
        }

        if self.lora_finetuning and 'lora_index' in input and input['lora_index'] > 0:
            output['three_dim_values'] = {
                'xyz': xyz_i,
                'color': color_i,
                'scales': scales_i,
                'rotations': rotations_i,
                'opacity': opacity_i,
            }

        if not self.training:
            output_testing = {
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)

        # if self.optimize_scene_latent_code and self.training:
        #     output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            condition = {}
            condition['scene_latent'] = scene_latent
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)
            beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        # normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)      # NOTE pnts_c로 미분.
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        # grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        # grads_inv = grads_batch.inverse()
        # normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        rgb_vals = feature_vectors[:, 0:3]
        scale_vals = feature_vectors[:, 3:6]
        rotation_vals = feature_vectors[:, 6:10]
        opacity_vals = feature_vectors[:, 10:11]
        return pnts_d, rgb_vals, scale_vals, rotation_vals, opacity_vals

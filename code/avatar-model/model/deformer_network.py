import torch
from model.embedder import *
import numpy as np
import torch.nn as nn
from utils import general as utils



class ForwardDeformer(nn.Module):
    def __init__(self,
                 conf,
                 FLAMEServer,
                 d_in,
                 dims,
                 multires,
                 optimize_scene_latent_code,
                 latent_code_dim,
                 num_exp=50,
                 deform_c=False,
                 deform_cc=False,
                 weight_norm=True,
                 ghostbone=False,
                 ):
        super().__init__()
        self.lora_finetuning = conf.get_bool('train.lora.lora_finetuning')

        self.optimize_scene_latent_code = optimize_scene_latent_code
        if self.optimize_scene_latent_code:
            self.scene_latent_dim = latent_code_dim
        else:
            self.scene_latent_dim = 0

        self.FLAMEServer = FLAMEServer
        # pose correctives, expression blendshapes and linear blend skinning weights
        d_out = 36 * 3 + num_exp * 3

        if deform_cc and not deform_c:
            assert False, 'deform_cc should be False when deform_c is False'

        if deform_c:
            d_out = d_out + 3
        if deform_cc:
            d_out = d_out + 128
            
        self.num_exp = num_exp
        self.deform_c = deform_c
        self.deform_cc = deform_cc
        dims = [d_in + self.scene_latent_dim] + dims + [d_out]  
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn                                           
            dims[0] = input_ch + self.scene_latent_dim  

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            # if l > self.num_layers - 5 and self.lora_finetuning:
            if self.lora_finetuning:
            # if 0 < l < self.num_layers - 3 and self.lora_finetuning:
                lora = utils.get_class(conf.get_string('model.lora_class'))(in_features=dims[l], #+self.scene_latent_dim, 
                                                                            out_features=out_dim)
                setattr(self, "lora" + str(l), lora)

        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)

        if self.deform_cc:
            self.cc_linear = nn.Linear(128, 128)
            if self.lora_finetuning:
                self.cc_lora = utils.get_class(conf.get_string('model.lora_class'))(in_features=128, 
                                                                                    out_features=128)
            self.cc = nn.Linear(128, 3)

        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)
        # NOTE custom
        if self.deform_cc:
            torch.nn.init.constant_(self.cc_linear.bias, 0.0)
            torch.nn.init.normal_(self.cc_linear.weight, 0.0, np.sqrt(2) / np.sqrt(128))
            torch.nn.init.constant_(self.cc.bias, 0.0)
            torch.nn.init.constant_(self.cc.weight, 0.0)

        self.ghostbone = ghostbone

    def canocanonical_deform(self, pnts_c, mask=None, cond=None):
        return pnts_c
    
    def query_weights(self, pnts_c, mask=None, cond=None):
        if mask is not None:
            pnts_c = pnts_c[mask]
            if self.optimize_scene_latent_code:
                condition_scene_latent = cond['scene_latent'][mask]

        if self.embed_fn is not None:
            x = self.embed_fn(pnts_c)
        else:
            x = pnts_c

        if self.optimize_scene_latent_code:
            if mask is None:
                condition_scene_latent = cond['scene_latent']
            assert condition_scene_latent.shape[0] == x.shape[0], 'double check'
            x = torch.cat([x, condition_scene_latent], dim=1) # [300000, 71]

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))

            # if l > self.num_layers - 5 and self.lora_finetuning:
            # if 0 < l < self.num_layers - 3 and self.lora_finetuning:
            if self.lora_finetuning:
                lora = getattr(self, "lora" + str(l))
                # input_lora = torch.cat([x, condition_scene_latent], dim=1)
                output_lora = lora(x)
                x = lin(x) + self.lora_weight * output_lora
            else:
                x = lin(x)

            x = self.softplus(x)

        blendshapes = self.blendshapes(x)
        posedirs = blendshapes[:, :36 * 3]
        shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
        lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
        # softmax implementation
        lbs_weights_exp = torch.exp(20 * lbs_weights)
        lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)
        if self.deform_c:
            pnts_c_flame = pnts_c + blendshapes[:, -3:]     # NOTE offset을 더해준다. paper: canonical offset.
        else:
            pnts_c_flame = pnts_c

        return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame

    def forward_lbs(self, pnts_c, pose_feature, betas, transformations, mask=None, cond=None):
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.query_weights(pnts_c, mask, cond)
        pts_p = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32)
        return pts_p, pnts_c_flame
    

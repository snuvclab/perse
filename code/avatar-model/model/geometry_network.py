import torch
from model.embedder import *
import numpy as np
import torch.nn as nn
from utils import general as utils


class GeometryNetwork(nn.Module):
    def __init__(
            self,
            conf,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            optimize_scene_latent_code,
            latent_code_dim,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
    ):
        super().__init__()
        self.optimize_scene_latent_code = optimize_scene_latent_code
        self.lora_finetuning = conf.get_bool('train.lora.lora_finetuning')

        self.scene_latent_dim = latent_code_dim

        dims = [d_in + self.scene_latent_dim] + dims # + [d_out + feature_vector_size]

        self.feature_vector_size = feature_vector_size
        self.embed_fn = None
        self.multires = multires
        self.bias = bias
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            # dims[0] = input_ch
            dims[0] = input_ch + self.scene_latent_dim

        self.num_layers = len(dims)
        self.skip_in = skip_in
        
        self.softplus = nn.Softplus(beta=100)
        
        # NOTE add gaussian related
        self.scaling_layer = nn.Sequential(nn.Linear(512, 512), 
                                           self.softplus,
                                           nn.Linear(512, 3))
        self.rotations_layer = nn.Sequential(nn.Linear(512, 512), 
                                             self.softplus,
                                             nn.Linear(512, 4))
        self.opacity_layer = nn.Sequential(nn.Linear(512, 512), 
                                           self.softplus,
                                           nn.Linear(512, 1))
        self.scale_ac = nn.Softplus(beta=100)
        self.rotations_ac = nn.functional.normalize
        self.opacity_ac = nn.Sigmoid()
        ##################################

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

            if self.lora_finetuning and (l + 1 not in self.skip_in):            # all layer에 lora적용
                lora = utils.get_class(conf.get_string('model.lora_class'))(in_features=dims[l], #+self.scene_latent_dim, 
                                                                            out_features=out_dim)
                setattr(self, "lora" + str(l), lora)

        self.out_layer = nn.Sequential(nn.Linear(512, 512), 
                                       self.softplus,
                                       nn.Linear(512, 512), 
                                       nn.Linear(512, 3))
        
        if self.lora_finetuning:
            for i, out_layer in enumerate(self.out_layer):
                if isinstance(out_layer, nn.Linear):                             # 이전 버전 (성공한 버전)
                # if isinstance(out_layer, nn.Linear) and i < len(self.out_layer) - 1:
                    in_features = out_layer.in_features
                    out_features = out_layer.out_features
                    lora = utils.get_class(conf.get_string('model.lora_class'))(in_features=in_features, 
                                                                                out_features=out_features)
                    setattr(self, "{}{}".format('out_layer_lora', str(i)), lora)
            for i, scaling_layer in enumerate(self.scaling_layer):
                if isinstance(scaling_layer, nn.Linear):
                # if isinstance(scaling_layer, nn.Linear) and i < len(self.scaling_layer) - 1:
                    in_features = scaling_layer.in_features
                    out_features = scaling_layer.out_features
                    lora = utils.get_class(conf.get_string('model.lora_class'))(in_features=in_features, 
                                                                                out_features=out_features)
                    setattr(self, "{}{}".format('scaling_layer_lora', str(i)), lora)
            for i, rotations_layer in enumerate(self.rotations_layer):
                if isinstance(rotations_layer, nn.Linear):
                # if isinstance(rotations_layer, nn.Linear) and i < len(self.rotations_layer) - 1:
                    in_features = rotations_layer.in_features
                    out_features = rotations_layer.out_features
                    lora = utils.get_class(conf.get_string('model.lora_class'))(in_features=in_features, 
                                                                                out_features=out_features)
                    setattr(self, "{}{}".format('rotations_layer_lora', str(i)), lora)
            for i, opacity_layer in enumerate(self.opacity_layer):
                if isinstance(opacity_layer, nn.Linear):
                # if isinstance(opacity_layer, nn.Linear) and i < len(self.opacity_layer) - 1:
                    in_features = opacity_layer.in_features
                    out_features = opacity_layer.out_features
                    lora = utils.get_class(conf.get_string('model.lora_class'))(in_features=in_features, 
                                                                                out_features=out_features)
                    setattr(self, "{}{}".format('opacity_layer_lora', str(i)), lora)

    def forward(self, input, condition):
        if self.embed_fn is not None:
            input = self.embed_fn(input) # 800, 3 -> 800, 39

        if self.optimize_scene_latent_code:
            input = torch.cat([input, condition['scene_latent']], dim=1)

        x = input # 6400, 71

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            # if l > self.num_layers - 4 and self.lora_finetuning and (l + 1 not in self.skip_in):
            if self.lora_finetuning and (l + 1 not in self.skip_in):
            # if 0 < l < self.num_layers - 2 and self.lora_finetuning and (l + 1 not in self.skip_in):
                lora = getattr(self, "lora" + str(l))
                # input_lora = torch.cat([x, condition['scene_latent']], dim=1)
                output_lora = lora(x)
                x = lin(x) + condition['lora_weight'] * output_lora
            else:
                x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        if not self.lora_finetuning:
            color = self.out_layer(x)
            scales = self.scaling_layer(x)
            rotations = self.rotations_layer(x)
            opacity = self.opacity_layer(x)
        else:
            color = x
            for i, out_layer in enumerate(self.out_layer):
                if isinstance(out_layer, nn.Linear):
                # if isinstance(out_layer, nn.Linear) and i < len(self.out_layer) - 1:
                    lora = getattr(self, "{}{}".format('out_layer_lora', str(i)))
                    # input_lora = torch.cat([color, condition['scene_latent']], dim=1)
                    output_lora = lora(color)
                    color = out_layer(color) + condition['lora_weight'] * output_lora
                else:
                    color = out_layer(color)

            scales = x
            for i, scaling_layer in enumerate(self.scaling_layer):
                if isinstance(scaling_layer, nn.Linear):
                # if isinstance(scaling_layer, nn.Linear) and i < len(self.scaling_layer) - 1:
                    lora = getattr(self, "{}{}".format('scaling_layer_lora', str(i)))
                    # input_lora = torch.cat([scales, condition['scene_latent']], dim=1)
                    output_lora = lora(scales)
                    scales = scaling_layer(scales) + condition['lora_weight'] * output_lora
                else:
                    scales = scaling_layer(scales)

            rotations = x
            for i, rotations_layer in enumerate(self.rotations_layer):
                if isinstance(rotations_layer, nn.Linear):
                # if isinstance(rotations_layer, nn.Linear) and i < len(self.rotations_layer) - 1:
                    lora = getattr(self, "{}{}".format('rotations_layer_lora', str(i)))
                    # input_lora = torch.cat([rotations, condition['scene_latent']], dim=1)
                    output_lora = lora(rotations)
                    rotations = rotations_layer(rotations) + condition['lora_weight'] * output_lora
                else:
                    rotations = rotations_layer(rotations)

            opacity = x
            for i, opacity_layer in enumerate(self.opacity_layer):
                if isinstance(opacity_layer, nn.Linear):
                # if isinstance(opacity_layer, nn.Linear) and i < len(self.opacity_layer) - 1:
                    lora = getattr(self, "{}{}".format('opacity_layer_lora', str(i)))
                    # input_lora = torch.cat([opacity, condition['scene_latent']], dim=1)
                    output_lora = lora(opacity)
                    opacity = opacity_layer(opacity) + condition['lora_weight'] * output_lora
                else:
                    opacity = opacity_layer(opacity)

        return color, scales, rotations, opacity

from model.embedder import *
import torch.nn as nn
import torch
from utils import general as utils


class GaussianNetwork(nn.Module):
    def __init__(
            self,
            conf,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            latent_code_dim,
            weight_norm=True,
            multires_view=0,
            multires_pnts=0,
    ):
        super().__init__()
        self.conf = conf
        self.scene_latent_dim = latent_code_dim

        dims = [d_in + feature_vector_size + self.scene_latent_dim] + dims
        self.d_in = d_in
        self.embedview_fn = None
        self.embedpnts_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3) + self.scene_latent_dim

        if multires_pnts > 0:
            embedpnts_fn, input_ch_pnts = get_embedder(multires_pnts)
            self.embedpnts_fn = embedpnts_fn
            dims[0] += (input_ch_pnts - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus(beta=100)

        regression_layers_dim = dims[-1]
        self.scaling_layer = nn.Sequential(nn.Linear(regression_layers_dim, regression_layers_dim), 
                                           self.relu,
                                           nn.Linear(regression_layers_dim, 3))
        self.rotations_layer = nn.Sequential(nn.Linear(regression_layers_dim, regression_layers_dim), 
                                             self.relu,
                                             nn.Linear(regression_layers_dim, 4))
        self.opacity_layer = nn.Sequential(nn.Linear(regression_layers_dim, regression_layers_dim), 
                                           self.relu,
                                           nn.Linear(regression_layers_dim, 1))
        self.color_layer = nn.Sequential(nn.Linear(regression_layers_dim, regression_layers_dim), 
                                         self.relu,
                                         nn.Linear(regression_layers_dim, 3))

    def forward(self, offset, condition):
        offset = torch.cat([offset, condition['scene_latent']], dim=1)

        x = offset

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        offset_s = self.scaling_layer(x)
        offset_r = self.rotations_layer(x)
        offset_c = self.color_layer(x)
        offset_o = self.opacity_layer(x)
        return offset_s, offset_r, offset_o, offset_c
    


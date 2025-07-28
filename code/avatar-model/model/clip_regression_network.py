from model.embedder import *
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from einops import rearrange
import clip
from utils import general as utils


def resize_tensor(image_tensor, size=224):
    return transforms.functional.resize(image_tensor, [size, size], interpolation=transforms.InterpolationMode.BICUBIC)


    
class CLIPRegressionNetwork(nn.Module):
    def __init__(
            self,
            conf,
            clip_model,
            clip_preprocess,
            d_in,
            d_out,
            dims,
            weight_norm=True,
    ):
        super().__init__()
        self.conf = conf
        self.img_dim = conf.get_list('dataset.train.img_res')
        self.lora_finetuning = conf.get_bool('train.lora.lora_finetuning')

        dims = [d_in] + dims + [d_out]
        self.d_in = d_in
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

            # LoRA related
            # if l > self.num_layers - 4 and self.lora_finetuning:
            if self.lora_finetuning:
                lora = utils.get_class(conf.get_string('model.lora_class'))(in_features=dims[l], 
                                                                            out_features=out_dim)
                setattr(self, "lora" + str(l), lora)

        self.relu = nn.ReLU()

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

    def forward(self, image, text_prompt, cond):
        image = resize_tensor(rearrange(image.squeeze(0).reshape(self.img_dim[0], self.img_dim[1], 3), 'h w c -> c h w'), size=self.clip_model.visual.input_resolution).unsqueeze(0)
        image_feature = self.clip_model.encode_image(image).float()

        text = clip.tokenize(text_prompt[0]).to(image_feature.device)
        text_feature = self.clip_model.encode_text(text).float()

        x = torch.cat([image_feature, text_feature], dim=1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            # if l > self.num_layers - 4 and self.lora_finetuning:
            if self.lora_finetuning:
                lora = getattr(self, "lora" + str(l))
                lora_output = lora(x)
                x = lin(x) + cond['lora_weight'] * lora_output
            else:
                x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        return x

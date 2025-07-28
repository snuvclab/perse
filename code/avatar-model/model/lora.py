import torch
import torch.nn as nn

# NOTE borrow from diffusers code.
class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None):
        super().__init__()
        
        self.not_lora = False
        if rank > min(in_features, out_features):
            # raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")
            print(f"[WARN] LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")
            self.not_lora = True

        if self.not_lora:
            middle_features = (in_features + out_features) // 2
            self.down = nn.Linear(in_features, middle_features, bias=False)
            self.up = nn.Linear(middle_features, out_features, bias=False)
        else:
            self.down = nn.Linear(in_features, rank, bias=False)
            self.up = nn.Linear(rank, out_features, bias=False)
        
        # self.down = nn.Linear(in_features, rank, bias=False)
        # self.up = nn.Linear(rank, out_features, bias=False)
        
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
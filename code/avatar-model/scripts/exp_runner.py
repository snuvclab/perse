import sys
sys.path.append('../code')
sys.path.append('./')
import argparse

from pyhocon import ConfigFactory
import torch
import random, os
import numpy as np
import utils.general as utils
from collections import OrderedDict
from utils.hutils import is_main_process

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict) or isinstance(v, OrderedDict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_conf', default='confs/base.conf', type=str)
    parser.add_argument('--exp_conf', type=str, required=True)
    parser.add_argument('--is_test', default=False, action="store_true", help='If set, only render images')
    parser.add_argument('--nepochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--wandb_workspace', type=str)
    parser.add_argument('--only_json', default=False, action="store_true", help='If set, do not load images during testing. ')
    parser.add_argument('--checkpoint', default='latest', type=str, help='The checkpoint epoch number in case of continuing from a previous run.')
    parser.add_argument('--local_rank', type=int, default=0)
    opt = parser.parse_args()
    base_conf = ConfigFactory.parse_file(opt.base_conf)
    exp_conf = ConfigFactory.parse_file(opt.exp_conf)
    conf = exp_conf.with_fallback(base_conf)

    if is_main_process():
        exp_conf_flat = flatten_dict(exp_conf)
        base_conf_flat = flatten_dict(base_conf)
        extra_keys = set(exp_conf_flat.keys()) - set(base_conf_flat.keys())
        # Assert if there are any extra keys
        if extra_keys:
            print(f"The following configuration keys are present in exp_conf but not in base_conf: {extra_keys}")
            assert False, f"Extra keys in exp_conf: {extra_keys}"

    torch.backends.cuda.matmul.allow_tf32 = conf.get_bool('train.tf32')
    torch.backends.cudnn.allow_tf32 = conf.get_bool('train.tf32')

    seed(42)

    runner = utils.get_class(conf.get_string('runner'))(opt=opt,
                                                        conf=conf,
                                                        nepochs=opt.nepochs,
                                                        checkpoint=opt.checkpoint,
                                                        path_ckpt=None)
    
    runner.run()

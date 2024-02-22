# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# Modified by Yuyue Zhou
# --------------------------------------------------------


from .vision_transformer import build_vit
from .simmim_revised import build_simmim

def build_model(config, is_pretrain=True):

    model = build_simmim(config)
    return model

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# https://github.com/NVlabs/DigitalTwinArt/blob/master/preproc/loftr_wrapper.py

import os
import torchvision
import torch
import numpy as np
import sys
from os.path import join as pjoin
base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
sys.path.insert(0, pjoin(base_dir, '..'))
base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
sys.path.insert(0, pjoin(base_dir, '..'))
from LoFTR.src.loftr import default_cfg, LoFTR

state_dict_path = './weights/indoor_ds.ckpt'

class LoftrRunner:
    def __init__(self):
        default_cfg['match_coarse']['thr'] = 0.2
        self.matcher = LoFTR(config=default_cfg)
        self.matcher.load_state_dict(torch.load(state_dict_path)['state_dict'])
        self.matcher = self.matcher.eval().cuda()

    @torch.no_grad()
    def predict(self, rgbAs: np.ndarray, rgbBs: np.ndarray):
        '''
        @rgbAs: (N,H,W,C)
        '''
        image0 = torch.from_numpy(rgbAs).permute(0, 3, 1, 2).float().cuda()
        image1 = torch.from_numpy(rgbBs).permute(0, 3, 1, 2).float().cuda()
        if image0.shape[1] == 3:
            image0 = torchvision.transforms.functional.rgb_to_grayscale(image0)
            image1 = torchvision.transforms.functional.rgb_to_grayscale(image1)
        image0 = image0 / 255.0
        image1 = image1 / 255.0
        default_value = image0[0, 0, 0, 0]
        last_data = {'image0': image0, 'image1': image1,
                     }
        
        batch_size = 4
        ret_keys = ['mkpts0_f', 'mkpts1_f', 'mconf', 'm_bids']
        with torch.cuda.amp.autocast(enabled=True):
            i_b = 0
            for b in range(0, len(last_data['image0']), batch_size):
                tmp = {'image0': last_data['image0'][b:b + batch_size],
                       'image1': last_data['image1'][b:b + batch_size]}
                with torch.no_grad():
                    self.matcher(tmp)
                tmp['m_bids'] += i_b
                for k in ret_keys:
                    if k not in last_data:
                        last_data[k] = []
                    last_data[k].append(tmp[k])
                i_b += len(tmp['image0'])

        for k in ret_keys:
            last_data[k] = torch.cat(last_data[k], dim=0)

        mkpts0 = last_data['mkpts0_f'].cpu().numpy()
        mkpts1 = last_data['mkpts1_f'].cpu().numpy()
        mconf = last_data['mconf'].cpu().numpy()
        pair_ids = last_data['m_bids'].cpu().numpy()
        corres = np.concatenate((mkpts0.reshape(-1, 2), mkpts1.reshape(-1, 2), mconf.reshape(-1, 1)), axis=-1).reshape(
            -1, 5).astype(np.float32)

        corres_tmp = []
        for i in range(len(rgbAs)):
            cur_corres = corres[pair_ids == i]
            corres_tmp.append(cur_corres)
        corres = corres_tmp

        del last_data, image0, image1
        torch.cuda.empty_cache()

        return corres
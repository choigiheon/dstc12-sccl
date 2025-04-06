"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os

import random
import torch
import numpy as np


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def setup_path(args):
    resPath = "SCCL"
    resPath += f'.{args.bert}'
    resPath += f'.{args.use_pretrain}'
    resPath += f'.{args.augtype}'
    resPath += f'.{args.dataname}'
    resPath += f".{args.text}"
    resPath += f'.lr{args.lr}'
    resPath += f'.lrscale{args.lr_scale}'
    resPath += f'.{args.objective}'
    resPath += f'.eta{args.eta}'
    resPath += f'.tmp{args.temperature}'
    resPath += f'.alpha{args.alpha}'
    resPath += f'.seed{args.seed}/'
    resPath = args.resdir + resPath
    print(f'results path: {resPath}')

    tensorboard = SummaryWriter(resPath)
    return resPath, tensorboard


def statistics_log(tensorboard, losses=None, global_step=0):
    print("[{}]-----".format(global_step))
    if losses is not None:
        for key, val in losses.items():
            if key in ["pos", "neg", "pos_diag", "pos_rand", "neg_offdiag"]:
                tensorboard.add_histogram('train/'+key, val, global_step)
            else:
                try:
                    tensorboard.add_scalar('train/'+key, val.item(), global_step)
                except:
                    tensorboard.add_scalar('train/'+key, val, global_step)
                print("{}:\t {:.3f}".format(key, val))



def print_args(args):
    print("\n===== 사용한 파라미터 =====")
    print("\n----- 기본 설정 -----")
    print(f"seed: {args.seed}")
    print(f"print_freq: {args.print_freq}")
    print(f"device: {args.device}")
    print(f"model_name: {args.model_name}")
    print(f"dropout: {args.dropout}")
    
    print("\n----- 데이터셋 설정 -----")
    print(f"dataset_file: {args.dataset_file}")
    print(f"result_file: {args.result_file}")
    print(f"num_clusters: {args.num_clusters}")
    print(f"max_length: {args.max_length}")
    
    print("\n----- 학습 파라미터 -----")
    print(f"lr: {args.lr}")
    print(f"lr_scale: {args.lr_scale}")
    print(f"joint-max_iter: {args.joint_max_iter}")
    print(f"pre-max_iter: {args.pre_max_iter}")
    
    print("\n----- 대조 학습 설정 -----")
    print(f"augtype: {args.augtype}")
    print(f"batch_size: {args.batch_size}")
    print(f"temperature: {args.temperature}")
    print(f"eta: {args.eta}")
    
    print("\n----- 클러스터링 설정 -----")
    print(f"alpha: {args.alpha}")
    # print(f"interval: {args.interval}")
    print(f"use_progressive: {args.use_progressive}")
    
    print("\n----- 평가 설정 -----")
    print(f"eval_interval: {args.eval_interval}")
    print("\n==========================\n")
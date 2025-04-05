"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
from models.Transformers import SCCLModel
import dataloader.dataloader as dataloader
from training import SCCLvTrainer,TrainType
from utils.kmeans import get_kmeans_centers, ProgressiveKMeans
from utils.optimizer import get_optimizer, get_model
import numpy as np

def run(args):
    # dataset loader
    # train_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader(args)
    torch.manual_seed(args.seed)
    train_loader = dataloader.dstc12_loader(args)

    # model
    model, tokenizer = get_model(args)
    
    model = SCCLModel(model, tokenizer, alpha=args.alpha) 
    model = model.to(args.device)
    model.train()

    # optimizer 
    optimizer = get_optimizer(model, args)
    
    cluster_model = ProgressiveKMeans(args.num_clusters, args.max_length, args, use_progressive=args.use_progressive)
    trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader, cluster_model, args)
    trainer.train(train_type=TrainType.pre_train)
    model.set_cluster_centers(cluster_model.get_hsc())
    trainer.train(train_type=TrainType.joint_train)
    trainer.predict(args.result_file)
    trainer.evaluate(args.dataset_file, args.result_file)
    
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=5, help="")
    parser.add_argument('--device', type=str, default='mps', help="")  
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-mpnet-base-v2', help="")
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-mpnet-base-v2', help="")
    parser.add_argument('--dropout', type=float, default=0.1, help="")
    # Dataset
    parser.add_argument('--dataset_file', type=str, default='./dstc12-data/AppenBanking/all.jsonl')
    parser.add_argument('--result_file', type=str, default='./appen_banking_predicted.jsonl', help="결과를 저장할 파일 경로")
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=40)
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-6, help="")
    parser.add_argument('--lr_scale', type=int, default=10, help="head에는 lr_scale 적용")
    parser.add_argument('--joint-max_iter', type=int, default=41*3)
    parser.add_argument('--pre-max_iter', type=int, default=41*3)
    # contrastive learning
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit']) # 건들지 말 것.
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=10, help="")
    
    # Clustering
    parser.add_argument('--num_clusters', type=int, default=14)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--use_progressive', type=bool, default=False)
    parser.add_argument('--n_init', type=int, default=100, help="Progressive KMeans의 초기화 횟수")
    
    # evaluation
    parser.add_argument('--eval_interval', type=int, default=41, help="eval 결과를 출력할 간격")
    
    args = parser.parse_args(argv)
    
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
    
    args.resPath = None

    return args



if __name__ == '__main__':
    import subprocess
       
    args = get_args(sys.argv[1:])
    run(args)
            



    

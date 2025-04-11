"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
import argparse
from models.Transformers import SCCLModel
import dataloader.dataloader as dataloader
from training import SCCLvTrainer,TrainType
from utils.kmeans import get_kmeans_centers, ProgressiveKMeans
from utils.optimizer import get_optimizer, get_model
import numpy as np
from sklearn.cluster import KMeans
import wandb
from spherecluster import VonMisesFisherMixture
from knockknock import discord_sender, email_sender

def run(args):
    # dataset loader
    torch.manual_seed(args.seed)

    # model
    model, tokenizer = get_model(args)
    
    model = SCCLModel(model, tokenizer, alpha=args.alpha, cluster_head_dim=(args.n_clusters, args.cluster_head_dim)) 
    model = model.to(args.device)
    model.train()

    # optimizer 
    optimizer = get_optimizer(model, args)
    cluster_model = ProgressiveKMeans(n_clusters=args.n_clusters, max_length=args.max_length, args=args)
    trainer = SCCLvTrainer(model, tokenizer, optimizer, cluster_model, args)
    
    # wandb 초기화 - 하나의 run에서 모든 stage 기록
    config = {
        "n_clusters": args.n_clusters,
        "alpha": args.alpha,
        "n_init": args.n_init,
        "pre_train_epoch": args.pre_train_epoch,
        "inter_train_epoch": args.inter_train_epoch,
        "joint_train_epoch": args.joint_train_epoch,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "total_epochs": args.pre_train_epoch + args.inter_train_epoch + args.joint_train_epoch,
        "update_interval": args.update_interval,
    }
    wandb.init(project="sccl-training", name="sccl_three_stage_training", config=config)
    
    # 단계별 훈련 및 전역 step 관리
    global_step = 0
    
    # 첫 번째 단계: 사전 훈련
    dstc12_all_loader = dataloader.dstc12_all_loader(args)
    global_step = trainer.train(TrainType.pre_train, dstc12_all_loader, global_step=global_step)
    
    # 두 번째 단계: 양성 쌍 기반 훈련
    loader_positive = dataloader.dstc12_loader_with_positive(args)
    global_step = trainer.train(TrainType.inter_train, loader_positive, global_step=global_step)
    
    # 세 번째 단계: 양성/음성 쌍 기반 훈련
    loader_negative = dataloader.dstc12_loader_with_negative(args)
    global_step = trainer.train(TrainType.joint_train, loader_negative, global_step=global_step)
    cluster_label_map = trainer.predict(args.result_file)
    metrics = trainer.evaluate(args.dataset_file, args.result_file)
    
    # 최종 평가 결과 로깅
    wandb.log({f"final_{k}": v for k, v in metrics.items()})
    # wandb 세션 종료
    wandb.finish()
    
    return cluster_label_map

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--device', type=str, default='auto', help="")  
    parser.add_argument('--model-name', type=str, default='sentence-transformers/all-mpnet-base-v2', help="")
    parser.add_argument('--dropout', type=float, default=0.1, help="")
    # Dataset
    parser.add_argument('--dataset-file', type=str, default='./dstc12-data/AppenBanking/all copy.jsonl')
    parser.add_argument('--preference-file', type=str, default='./dstc12-data/AppenBanking/preference_pairs.json')
    parser.add_argument('--result-file', type=str, default='./appen_banking_predicted.jsonl')
    parser.add_argument('--max-length', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-7, help="")
    parser.add_argument('--lr-scale', type=int, default=10)
    parser.add_argument('--pre-train-epoch', type=int, default=3)
    parser.add_argument('--inter-train-epoch', type=int, default=3)
    parser.add_argument('--joint-train-epoch', type=int, default=3)
    
    # contrastive learning
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--negative-alpha', type=float, default=5.0, help="negative alpha required by contrastive loss")
    
    # Clustering
    parser.add_argument('--n-clusters', type=int, default=14)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--n-init', type=int, default=100, help="Kmeans++의 초기화 횟수")
    parser.add_argument('--update-interval', type=int, default=1, help="update interval")
    parser.add_argument('--cluster-head-dim', type=int, default=768, help="cluster head dimension")
    
    # evaluation
    parser.add_argument('--print-freq', type=float, default=1, help="loss 출력 간격 (iter 기준)")
    parser.add_argument('--eval-interval', type=int, default=1, help="eval 결과를 출력할 간격 (epoch 기준)")
    
    args = parser.parse_args(argv)

    return args

@discord_sender("https://discord.com/api/webhooks/1359959139052290359/DFz7VrxBveCDUsEZYiPGXhDZ8IlccXXw4gFf8jH7QsZzyFock549yEwLW61HEo1Sgt9O")
def main():
    args = get_args(sys.argv[1:])
    cluster_label_map = run(args)
    cluster_label_map = {k: int(float(v)) for k, v in cluster_label_map.items()}
    return cluster_label_map


if __name__ == '__main__':
    cluster_label_map = main()

    # 저장할 파일 경로 생성 (result_file 경로에서 확장자 변경)
    json_output_path = "./cluster_label_map.json"

    # JSON 파일로 저장
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_label_map, f, ensure_ascii=False, indent=4)

    print(f"클러스터 라벨 맵이 '{json_output_path}'에 저장되었습니다.")
            



    

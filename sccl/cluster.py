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

def run(args):
    # dataset loader
    torch.manual_seed(args.seed)
    train_loader = dataloader.dstc12_loader(args)

    # model
    model, tokenizer = get_model(args)
    
    model = SCCLModel(model, tokenizer, alpha=args.alpha) 
    model = model.to(args.device)
    model.train()

    # optimizer 
    optimizer = get_optimizer(model, args)
    
    cluster_model = ProgressiveKMeans(args.n_clusters, args.max_length, args)
    trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader, cluster_model, args)
    trainer.train(train_type=TrainType.pre_train)
    
    model.set_cluster_centers(cluster_centers=cluster_model.get_hsc())
    optimizer = get_optimizer(model, args) # 파라미터가 변경되었으므로 새로운 optimizer 생성
    trainer.set_optimizer(optimizer)
    
    trainer.train(train_type=TrainType.joint_train)
    cluster_label_map = trainer.predict(args.result_file)
    trainer.evaluate(args.dataset_file, args.result_file)
    
    return cluster_label_map

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--device', type=str, default='auto', help="")  
    parser.add_argument('--model-name', type=str, default='sentence-transformers/all-mpnet-base-v2', help="")
    parser.add_argument('--dropout', type=float, default=0.1, help="")
    # Dataset
    parser.add_argument('--dataset-file', type=str, default='./dstc12-data/AppenBanking/all copy.jsonl')
    parser.add_argument('--result-file', type=str, default='./appen_banking_predicted.jsonl')
    parser.add_argument('--max-length', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-7, help="")
    parser.add_argument('--pre-train-epoch', type=int, default=3)
    parser.add_argument('--joint-train-epoch', type=int, default=3)
    
    # contrastive learning
    parser.add_argument('--lr-scale', type=int, default=10)
    parser.add_argument('--augtype', type=str, default='simcse', choices=['simcse', 'explicit'])
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=10, help="")
    
    # Clustering
    parser.add_argument('--n-clusters', type=int, default=14)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--n-init', type=int, default=100, help="Kmeans++의 초기화 횟수")
    parser.add_argument('--kmeans-interval', type=int, default=1, help="Progressive KMeans 수행 간격 (epoch 기준)")
    
    # evaluation
    parser.add_argument('--print-freq', type=float, default=1, help="loss 출력 간격 (iter 기준)")
    parser.add_argument('--eval-interval', type=int, default=1, help="eval 결과를 출력할 간격 (epoch 기준)")
    
    args = parser.parse_args(argv)

    return args



if __name__ == '__main__':

    args = get_args(sys.argv[1:])

    cluster_label_map = run(args)
    cluster_label_map = {k: int(v) for k, v in cluster_label_map.items()}

    # 저장할 파일 경로 생성 (result_file 경로에서 확장자 변경)
    json_output_path = "./cluster_label_map.json"

    # JSON 파일로 저장
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_label_map, f, ensure_ascii=False, indent=4)

    print(f"클러스터 라벨 맵이 '{json_output_path}'에 저장되었습니다.")
            



    

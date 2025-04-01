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
from training import SCCLvTrainer
from utils.kmeans import get_kmeans_centers
from utils.optimizer import get_optimizer, get_model
import numpy as np


def run(args):
    # dataset loader
    # train_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader(args)
    train_loader = dataloader.dstc12_loader(args)

    # model
    model, tokenizer = get_model(args)
    
    # initialize cluster centers
    cluster_centers = get_kmeans_centers(model, tokenizer, train_loader, args.num_clusters, args.max_length, args=args)
    
    model = SCCLModel(model, tokenizer, cluster_centers=cluster_centers, alpha=args.alpha) 
    model.train()
    model = model.to(args.device)

    # optimizer 
    optimizer = get_optimizer(model, args)
    
    trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader, args)
    trainer.train()
    trainer.predict()
    
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=5, help="")
    parser.add_argument('--device', type=str, default='mps', help="")  
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="")
    parser.add_argument('--dropout', type=float, default=0.1, help="")
    # Dataset
    parser.add_argument('--dataset_file', type=str, default='./dstc12-data/AppenBanking/all.jsonl')
    parser.add_argument('--result_file', type=str, default='./appen_banking_predicted.jsonl', help="결과를 저장할 파일 경로")
    parser.add_argument('--dataname', type=str, default='searchsnippets', help="")
    parser.add_argument('--num_clusters', type=int, default=10, help="")
    parser.add_argument('--max_length', type=int, default=100)
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-6, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="head에는 lr_scale 적용")
    parser.add_argument('--max_iter', type=int, default=0)
    # contrastive learning
    parser.add_argument('--objective', type=str, default='SCCL', choices=['SCCL', 'contrastive'])
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit']) # 건들지 말 것.
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=10, help="")
    
    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    args.resPath = None

    return args



if __name__ == '__main__':
    import subprocess
       
    args = get_args(sys.argv[1:])
    run(args)
            



    

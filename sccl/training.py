"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import time
import numpy as np
from sklearn import cluster
import json
from tqdm import tqdm
from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_dstc12_loader

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss

class TrainType:
    pre_train = "CL"
    joint_train = "SCCL"

class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, cluster_model, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        self.cluster_model = cluster_model
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")
        
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch, args):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
            
        elif len(batch) == 1: # Virtual Augmentation
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = feat1.copy()
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.to(self.args.device), attention_mask.to(self.args.device)
        
        
    def train_step_virtual(self, input_ids, attention_mask, objective):
        
        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]
        
        # Clustering loss
        if objective == TrainType.joint_train: # SCCL
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += self.eta*cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train_step_explicit(self, input_ids, attention_mask):
        
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        if self.args.objective == TrainType.joint_train:
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += self.eta*cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    def train(self, train_type):
        max_iter = self.args.joint_max_iter if train_type == TrainType.joint_train else self.args.pre_max_iter
        print('\n={}/{}=Iterations/Batches'.format(max_iter, len(self.train_loader)))
        self.model.train()
        
        train_loader_iter = iter(self.train_loader)
        
        # For reference
        self.predict(self.args.result_file)
        self.evaluate(self.args.dataset_file, self.args.result_file)
        
        for i in tqdm(np.arange(max_iter)):
            
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)
                
                all_embeddings, all_utterances = self.get_embeddings(self.train_loader)
                self.cluster_model.update(all_embeddings) # 에포크가 끝날떄마다 클러스터 업데이트

            input_ids, attention_mask = self.prepare_transformer_input(batch, self.args)

            losses = self.train_step_virtual(input_ids, attention_mask, objective=TrainType.joint_train) if train_type == TrainType.joint_train else self.train_step_virtual(input_ids, attention_mask, objective=TrainType.pre_train)
                
            if (i % self.args.eval_interval == 0) and (i != 0):
                self.predict(self.args.result_file)
                self.evaluate(self.args.dataset_file, self.args.result_file)
                self.model.train()

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==max_iter)):
                print(f"loss: {losses['loss']}, pos_mean: {losses['pos_mean']}, neg_mean: {losses['neg_mean']}")
                if train_type == TrainType.joint_train:
                    print(f"cluster_loss: {losses['cluster_loss']}")
                self.model.train()

        return None   
    
    def predict(self, result_file):
        dataloader = unshuffle_dstc12_loader(self.args)
        print('---- {} prediction batches ----'.format(len(dataloader)))     
        self.model.eval()
        
        # K-means 클러스터링 수행
        all_embeddings, all_utterances = self.get_embeddings(dataloader)
        cluster_labels, high_score_centers = self.cluster_model.predict(all_embeddings) 
        
        print(f"클러스터링 완료: {self.args.num_clusters}개 클러스터")
        
        # 각 발화문에 클러스터 라벨 매핑
        cluster_label_map = {utterance: str(label) for utterance, label in zip(all_utterances, cluster_labels)}
        
        # 원본 데이터셋 로드
        with open(self.args.dataset_file) as f:
            dataset = [json.loads(line) for line in f]
         
        # 테마 라벨이 있는 발화문 추출
        themed_utterances = set()
        for dialogue in dataset:
            for turn in dialogue['turns']:
                if turn.get('theme_label') is not None:
                    themed_utterances.add(turn['utterance'])
        
        print(f"theme_label이 있는 발화문: {len(themed_utterances)}개")
        
        # 예측 결과를 원본 데이터셋에 추가
        dataset_predicted = dataset.copy()
        for dialogue in dataset_predicted:
            for turn in dialogue['turns']:
                if turn.get('theme_label') is not None:
                    # 발화문이 cluster_label_map에 없는 경우 처리
                    if turn['utterance'] in cluster_label_map:
                        turn['theme_label_predicted'] = cluster_label_map[turn['utterance']]
                    else:
                        print(f"경고: '{turn['utterance']}'에 대한 클러스터 라벨을 찾을 수 없습니다.")
                        # 가장 가까운 클러스터 할당 또는 기본값 설정
                        turn['theme_label_predicted'] = 0
        
            
        with open(result_file, 'w') as result_out:
            for dialogue in dataset_predicted:
                print(json.dumps(dialogue), file=result_out)
                
    def get_embeddings(self, dataloader):
        all_embeddings = []
        all_utterances = []
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="임베딩 추출"):
                text = batch['text']
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].to(self.args.device), 
                                       feat['attention_mask'].to(self.args.device), 
                                       task_type="evaluate")
                
                # 임베딩과 해당 발화문 저장
                all_embeddings.append(embeddings.detach().cpu())
                all_utterances.extend(text)
        
        # 모든 임베딩 결합
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        return all_embeddings, all_utterances

    def evaluate(self, dataset_file, result_file):
        os.system(f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate dstc12 && . ./set_paths.sh && python3 scripts/run_evaluation.py {dataset_file} {result_file}'")

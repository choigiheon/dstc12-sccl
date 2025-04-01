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

class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        
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
        
        
    def train_step_virtual(self, input_ids, attention_mask):
        
        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]
        
        # Clustering loss
        if self.args.objective == "SCCL":
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
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += self.eta*cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        
        train_loader_iter = iter(self.train_loader)
        for i in tqdm(np.arange(self.args.max_iter+1)):
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch, self.args)

            losses = self.train_step_virtual(input_ids, attention_mask) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask)

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                # statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                # self.evaluate_embedding(i)
                self.model.train()
                print(f"loss: {losses['loss']}, pos_mean: {losses['pos_mean']}, neg_mean: {losses['neg_mean']}")
                if self.args.objective == "SCCL":
                    print(f"cluster_loss: {losses['cluster_loss']}")

        return None   
    
    def predict(self):
        dataloader = unshuffle_dstc12_loader(self.args)
        print('---- {} prediction batches ----'.format(len(dataloader)))     
        self.model.eval()
        
        # 모든 발화문의 임베딩 추출
        all_embeddings = []
        all_utterances = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                text = batch['text']
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].to(self.args.device), 
                                       feat['attention_mask'].to(self.args.device), 
                                       task_type="evaluate")
                
                # 임베딩과 해당 발화문 저장
                all_embeddings.append(embeddings.detach().cpu())
                all_utterances.extend(text)
                
                if i % 10 == 0:
                    print(f"Processed {i}/{len(dataloader)} batches")
        
        # 모든 임베딩 결합
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        print(f"총 {len(all_utterances)}개 발화문 임베딩 추출 완료, 임베딩 크기: {all_embeddings.shape}")
        
        # K-means 클러스터링 수행
        kmeans = cluster.KMeans(n_clusters=self.args.num_clusters, random_state=self.args.seed, n_init=100, init='k-means++')
        cluster_labels = kmeans.fit_predict(all_embeddings)
        
        print(f"클러스터링 완료: {self.args.num_clusters}개 클러스터")
        
        # 각 발화문에 클러스터 라벨 매핑
        cluster_label_map = {utterance: str(label) for utterance, label in zip(all_utterances, cluster_labels)}
        
        # 클러스터 분포 확인
        cluster_counts = np.bincount(cluster_labels, minlength=self.args.num_clusters)
        for i, count in enumerate(cluster_counts):
            print(f"클러스터 {i}: {count}개 발화문")
        
        # 원본 데이터셋 로드
        with open(self.args.dataset_file) as f:
            dataset = [json.loads(line) for line in f]
         
        # 테마 라벨이 있는 발화문 추출
        themed_utterances = set()
        for dialogue in dataset:
            for turn in dialogue['turns']:
                if turn.get('theme_label') is not None:
                    themed_utterances.add(turn['utterance'])
        
        print(f"테마 라벨이 있는 발화문: {len(themed_utterances)}개")
        
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
        
            
        with open(self.args.result_file, 'w') as result_out:
            for dialogue in dataset_predicted:
                print(json.dumps(dialogue), file=result_out)

    # def evaluate_embedding(self, step):
    #     dataloader = unshuffle_loader(self.args)
    #     print('---- {} evaluation batches ----'.format(len(dataloader)))
        
    #     self.model.eval()
    #     for i, batch in enumerate(dataloader):
    #         with torch.no_grad():
    #             text, label = batch['text'], batch['label'] 
    #             feat = self.get_batch_token(text)
    #             embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

    #             model_prob = self.model.get_cluster_prob(embeddings)
    #             if i == 0:
    #                 all_labels = label
    #                 all_embeddings = embeddings.detach()
    #                 all_prob = model_prob
    #             else:
    #                 all_labels = torch.cat((all_labels, label), dim=0)
    #                 all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
    #                 all_prob = torch.cat((all_prob, model_prob), dim=0)
                    
    #     # Initialize confusion matrices
    #     confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        
    #     all_pred = all_prob.max(1)[1]
    #     confusion_model.add(all_pred, all_labels)
    #     confusion_model.optimal_assignment(self.args.num_classes)
    #     acc_model = confusion_model.acc()

    #     kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
    #     embeddings = all_embeddings.cpu().numpy()
    #     kmeans.fit(embeddings)
    #     pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
        
    #     # clustering accuracy 
    #     confusion.add(pred_labels, all_labels)
    #     confusion.optimal_assignment(self.args.num_classes)
    #     acc = confusion.acc()

    #     ressave = {"acc":acc, "acc_model":acc_model}
    #     ressave.update(confusion.clusterscores())
    #     for key, val in ressave.items():
    #         self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)
            
    #     np.save(self.args.resPath + 'acc_{}.npy'.format(step), ressave)
    #     np.save(self.args.resPath + 'scores_{}.npy'.format(step), confusion.clusterscores())
    #     np.save(self.args.resPath + 'mscores_{}.npy'.format(step), confusion_model.clusterscores())
    #     # np.save(self.args.resPath + 'mpredlabels_{}.npy'.format(step), all_pred.cpu().numpy())
    #     # np.save(self.args.resPath + 'predlabels_{}.npy'.format(step), pred_labels.cpu().numpy())
    #     # np.save(self.args.resPath + 'embeddings_{}.npy'.format(step), embeddings)
    #     # np.save(self.args.resPath + 'labels_{}.npy'.format(step), all_labels.cpu())

    #     print('[Representation] Clustering scores:',confusion.clusterscores()) 
    #     print('[Representation] ACC: {:.3f}'.format(acc)) 
    #     print('[Model] Clustering scores:',confusion_model.clusterscores()) 
    #     print('[Model] ACC: {:.3f}'.format(acc_model))
    #     return None



             
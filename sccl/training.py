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
from learner.contrastive_utils import PairConLossPositive, PairConLossNegative

class TrainType:
    pos_train_pre = "positive_train"
    neg_train_joint = "negative_train"

class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, cluster_model, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.args = args
        self.cluster_model = cluster_model
        self.contrast_loss_positive = PairConLossPositive(temperature=self.args.temperature)
        self.contrast_loss_negative = PairConLossNegative(temperature=self.args.temperature)
        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}\n")
        
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch, train_type):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
            
        elif len(batch) == 2:
            if train_type == TrainType.pos_train_pre:
                text1, text2 = batch['text_1'], batch['text_2']
                feat1 = self.get_batch_token(text1)
                feat2 = self.get_batch_token(text2)
                input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
                attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            elif train_type == TrainType.neg_train_joint:
                text1, text2 = batch['text_1'], batch['text_2']
                feat1_1 = self.get_batch_token(text1)
                feat1_2 = self.get_batch_token(text1)
                feat2_1 = self.get_batch_token(text2)
                feat2_2 = self.get_batch_token(text2)
                input_ids = torch.cat([feat1_1['input_ids'].unsqueeze(1), feat1_2['input_ids'].unsqueeze(1), feat2_1['input_ids'].unsqueeze(1), feat2_2['input_ids'].unsqueeze(1)], dim=1)
                attention_mask = torch.cat([feat1_1['attention_mask'].unsqueeze(1), feat1_2['attention_mask'].unsqueeze(1), feat2_1['attention_mask'].unsqueeze(1), feat2_2['attention_mask'].unsqueeze(1)], dim=1)
            
            
        elif len(batch) == 1: # simcse Augmentation
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = feat1.copy()
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.to(self.args.device), attention_mask.to(self.args.device)
        
        
    def train_step_pre(self, input_ids, attention_mask):
        
        embd1, embd2 = self.model(input_ids, attention_mask, task_type=TrainType.pos_train_pre)

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss_positive(feat1, feat2)
        loss = losses["loss"]

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    def train_step_joint(self, input_ids, attention_mask):
        embd1_1, embd1_2, embd2_1, embd2_2 = self.model(input_ids, attention_mask, task_type=TrainType.neg_train_joint)

        # Instance-CL loss
        feat1_1, feat1_2, feat2_1, feat2_2 = self.model.contrast_logits_negative(embd1_1, embd1_2, embd2_1, embd2_2)
        losses = self.contrast_loss_negative(feat1_1, feat1_2, feat2_1, feat2_2)
        loss = losses["loss"]
        return losses
    
    def train(self, train_type, train_loader, eval_loader):
        max_epoch = self.args.joint_train_epoch if train_type == TrainType.neg_train_joint else self.args.pre_train_epoch
        print("Train Type: ", "pre_train_pos" if train_type == TrainType.pos_train_pre else "joint_train_neg")
        print('\n={}/{}=Epochs/Batches'.format(max_epoch, len(train_loader)))
        self.model.train()
        
        # For reference
        self.predict(self.args.result_file)
        metrics = self.evaluate(self.args.dataset_file, self.args.result_file)
        print(f"Initial metrics: {metrics}")
        
        for epoch in tqdm(np.arange(max_epoch)):
            
            # 각 에포크마다 전체 데이터셋을 순회
            batch_count = 0
            for batch in tqdm(train_loader, desc=f"에포크 {epoch+1}/{max_epoch}"):
                input_ids, attention_mask = self.prepare_transformer_input(batch, train_type)
                
                if train_type == TrainType.pos_train_pre:
                    losses = self.train_step_pre(input_ids, attention_mask)
                elif train_type == TrainType.neg_train_joint:
                    losses = self.train_step_joint(input_ids, attention_mask)
                
                batch_count += 1
                
                # 손실 출력
                if ((batch_count % self.args.print_freq == 0)):
                    print(f"에포크 {epoch+1}/{max_epoch}, 배치 {batch_count}\n, loss: {losses['loss']}\n, pos_similarity: {losses['pos_similarity']}\n, other_similarity: {losses['other_similarity']}")
                    if train_type == TrainType.neg_train_joint:
                        print(f"neg_similarity: {losses['neg_similarity']}")
            
            if epoch % self.args.eval_interval == 0:
                self.predict(self.args.result_file)
                self.evaluate(self.args.dataset_file, self.args.result_file)
                self.model.train()

        return None   
    
    def predict(self, result_file):
        """
        클러스터링 결과를 예측하고 결과 파일에 저장하는 함수입니다.
        
        Args:
            result_file (str): 예측 결과를 저장할 파일 경로
            
        Returns:
            dict: 각 발화문에 대한 클러스터 라벨 매핑 (utterance -> cluster_label)
            
        프로세스:
            1. 데이터로더를 통해 모든 발화문을 불러옵니다.
            2. 모델을 평가 모드로 설정합니다.
            3. 모든 발화문의 임베딩을 계산합니다.
            4. K-means 클러스터링을 수행하여 각 발화문에 클러스터 라벨을 할당합니다.
            5. 원본 데이터셋에 예측된 클러스터 라벨을 추가합니다.
            6. 결과를 파일로 저장합니다.
        """

        
        dataloader = unshuffle_dstc12_loader(self.args)
        print('---- {} prediction batches ----'.format(len(dataloader)))     
        self.model.eval()
        
        # K-means 클러스터링 수행
        all_embeddings, all_utterances = self.get_embeddings(dataloader)
        self.cluster_model.fit(all_embeddings)
        cluster_labels = self.cluster_model.predict(all_embeddings)
        
        print(f"클러스터링 완료: {self.args.n_clusters}개 클러스터")
        
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
                
        return cluster_label_map
                
    def get_embeddings(self, dataloader):
        all_embeddings = []
        all_utterances = []
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="임베딩 추출"):
                text = batch['text']
                feat = self.get_batch_token(text)
                # embeddings = self.model(feat['input_ids'].to(self.args.device), 
                #                        feat['attention_mask'].to(self.args.device), 
                #                        task_type="evaluate")
                embeddings = self.model.contrast_embed(feat['input_ids'].to(self.args.device), 
                                                     feat['attention_mask'].to(self.args.device))
                
                # 임베딩과 해당 발화문 저장
                all_embeddings.append(embeddings.detach().cpu())
                all_utterances.extend(text)
        
        # 모든 임베딩 결합
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        return all_embeddings, all_utterances

    def evaluate(self, dataset_file, result_file):
        """
        run_evaluate.py 코드를 참고하여 작성되었습니다.
        """
        # 필요한 모듈 임포트
        import sys
        import os
        
        # dstc12 패키지 경로 추가
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
        
        from dstc12.eval import (
            acc,
            nmi,
            rouge_with_multiple_references,
            cosine_similarity_with_multiple_references
        )
        from langchain_huggingface import HuggingFaceEmbeddings
        import json
        
        print("평가 진행 중...")
        
        # 데이터 로드
        with open(dataset_file) as f:
            ground_truth = [json.loads(line) for line in f]
        with open(result_file) as f:
            predictions = [json.loads(line) for line in f]
            
        # 필요한 데이터 추출
        label1_references, label2_references, label_predictions = [], [], []
        for dialog_gt, dialog_pred in zip(ground_truth, predictions):
            assert len(dialog_gt['turns']) == len(dialog_pred['turns'])
            for utterance_gt, utterance_pred in zip(dialog_gt['turns'], dialog_pred['turns']):
                assert utterance_gt['utterance_id'] == utterance_pred['utterance_id']
                if utterance_gt['theme_label'] is None:
                    continue
                uid = utterance_gt['utterance_id']
                label1_references.append(utterance_gt['theme_label']['label_1'])
                label2_references.append(utterance_gt['theme_label']['label_2'])
                label_predictions.append(utterance_pred['theme_label_predicted'])
        
        # 임베딩 모델 로드
        embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # 임베딩 계산
        reference_1_embeddings = embeddings.embed_documents(label1_references)
        reference_2_embeddings = embeddings.embed_documents(label2_references)
        predictions_embeddings = embeddings.embed_documents(label_predictions)
        
        # 평가 지표 계산
        avg_acc = acc(references=label1_references, predictions=label_predictions)
        avg_nmi = nmi(references=label1_references, predictions=label_predictions)
        avg_rouge = rouge_with_multiple_references(
            [[label_1, label_2] for label_1, label_2 in zip(label1_references, label2_references)],
            label_predictions
        )
        avg_cosine_similarity = cosine_similarity_with_multiple_references(
            (reference_1_embeddings, reference_2_embeddings),
            predictions_embeddings
        )
        
        # 결과 출력
        metrics = {
            'acc': avg_acc,
            'nmi': avg_nmi,
            'rouge_1': avg_rouge['rouge1'].fmeasure,
            'rouge_2': avg_rouge['rouge2'].fmeasure,
            'rouge_l': avg_rouge['rougeL'].fmeasure,
            'cosine_similarity': avg_cosine_similarity,
        }
        
        for metric, value in metrics.items():
            print(f'{metric}: {value:.3f}')
            
        return metrics

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
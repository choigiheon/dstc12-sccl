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
import wandb
import datetime

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
        
        # wandb 초기화 (args에 wandb 설정이 없는 경우에 대비)
        if not hasattr(args, 'wandb_project'):
            self.args.wandb_project = "SCCL-DSTC12"
        if not hasattr(args, 'wandb_entity'):
            self.args.wandb_entity = None
        
        # wandb 설정 확인 및 초기화
        if not wandb.run:
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                config=vars(args),
                name=f"SCCL-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        
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
        self.gstep = 0  # 초기 iteration 설정
        self.predict(self.args.result_file)
        metrics = self.evaluate(self.args.dataset_file, self.args.result_file)
        print("\n[초기 평가 결과]")
        
        for i in tqdm(np.arange(max_iter)):
            self.gstep = i  # 현재 iteration 업데이트
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)
                
                all_embeddings, all_utterances = self.get_embeddings(self.train_loader)
                self.cluster_model.update(all_embeddings) # 에포크가 끝날떄마다 클러스터 업데이트

            input_ids, attention_mask = self.prepare_transformer_input(batch, self.args)

            losses = self.train_step_virtual(input_ids, attention_mask, objective=TrainType.joint_train) if train_type == TrainType.joint_train else self.train_step_virtual(input_ids, attention_mask, objective=TrainType.pre_train)
                
            # wandb에 학습 손실 로깅
            wandb.log({
                "train/iteration": self.gstep,
                "train/loss": losses["loss"],
                "train/pos_mean": losses["pos_mean"],
                "train/neg_mean": losses["neg_mean"]
            })
            
            if train_type == TrainType.joint_train and "cluster_loss" in losses:
                wandb.log({"train/cluster_loss": losses["cluster_loss"]})
                
            if (i % self.args.eval_interval == 0) and (i != 0):
                self.predict(self.args.result_file)
                metrics = self.evaluate(self.args.dataset_file, self.args.result_file)
                
                self.model.train()

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==max_iter)):
                print(f"loss: {losses['loss']}, pos_mean: {losses['pos_mean']}, neg_mean: {losses['neg_mean']}")
                if train_type == TrainType.joint_train:
                    print(f"cluster_loss: {losses['cluster_loss']}")
                self.model.train()
                
        # 마지막 배치에서 클러스터 업데이트
        all_embeddings, all_utterances = self.get_embeddings(self.train_loader)
        self.cluster_model.update(all_embeddings)

        return None   
    
    def log_metrics(self, metrics, step):
        wandb.log(metrics, step=step)
    
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
                
                if embeddings.device.type != 'cpu':
                    embeddings = embeddings.cpu()
                
                # 임베딩과 해당 발화문 저장
                all_embeddings.append(embeddings.detach())
                all_utterances.extend(text)
        
        # 모든 임베딩 결합 및 numpy 변환
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        return all_embeddings, all_utterances

    def log_statics(self, dataset_file, result_file, metrics, embedding_model_name=None):
        """
        WandB를 이용하여 평가 결과를 로깅하는 함수
        
        Args:
            dataset_file: 데이터셋 파일 경로
            result_file: 결과 파일 경로
            metrics: 평가 지표 딕셔너리
            embedding_model_name: 임베딩 모델 이름 (선택적)
        """
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 평가 결과 및 파라미터 로깅
        log_dict = {
            "eval/timestamp": formatted_time,
            "eval/iteration": self.gstep,
        }
        
        # 평가 지표 로깅
        for metric_name, metric_value in metrics.items():
            log_dict[f"eval/{metric_name}"] = metric_value
        
        # 기본 파라미터 로깅
        log_dict.update({
            # 기본 설정
            "params/seed": self.args.seed,
            "params/print_freq": self.args.print_freq,
            "params/device": self.args.device,
            "params/model_name": self.args.model_name,
            "params/dropout": self.args.dropout,
            
            # 데이터셋 설정
            "params/dataset_file": dataset_file,
            "params/result_file": result_file,
            "params/num_clusters": self.args.num_clusters,
            "params/max_length": self.args.max_length,
            
            # 학습 파라미터
            "params/lr": self.args.lr,
            "params/lr_scale": self.args.lr_scale if hasattr(self.args, 'lr_scale') else None,
            "params/joint_max_iter": self.args.joint_max_iter,
            "params/pre_max_iter": self.args.pre_max_iter if hasattr(self.args, 'pre_max_iter') else None,
            
            # 대조 학습 설정
            "params/augtype": self.args.augtype if hasattr(self.args, 'augtype') else 'virtual',
            "params/batch_size": self.args.batch_size,
            "params/temperature": self.args.temperature,
            "params/eta": self.args.eta,
            
            # 클러스터링 설정
            "params/alpha": self.args.alpha if hasattr(self.args, 'alpha') else None,
            "params/use_progressive": self.args.use_progressive if hasattr(self.args, 'use_progressive') else None,
            "params/n_init": self.args.n_init if hasattr(self.args, 'n_init') else None,
            
            # 평가 설정
            "params/eval_interval": self.args.eval_interval,
        })
        
        if embedding_model_name:
            log_dict["params/embedding_model"] = embedding_model_name
            
        # None 값 제거 (wandb에서 오류 방지)
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
            
        # wandb에 로그 추가
        wandb.log(log_dict)
        
        # 콘솔 출력 (옵션)
        print(f"\n[평가 로그 - {formatted_time}, Iteration: {self.gstep}]")
        
        # 파라미터 그룹별 출력
        print("\n----- 기본 설정 -----")
        print(f"  seed: {self.args.seed}")
        print(f"  device: {self.args.device}")
        print(f"  model_name: {self.args.model_name}")
        
        print("\n----- 데이터셋 설정 -----")
        print(f"  dataset_file: {dataset_file}")
        print(f"  result_file: {result_file}")
        print(f"  max_length: {self.args.max_length}")
        print(f"  batch_size: {self.args.batch_size}")
        
        print("\n----- 학습 및 클러스터링 설정 -----")
        print(f"  num_clusters: {self.args.num_clusters}")
        print(f"  temperature: {self.args.temperature}")
        print(f"  eta: {self.args.eta}")
        if hasattr(self.args, 'alpha'):
            print(f"  alpha: {self.args.alpha}")
        if hasattr(self.args, 'use_progressive'):
            print(f"  use_progressive: {self.args.use_progressive}")
        
        print("\n----- 평가 결과 -----")
        max_key_length = max(len(key) for key in metrics.keys())
        for metric, value in metrics.items():
            print(f"  {metric:{max_key_length}}: {value:.4f}")
        print("="*50)
            
    def evaluate(self, dataset_file, result_file):
        # 필요한 모듈 임포트
        import sys
        import os
        
        # dstc12 패키지 경로 추가 (OS 독립적인 방법)
        current_dir = os.path.dirname(os.path.abspath(__file__))  # sccl 디렉토리
        project_root = os.path.dirname(current_dir)  # 프로젝트 루트 디렉토리
        src_path = os.path.join(project_root, 'src')
        if os.path.exists(src_path):
            sys.path.append(src_path)
        
        # 현재 작업 디렉토리 기준 상대 경로 추가
        src_alt_path = os.path.join(os.getcwd(), 'src')
        if os.path.exists(src_alt_path):
            sys.path.append(src_alt_path)
        
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
            
        # 로그 함수 호출
        self.log_statics(dataset_file, result_file, metrics, embedding_model_name)
            
        return metrics

"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 12/12/2021
"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


class PairConLossPositive(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLossPositive, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLoss with Euclidean Distance \n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        
        features = torch.cat([features_1, features_2], dim=0)  # 크기: [2*batch_size, hidden_dim]
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)  # 크기: [batch_size, batch_size]
        mask = mask.repeat(2, 2)  # 크기: [2*batch_size, 2*batch_size]
        mask = ~mask  # 크기: [2*batch_size, 2*batch_size]
        
        # 양성 쌍의 유클리디안 거리 계산
        pos_dist = torch.sum((features_1 - features_2)**2, dim=-1)  # 크기: [batch_size]
        
        # 모든 가능한 쌍 간의 유클리디안 거리 계산
        norm_sq = torch.sum(features**2, dim=1, keepdim=True)  # ||a||^2
        dist_matrix = norm_sq + norm_sq.t() - 2 * torch.mm(features, features.t())  # ||a-b||^2
        
        # 마스크를 사용하여 다른 샘플과의 거리만 선택
        neg_dist = dist_matrix.masked_select(mask).view(2*batch_size, -1)  # 크기: [2*batch_size, 2*batch_size-2]
        
        # 각 샘플에 대해 가장 가까운 네거티브 샘플 선택 (hard negative mining)
        closest_neg_dist, _ = neg_dist.min(dim=1)  # 크기: [2*batch_size]
        
        # pos_dist와 closest_neg_dist 비교를 위해 pos_dist 복제
        pos_dist_expanded = torch.cat([pos_dist, pos_dist], dim=0)  # 크기: [2*batch_size]
        
        # 트리플렛 손실 계산: max(0, pos_dist - neg_dist)
        triplet_loss = torch.clamp(pos_dist_expanded - closest_neg_dist, min=0)
        
        # 평균 거리 계산 (로깅용)
        pos_dist_mean = torch.mean(pos_dist)
        neg_dist_mean = torch.mean(closest_neg_dist)
        
        # 최종 손실 계산
        loss = triplet_loss.mean()
        
        return {"loss": loss, "pos_similarity": -pos_dist_mean.detach().cpu().numpy(), "other_similarity": -neg_dist_mean.detach().cpu().numpy()}
            

class PairConLossNegative(nn.Module):
    def __init__(self, temperature=0.05, negative_alpha=5.0):
        super(PairConLossNegative, self).__init__()
        self.temperature = temperature
        self.eps = 1e-09
        self.negative_alpha = negative_alpha  # 부정 쌍에 대한 가중치
        print(f"\n Initializing PairConLossNegative with Euclidean Distance \n")
    
    def entailment_loss(self, features_1_1, features_1_2, features_2_1):
        device = features_1_1.device
        batch_size = features_1_1.shape[0]
        
        features = torch.cat([features_1_1, features_1_2], dim=0)  # 크기: [2*batch_size, hidden_dim]
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)  # 크기: [batch_size, batch_size]
        mask = mask.repeat(2, 2)  # 크기: [2*batch_size, 2*batch_size]
        mask = ~mask  # 크기: [2*batch_size, 2*batch_size]
        
        # 양성 쌍의 유클리디안 거리 계산
        pos_dist = torch.sum((features_1_1 - features_1_2)**2, dim=-1)  # 크기: [batch_size]
        
        # 부정 쌍의 유클리디안 거리 계산
        neg_dist = torch.sum((features_1_1 - features_2_1)**2, dim=-1)  # 크기: [batch_size]
        
        # 확실하게 부정 쌍의 거리를 양성 쌍보다 크게 만들기 위한 손실 계산
        # 부정 쌍의 거리가 양성 쌍의 거리보다 작으면 손실 발생
        loss_margin = torch.clamp(pos_dist - neg_dist, min=0)
        
        # 부정 쌍에 대한 가중치 적용
        loss_margin = loss_margin * self.negative_alpha
        
        # 추가적으로 모든 가능한 쌍 간의 유클리디안 거리 계산
        norm_sq = torch.sum(features**2, dim=1, keepdim=True)  # ||a||^2
        dist_matrix = norm_sq + norm_sq.t() - 2 * torch.mm(features, features.t())  # ||a-b||^2
        
        # 마스크를 사용하여 다른 샘플과의 거리만 선택
        other_dist = dist_matrix.masked_select(mask).view(2*batch_size, -1)  # 크기: [2*batch_size, 2*batch_size-2]
        
        # 각 샘플에 대해 가장 가까운 네거티브 샘플 선택 (hard negative mining)
        closest_other_dist, _ = other_dist.min(dim=1)  # 크기: [2*batch_size]
        
        # pos_dist와 closest_other_dist 비교를 위해 pos_dist 복제
        pos_dist_expanded = torch.cat([pos_dist, pos_dist], dim=0)  # 크기: [2*batch_size]
        
        # 트리플렛 손실 계산
        triplet_loss = torch.clamp(pos_dist_expanded - closest_other_dist, min=0)
        
        # 최종 손실: 부정 쌍에 대한 마진 손실 + 트리플렛 손실
        loss = loss_margin.mean() + triplet_loss.mean()
        
        # 평균 거리 계산 (로깅용)
        pos_dist_mean = torch.mean(pos_dist)
        neg_dist_mean = torch.mean(neg_dist)
        other_dist_mean = torch.mean(closest_other_dist)
        
        return {"loss": loss, "pos_similarity": -pos_dist_mean.detach().cpu().numpy(), 
                "other_similarity": -other_dist_mean.detach().cpu().numpy(), 
                "neg_similarity": -neg_dist_mean.detach().cpu().numpy()}
        
    def forward(self, features_1_1, features_1_2, features_2_1, features_2_2):
        loss_entailment1 = self.entailment_loss(features_1_1, features_1_2, features_2_1)
        loss_entailment2 = self.entailment_loss(features_2_1, features_2_2, features_1_1)
        loss = (loss_entailment1["loss"] + loss_entailment2["loss"]) / 2
        pos_similarity = (loss_entailment1["pos_similarity"] + loss_entailment2["pos_similarity"]) / 2
        other_similarity = (loss_entailment1["other_similarity"] + loss_entailment2["other_similarity"]) / 2
        neg_similarity = (loss_entailment1["neg_similarity"] + loss_entailment2["neg_similarity"]) / 2
        return {"loss":loss, "pos_similarity":pos_similarity, "other_similarity":other_similarity, "neg_similarity":neg_similarity}
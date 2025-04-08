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
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        
        features = torch.cat([features_1, features_2], dim=0)  # 크기: [2*batch_size, hidden_dim]
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)  # 크기: [batch_size, batch_size]
        mask = mask.repeat(2, 2)  # 크기: [2*batch_size, 2*batch_size]
        mask = ~mask  # 크기: [2*batch_size, 2*batch_size]
        
        pos_sim = torch.sum(features_1*features_2, dim=-1)  # 크기: [batch_size]
        pos_exp = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)  # 크기: [batch_size]
        pos_exp = torch.cat([pos_exp, pos_exp], dim=0)  # 크기: [2*batch_size]
        
        other_sim = torch.mm(features, features.t().contiguous())  # 크기: [2*batch_size, 2*batch_size]
        other_exp = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)  # 크기: [2*batch_size, 2*batch_size]
        other_exp = other_exp.masked_select(mask).view(2*batch_size, -1)  # 크기: [2*batch_size, 2*batch_size-1]
        
        other_sim_mean = torch.mean(other_sim)  # 크기: 스칼라
        pos_sim_mean = torch.mean(pos_sim)  # 크기: 스칼라
        Ng = other_exp.sum(dim=-1)  # 크기: [2*batch_size]
            
        loss_pos = (- torch.log(pos_exp / (Ng+pos_exp))).mean()  # 크기: 스칼라
        
        return {"loss":loss_pos, "pos_similarity":pos_sim_mean.detach().cpu().numpy(), "other_similarity":other_sim_mean.detach().cpu().numpy()}
            

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
        
        # 유사도 계산
        pos_sim = torch.sum(features_1_1*features_1_2, dim=-1)  # 크기: [batch_size]
        other_sim = torch.mm(features, features.t().contiguous())  # 크기: [2*batch_size, 2*batch_size]
        
        # 지수 계산
        pos_exp = torch.exp(pos_sim / self.temperature)  # 크기: [batch_size]
        pos_exp = torch.cat([pos_exp, pos_exp], dim=0)  # 크기: [2*batch_size]
        other_exp = torch.exp(other_sim / self.temperature)  # 크기: [2*batch_size, 2*batch_size]
        other_exp = other_exp.masked_select(mask).view(2*batch_size, -1)  # 크기: [2*batch_size, 2*batch_size-2]
        
        Ng = other_exp.sum(dim=-1)  # 크기: [2*batch_size]
            
        loss_entailment = (- torch.log(pos_exp / (Ng+pos_exp))).mean()  # 크기: 스칼라
        
        # ------------------------------ 유클리디안 거리 계산 ------------------------------
        
        # 양성 쌍의 유클리디안 거리 계산
        pos_dist = torch.sum(((features_1_1 - features_1_2)/2.0)**2, dim=-1)  # 크기: [batch_size]
        
        # 부정 쌍의 유클리디안 거리 계산
        neg_dist = torch.sum(((features_1_1 - features_2_1)/2.0)**2, dim=-1)  # 크기: [batch_size]
        
        # 확실하게 부정 쌍의 거리를 양성 쌍보다 크게 만들기 위한 손실 계산
        # 부정 쌍의 거리가 양성 쌍의 거리보다 작으면 손실 발생
        loss_margin = torch.clamp(pos_dist - neg_dist, min=0)
        
        # 부정 쌍에 대한 가중치 적용
        loss_margin = loss_margin # * self.negative_alpha
        
        # 양성 샘플(features_1_1)과 대응되는 부정 샘플(features_2_1) 사이의 거리
        pos_neg_dist = neg_dist  # 이미 계산된 거리 재사용
        
        # 모든 샘플 연결
        all_features = torch.cat([features_1_1, features_1_2, features_2_1], dim=0)  # 크기: [3*batch_size, hidden_dim]
        
        # 양성 샘플(features_1_1)과 다른 모든 샘플 사이의 거리 계산
        # cdist 사용 후 나누기 4 적용
        pos_to_all_dist = torch.cdist(features_1_1, all_features, p=2) / 4.0  # 크기: [batch_size, 3*batch_size]
        
        # 양성 샘플에서 대응되는 부정 샘플을 제외한 나머지 샘플들 마스킹
        mask_exclude = torch.zeros(batch_size, 3*batch_size, dtype=torch.bool).to(device)
        for i in range(batch_size):
            # 자기 자신 마스킹 (features_1_1의 i번째)
            mask_exclude[i, i] = True
            # 다른 양성 샘플 마스킹 (features_1_2의 i번째)
            mask_exclude[i, i + batch_size] = True
            # 대응되는 부정 샘플은 마스킹하지 않음 (features_2_1의 i번째)
        
        # 마스킹된 부분을 큰 값으로 설정하여 최소값 계산에서 제외
        pos_to_all_dist_masked = pos_to_all_dist.clone()
        pos_to_all_dist_masked.masked_fill_(mask_exclude, float('inf'))
        
        # 양성 샘플에서 가장 가까운 다른 샘플과의 거리 계산
        closest_other_dist = torch.min(pos_to_all_dist_masked, dim=1)[0]  # 크기: [batch_size]
        
        # 트리플렛 손실 계산: 양성 샘플과 가장 가까운 다른 샘플 사이의 거리가
        # 양성 샘플과 대응되는 부정 샘플 사이의 거리보다 크도록
        triplet_loss = torch.clamp(closest_other_dist - pos_neg_dist, min=0)
        
        # 최종 손실: 부정 쌍에 대한 마진 손실 + 트리플렛 손실
        loss = loss_entailment + loss_margin.mean() + triplet_loss.mean()
        
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
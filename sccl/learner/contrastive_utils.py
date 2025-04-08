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
    def __init__(self, temperature=0.05):
        super(PairConLossNegative, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLossNegative \n")
    
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
        neg_sim = torch.sum(features_1_1*features_2_1, dim=-1)  # 크기: [batch_size]
        
        # 지수 계산
        pos_exp = torch.exp(pos_sim / self.temperature)  # 크기: [batch_size]
        pos_exp = torch.cat([pos_exp, pos_exp], dim=0)  # 크기: [2*batch_size]
        other_exp = torch.exp(other_sim / self.temperature)  # 크기: [2*batch_size, 2*batch_size]
        other_exp = other_exp.masked_select(mask).view(2*batch_size, -1)  # 크기: [2*batch_size, 2*batch_size-1]
        neg_exp = torch.exp(neg_sim / self.temperature)  # 크기: [batch_size]
        neg_exp = torch.cat([neg_exp, neg_exp], dim=0)  # 크기: [2*batch_size]
        
        other_mean = torch.mean(other_sim)  # 크기: 스칼라
        pos_mean = torch.mean(pos_sim)  # 크기: 스칼라
        neg_mean = torch.mean(neg_sim)  # 크기: 스칼라
        Ng = other_exp.sum(dim=-1)  # 크기: [2*batch_size]
            
        loss_pos = (- torch.log(pos_exp / (Ng+pos_exp+neg_exp))).mean()  # 크기: 스칼라
        return {"loss":loss_pos, "pos_similarity":pos_mean.detach().cpu().numpy(), "other_similarity":other_mean.detach().cpu().numpy(), "neg_similarity":neg_mean.detach().cpu().numpy()}
        
    def forward(self, features_1_1, features_1_2, features_2_1, features_2_2):
        loss_entailment1 = self.entailment_loss(features_1_1, features_1_2, features_2_1)
        loss_entailment2 = self.entailment_loss(features_2_1, features_2_2, features_1_1)
        loss = (loss_entailment1["loss"] + loss_entailment2["loss"]) / 2
        pos_similarity = (loss_entailment1["pos_similarity"] + loss_entailment2["pos_similarity"]) / 2
        other_similarity = (loss_entailment1["other_similarity"] + loss_entailment2["other_similarity"]) / 2
        neg_similarity = (loss_entailment1["neg_similarity"] + loss_entailment2["neg_similarity"]) / 2
        return {"loss":loss, "pos_similarity":pos_similarity, "other_similarity":other_similarity, "neg_similarity":neg_similarity}
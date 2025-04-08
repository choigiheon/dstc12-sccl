"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel
from training import TrainType
# from transformers import AutoModel, AutoTokenizer
from copy import deepcopy

class SCCLModel(nn.Module):
    def __init__(self, model, tokenizer, alpha=1.0):
        super(SCCLModel, self).__init__()
        
        self.tokenizer = tokenizer
        self.model = model
        self.emb_size = self.model.config.hidden_size
        self.alpha = alpha
        
        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        # init 
        self.contrast_head.apply(self.init_weights)
      
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, task_type="evaluate"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        
        elif task_type == TrainType.joint_train:
            input_ids_1, input_ids_2, input_ids_3, input_ids_4 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2, attention_mask_3, attention_mask_4 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)
            mean_output_4 = self.get_mean_embeddings(input_ids_4, attention_mask_4)
            return mean_output_1, mean_output_2, mean_output_3, mean_output_4
        
        elif task_type == TrainType.inter_train:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            return mean_output_1, mean_output_2
        
        elif task_type == TrainType.pre_train:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1) # input_ids_1 == input_ids_2
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)  # attention_mask_1 == attention_mask_2
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            return mean_output_1, mean_output_2
        
        elif task_type == "explicit":
            input_ids_1, input_ids_2, input_ids_3 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2, attention_mask_3 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)
            return mean_output_1, mean_output_2, mean_output_3
        
        else:
            raise Exception("TRANSFORMER ENCODING TYPE ERROR! OPTIONS: [EVALUATE, SIMCSE, EXPLICIT]")
      
    
    def get_mean_embeddings(self, input_ids, attention_mask):
        model_output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(model_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2
    
    def contrast_embed(self, input_ids, attention_mask):
        model_output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(model_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        contrast_output = self.contrast_head(mean_output)
        return contrast_output
    
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1


    def contrast_logits_negative(self, embd1_1, embd1_2, embd2_1, embd2_2):
        emb1_1 = F.normalize(self.contrast_head(embd1_1), dim=1)
        emb1_2 = F.normalize(self.contrast_head(embd1_2), dim=1)
        emb2_1 = F.normalize(self.contrast_head(embd2_1), dim=1)
        emb2_2 = F.normalize(self.contrast_head(embd2_2), dim=1)
        
        return emb1_1, emb1_2, emb2_1, emb2_2
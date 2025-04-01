"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}


def get_optimizer(model, args):
    
    optimizer = torch.optim.Adam([
        {'params':model.model.parameters()}, 
        {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}
    ], lr=args.lr)
    
    print(optimizer)
    return optimizer 
    

def get_model(args):
    
    # if args.use_pretrain == "SBERT":
    #     bert_model = get_sbert(args)
    #     tokenizer = bert_model[0].tokenizer
    #     model = bert_model[0].auto_model
    #     print("..... loading Sentence-BERT !!!")
    # else:
    #     config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
    #     model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
    #     tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
    #     print("..... loading plain BERT !!!")
    
    config = AutoConfig.from_pretrained(args.model_name)
    config.hidden_dropout_prob = args.dropout
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=args.model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
    return model, tokenizer


def get_sbert(args):
    sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    return sbert









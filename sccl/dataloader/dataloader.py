"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import json
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset
from loguru import logger

class SimCSEAugSamples(Dataset):
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx]}

class SimCSEAugSamplesPairs(Dataset):
    def __init__(self, train_x1, train_x2):
        self.train_x1 = train_x1
        self.train_x2 = train_x2

    def __len__(self):
        return len(self.train_x1)
    
    def __getitem__(self, idx):
        return {'text_1': self.train_x1[idx], 'text_2': self.train_x2[idx]  }
    
class ExplitAugSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'augmentation_1': self.train_x1[idx], 'augmentation_2': self.train_x2[idx], 'label': self.train_y[idx]}
       

def explict_augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_text1 = train_data[args.augmentation_1].fillna('.').values
    train_text2 = train_data[args.augmentation_2].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = ExplitAugSamples(train_text, train_text1, train_text2, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


def simcse_augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = SimCSEAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    return train_loader


def unshuffle_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = SimCSEAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader

def unshuffle_dstc12_loader(args):
    with open(file=args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]
    themed_utterances = [] # ordered
    for dialogue in dataset:
        for turn in dialogue['turns']:
            if turn['theme_label'] is not None:
                themed_utterances.append(turn['utterance'])
    
    train_dataset = SimCSEAugSamples(list(themed_utterances))
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)   
    return train_loader
    

def dstc12_theme_loader(args):
    with open(args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]
    themed_utterances = set([])
    for dialogue in dataset:
        for turn in dialogue['turns']:
            if turn['theme_label'] is not None:
                themed_utterances.add(turn['utterance'])
    
    train_dataset = SimCSEAugSamples(list(themed_utterances))
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    return train_loader

def dstc12_all_loader(args):
    with open(args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]
    all_utterances = set([])
    for dialogue in dataset:
        for turn in dialogue['turns']:
            all_utterances.add(turn['utterance'])

    train_dataset = SimCSEAugSamples(list(all_utterances))
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader

def dstc12_loader_with_negative(args):
    with open(args.preference_file) as f:
        dataset = json.load(f)
    negative_pairs = dataset["cannot_link"]
    
    with open(args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]
        
    # 대화 ID와 발화 매핑을 위한 딕셔너리 생성
    utterance_map = {}
    for dialogue in dataset:
        for (turn, utterance) in enumerate(dialogue['turns']):
            utterance_id = utterance['utterance_id']
            utterance_map[utterance_id] = utterance['utterance']
    # 부정적 쌍에서 텍스트 추출
    text_1 = []
    text_2 = []
    for id1, id2 in negative_pairs:
        if id1 in utterance_map and id2 in utterance_map:
            text_1.append(utterance_map[id1])
            text_2.append(utterance_map[id2])
    
    train_dataset = SimCSEAugSamplesPairs(text_1, text_2)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    return train_loader

def dstc12_loader_with_positive(args):
    with open(args.preference_file) as f:
        dataset = json.load(f)
    positive_pairs = dataset["should_link"]
    
    with open(args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]
        
    # 대화 ID와 발화 매핑을 위한 딕셔너리 생성
    utterance_map = {}
    for dialogue in dataset:
        for (turn, utterance) in enumerate(dialogue['turns']):
            utterance_id = utterance['utterance_id']
            utterance_map[utterance_id] = utterance['utterance']
    # 부정적 쌍에서 텍스트 추출
    text_1 = []
    text_2 = []
    for id1, id2 in positive_pairs:
        if id1 in utterance_map and id2 in utterance_map:
            text_1.append(utterance_map[id1])
            text_2.append(utterance_map[id2])
    train_dataset = SimCSEAugSamplesPairs(text_1, text_2)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    return train_loader
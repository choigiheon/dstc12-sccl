"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from tqdm import tqdm

def get_mean_embeddings(bert, input_ids, attention_mask):
        bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    
def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return {'input_ids': token_feat['input_ids'], 'attention_mask': token_feat['attention_mask']}


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length, args):
    
    print("임베딩 추출 중...")
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="임베딩 추출"):
        text = batch['text']
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)
        
        if i == 0:
            all_embeddings = corpus_embeddings.detach().numpy()
        else:
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().numpy()), axis=0)
            
    print(f"총 {all_embeddings.shape[0]}개 발화문의 임베딩 추출 완료")

    # Perform KMeans clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes, random_state=args.seed)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, pred_labels:{}".format(all_embeddings.shape, len(pred_labels)))

    confusion.optimal_assignment(num_classes)
    print("Iterations:{}, centers:{}".format(clustering_model.n_iter_, clustering_model.cluster_centers_.shape))
    
    return clustering_model.cluster_centers_




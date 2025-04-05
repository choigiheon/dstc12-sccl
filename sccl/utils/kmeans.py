"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
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


def get_kmeans_centers(model, tokenizer, train_loader, num_classes, max_length, args):
    
    print("임베딩 추출 중...")
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="임베딩 추출"):
        text = batch['text']
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(model, **tokenized_features)
        
        # GPU 텐서를 CPU로 이동시킨 후 numpy 변환
        if torch.is_tensor(corpus_embeddings) and corpus_embeddings.device.type != 'cpu':
            corpus_embeddings_np = corpus_embeddings.cpu().detach().numpy()
        else:
            corpus_embeddings_np = corpus_embeddings.detach().numpy()
        
        if i == 0:
            all_embeddings = corpus_embeddings_np
        else:
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings_np), axis=0)
            
    print(f"총 {all_embeddings.shape[0]}개 발화문의 임베딩 추출 완료")

    # Perform KMeans clustering
    clustering_model = KMeans(n_clusters=num_classes, random_state=args.seed)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, pred_labels:{}".format(all_embeddings.shape, len(pred_labels)))

    print("Iterations:{}, centers:{}".format(clustering_model.n_iter_, clustering_model.cluster_centers_.shape))
    
    return clustering_model.cluster_centers_



"""
Progressive KMeans
"""
class ProgressiveKMeans:
    def __init__(self, n_clusters, max_length, args, use_progressive=True):
        self.n_clusters = n_clusters
        self.max_length = max_length
        self.args = args
        self.cluster_centers = None
        self.high_score_centers = None
        self.labels = None
        self.use_progressive = use_progressive
        self.n_init = args.n_init
    
    def update(self, all_embeddings):
        if self.use_progressive and self.high_score_centers is not None:
            # GPU 텐서인 high_score_centers를 CPU로 이동시킨 후 numpy로 변환
            if torch.is_tensor(self.high_score_centers) and self.high_score_centers.device.type != 'cpu':
                init_centers = self.high_score_centers.cpu().numpy()
            else:
                init_centers = self.high_score_centers
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, init=init_centers, random_state=self.args.seed)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, init="k-means++", random_state=self.args.seed)
        
        # GPU 텐서를 CPU로 이동한 후 numpy로 변환
        if torch.is_tensor(all_embeddings):
            all_embeddings_np = all_embeddings.cpu().numpy()
        else:
            all_embeddings_np = all_embeddings
            
        kmeans.fit(all_embeddings_np)
        self.cluster_centers = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        
        self.high_score_centers = self.calculate_high_score_centers(all_embeddings_np, self.cluster_centers, self.labels)
        
    def predict(self, all_embeddings):
        if self.use_progressive and self.high_score_centers is not None:
            # GPU 텐서인 high_score_centers를 CPU로 이동시킨 후 numpy로 변환
            if torch.is_tensor(self.high_score_centers) and self.high_score_centers.device.type != 'cpu':
                init_centers = self.high_score_centers.cpu().numpy()
            else:
                init_centers = self.high_score_centers
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, init=init_centers, random_state=self.args.seed)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, init="k-means++", random_state=self.args.seed)
        
        # GPU 텐서를 CPU로 이동한 후 numpy로 변환
        if torch.is_tensor(all_embeddings):
            all_embeddings_np = all_embeddings.cpu().numpy()
        else:
            all_embeddings_np = all_embeddings
        
        kmeans.fit(all_embeddings_np)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        high_score_centers = self.calculate_high_score_centers(all_embeddings_np, cluster_centers, labels)
        return labels, high_score_centers
    
    def calculate_high_score_centers(self, all_embeddings, cluster_centers, labels):
        # GPU 텐서를 CPU로 이동한 후 numpy로 변환
        if torch.is_tensor(all_embeddings) and all_embeddings.is_cuda:
            all_embeddings_np = all_embeddings.cpu().numpy()
        else:
            all_embeddings_np = all_embeddings
            
        # 실루엣 점수 계산
        silhouette_vals = silhouette_samples(all_embeddings_np, labels)
        
        # 각 클러스터별 상위 20개 실루엣 점수를 가진 샘플의 centroid 계산
        high_score_centers = np.zeros_like(cluster_centers)
        
        for i in range(self.n_clusters):
            # i번째 클러스터에 속한 샘플들의 인덱스
            cluster_indices = np.where(labels == i)[0]
            
            # 해당 클러스터의 실루엣 점수
            cluster_silhouette_vals = silhouette_vals[cluster_indices]
            
            # 실루엣 점수 기준으로 상위 20개 샘플 선택 (클러스터 크기가 20보다 작을 경우 전체 선택)
            if len(cluster_indices) > 20:
                top_indices = cluster_indices[np.argsort(-cluster_silhouette_vals)[:20]]
            else:
                top_indices = cluster_indices
            
            # 선택된 샘플들의 평균 계산하여 high_score_centers에 저장
            high_score_centers[i] = np.mean(all_embeddings_np[top_indices], axis=0)
            
        return torch.tensor(high_score_centers, device=self.args.device)
        
    def get_hsc(self):
        return self.high_score_centers
    
    def get_clusters(self):
        return self.cluster_centers, self.labels

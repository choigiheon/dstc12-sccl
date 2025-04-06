# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from argparse import ArgumentParser
import json
import os
import copy
import collections

import getpass
import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel
from sklearn.cluster import KMeans

from dstc12.prompts import LABEL_CLUSTERS_PROMPT
from dstc12.utils import get_llm, DotAllRegexParser
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-file', type=str, default="./dstc12-data/AppenBanking/all.jsonl")
    # parser.add_argument('preferences_file', type=str)
    parser.add_argument('--result-file', type=str, default="./appen_banking_predicted.jsonl")
    parser.add_argument('--n-clusters', type=int, default=14)
    parser.add_argument('--random-state', type=int, default=42)
    # parser.add_argument('--embedding-model-name', type=str, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--llm-name', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--cluster-label-map', type=str, default='./cluster_label_map.json')
    return parser.parse_args()


# def find_second_closest_cluster(emb, centroids):
#     distances = [np.linalg.norm(emb - centroid) for centroid in centroids]
#     sorted_indices = np.argsort(distances)
#     return sorted_indices[1]


# def apply_preferences_to_clusters(utterances, utterance_embs, cluster_labels, cluster_centroids, shouldlink_pairs, cannot_link_pairs):
#     assert len(utterances) == len(cluster_labels)

#     datapoint_modification_counter = collections.defaultdict(lambda: 0)

#     utterance_cluster_mapping = collections.defaultdict(lambda: -1)
#     utterance_idx_mapping = collections.defaultdict(lambda: -1)
#     for utt_idx, cluster_label in enumerate(cluster_labels):
#         utterance = utterances[utt_idx]
#         utterance_cluster_mapping[utterance] = cluster_label
#         utterance_idx_mapping[utterance] = utt_idx
#     modified_cluster_labels = copy.deepcopy(cluster_labels)
#     for utt_a, utt_b in shouldlink_pairs:
#         cluster_a, cluster_b = utterance_cluster_mapping[utt_a], utterance_cluster_mapping[utt_b]
#         if cluster_a != cluster_b:
#             utt_b_idx = utterance_idx_mapping[utt_b]
#             modified_cluster_labels[utt_b_idx] = cluster_a
#             utterance_cluster_mapping[utt_b] = cluster_a
#             datapoint_modification_counter[utt_b_idx] += 1
#     for utt_a, utt_b in cannot_link_pairs:
#         cluster_a, cluster_b = utterance_cluster_mapping[utt_a], utterance_cluster_mapping[utt_b]
#         if cluster_a == cluster_b:
#             utt_b_idx = utterance_idx_mapping[utt_b]
#             utt_b_new_cluster = find_second_closest_cluster(utterance_embs[utt_b_idx], cluster_centroids)
#             modified_cluster_labels[utt_b_idx] = utt_b_new_cluster
#             utterance_cluster_mapping[utt_b] = utt_b_new_cluster
#             datapoint_modification_counter[utt_b_idx] += 1
#     return modified_cluster_labels


def main(utterances, linking_preferences, embedding_model_name, llm_name, n_clusters, random_state):
    llm = get_llm(llm_name)
    chain = (
        LABEL_CLUSTERS_PROMPT |
        llm |
        RunnableParallel(
            theme_label=DotAllRegexParser(regex=r'<theme_label>(.*?)</theme_label>', output_keys=['theme_label']),
            theme_label_explanation=DotAllRegexParser(regex=r'<theme_label_explanation>(.*?)</theme_label_explanation>', output_keys=['theme_label_explanation'])
        )
     )
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    # query_embeddings = [embeddings.embed_query(utterance) for utterance in tqdm.tqdm(utterances)]
    # kmeans = KMeans(n_clusters=n_clusters, n_init=1, init='k-means++', random_state=random_state)
    # kmeans.fit(query_embeddings)
    # clusters = kmeans.labels_
    # centroids = kmeans.cluster_centers_
    # clusters_with_preferences = apply_preferences_to_clusters(
    #     utterances,
    #     query_embeddings,
    #     clusters,
    #     centroids,
    #     linking_preferences['should_link'],
    #     linking_preferences['cannot_link']
    # )
    cluster_with_label = json.load(open(args.cluster_label_map)) # prefernce가 이미 clustering에 적용되었다고 가정함.
    clustered_utterances = [[] for _ in range(n_clusters)]
    for i, utterance in enumerate(iterable=cluster_with_label):
        clustered_utterances[cluster_with_label[utterance]].append(utterance)
    cluster_label_map = {}
    for i, cluster in tqdm.tqdm(enumerate(clustered_utterances)):
        outputs_parsed = chain.invoke({'utterances': '\n'.join(cluster)})
        for utterance in cluster:
            cluster_label_map[utterance] = outputs_parsed['theme_label']['theme_label']
    return cluster_label_map



if __name__ == '__main__':
    args = parse_args()

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

    with open(args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]
    themed_utterances = set([])
    for dialogue in dataset:
        for turn in dialogue['turns']:
            if turn['theme_label'] is not None:
                themed_utterances.add(turn['utterance'])

    cluster_label_map = main(
        list(themed_utterances),
        linking_preferences=None,
        embedding_model_name=None,
        llm_name=args.llm_name,
        n_clusters=args.n_clusters,
        random_state=args.random_state
    )
    dataset_predicted = copy.deepcopy(dataset)
    for dialogue in dataset_predicted:
        for turn in dialogue['turns']:
            if turn['theme_label'] is not None:
                turn['theme_label_predicted'] = cluster_label_map[turn['utterance']]
    with open(args.result_file, 'w') as result_out:
        for dialogue in dataset_predicted:
            print(json.dumps(dialogue), file=result_out)
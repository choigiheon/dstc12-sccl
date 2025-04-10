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
import wandb


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-file', type=str, default="./dstc12-data/AppenBanking/all copy.jsonl")
    parser.add_argument('--result-file', type=str, default="./appen_banking_predicted.jsonl")
    parser.add_argument('--n-clusters', type=int, default=14)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--llm-name', type=str, default='')
    parser.add_argument('--cluster-label-map', type=str, default='./cluster_label_map.json')
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def main(utterances, linking_preferences, embedding_model_name, llm_name, n_clusters, random_state):
    llm = get_llm(llm_name, device=args.device)
    chain = (
        LABEL_CLUSTERS_PROMPT |
        llm |
        RunnableParallel(
            theme_label=DotAllRegexParser(regex=r'<theme_label>(.*?)</theme_label>', output_keys=['theme_label']),
            theme_label_explanation=DotAllRegexParser(regex=r'<theme_label_explanation>(.*?)</theme_label_explanation>', output_keys=['theme_label_explanation'])
        )
    )
    cluster_with_label = json.load(open(args.cluster_label_map)) # prefernce가 이미 clustering에 적용되었다고 가정함.
    clustered_utterances = [[] for _ in range(n_clusters)]
    for i, utterance in enumerate(iterable=cluster_with_label):
        clustered_utterances[cluster_with_label[utterance]].append(utterance)
    cluster_label_map = {}
    for i, cluster in tqdm.tqdm(enumerate(clustered_utterances)):
        
        try:
            # outputs_parsed = chain.invoke({'utterances': '\n'.join(cluster)})
            outputs_parsed = {}
            outputs_parsed['theme_label'] = {}
            outputs_parsed['theme_label']['theme_label'] = f"theme_{i}"
        except Exception as e:
            print("오류가 발생한 프롬프트: {}".format({'utterances': '\n'.join(cluster)}))
            raise e
        for utterance in cluster:
            cluster_label_map[utterance] = outputs_parsed['theme_label']['theme_label']
    return cluster_label_map



if __name__ == '__main__':
    args = parse_args()

    # wandb 초기화 및 파라미터 로깅
    wandb.init(project="theme-detection", name="theme-detection-run")
    wandb.config.update({
        "dataset_file": args.dataset_file,
        "result_file": args.result_file,
        "n_clusters": args.n_clusters,
        "random_state": args.random_state,
        "llm_name": args.llm_name,
        "cluster_label_map": args.cluster_label_map,
        "device": args.device
    })

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
    
    # wandb 세션 종료
    wandb.finish()
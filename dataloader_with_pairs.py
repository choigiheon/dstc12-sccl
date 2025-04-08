import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

class DSTC12PairsDataset(Dataset):
    def __init__(self, preference_file, dataset_file, tokenizer, max_length=100):
        """
        DSTC12 데이터에서 preference pairs를 사용하는 데이터셋 클래스
        
        Args:
            preference_file (str): preference_pairs.json 파일 경로
            dataset_file (str): all.jsonl 데이터셋 파일 경로
            tokenizer: 사용할 토크나이저
            max_length (int): 최대 시퀀스 길이
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # utterance ID와 텍스트 매핑 생성
        self.utterance_map = self._create_utterance_id_to_text_map(dataset_file)
        print(f"총 매핑된 utterance: {len(self.utterance_map)}")
        
        # preference pairs 로드
        self.should_link_pairs, self.cannot_link_pairs = self._load_preference_pairs(preference_file)
        print(f"총 should_link 쌍: {len(self.should_link_pairs)}")
        print(f"총 cannot_link 쌍: {len(self.cannot_link_pairs)}")
        
        # 유효한 쌍만 필터링
        self.valid_should_link_pairs = []
        self.valid_cannot_link_pairs = []
        
        for pair in self.should_link_pairs:
            if pair[0] in self.utterance_map and pair[1] in self.utterance_map:
                self.valid_should_link_pairs.append(pair)
                
        for pair in self.cannot_link_pairs:
            if pair[0] in self.utterance_map and pair[1] in self.utterance_map:
                self.valid_cannot_link_pairs.append(pair)
                
        print(f"유효한 should_link 쌍: {len(self.valid_should_link_pairs)}")
        print(f"유효한 cannot_link 쌍: {len(self.valid_cannot_link_pairs)}")
        
        # 모든 유효한 쌍 결합
        self.all_pairs = []
        for pair in self.valid_should_link_pairs:
            self.all_pairs.append((pair[0], pair[1], 1))  # 1은 should_link를 나타냄
            
        for pair in self.valid_cannot_link_pairs:
            self.all_pairs.append((pair[0], pair[1], 0))  # 0은 cannot_link를 나타냄
            
        print(f"총 유효한 쌍: {len(self.all_pairs)}")
    
    def _load_preference_pairs(self, preference_file):
        """
        preference_pairs.json 파일에서 should_link와 cannot_link 쌍을 로드합니다.
        """
        with open(preference_file, 'r', encoding='utf-8') as f:
            preference_pairs = json.load(f)
        
        return preference_pairs["should_link"], preference_pairs["cannot_link"]
    
    def _create_utterance_id_to_text_map(self, dataset_file):
        """
        all.jsonl 파일에서 utterance_id를 키로 하고 utterance 텍스트를 값으로 하는 딕셔너리를 생성합니다.
        """
        utterance_map = {}
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading utterances from dataset"):
                conversation = json.loads(line)
                for turn in conversation["turns"]:
                    utterance_id = turn.get("utterance_id")
                    utterance_text = turn.get("utterance")
                    if utterance_id and utterance_text:
                        utterance_map[utterance_id] = utterance_text
        
        return utterance_map
    
    def __len__(self):
        return len(self.all_pairs)
    
    def __getitem__(self, idx):
        utterance_id_1, utterance_id_2, label = self.all_pairs[idx]
        
        text_1 = self.utterance_map[utterance_id_1]
        text_2 = self.utterance_map[utterance_id_2]
        
        # 텍스트 토큰화
        encoding_1 = self.tokenizer(
            text_1,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding_2 = self.tokenizer(
            text_2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 배치 차원 제거
        for key in encoding_1:
            encoding_1[key] = encoding_1[key].squeeze(0)
            
        for key in encoding_2:
            encoding_2[key] = encoding_2[key].squeeze(0)
        
        return {
            'text_1': text_1,
            'text_2': text_2,
            'utterance_id_1': utterance_id_1,
            'utterance_id_2': utterance_id_2,
            'input_ids_1': encoding_1['input_ids'],
            'attention_mask_1': encoding_1['attention_mask'],
            'input_ids_2': encoding_2['input_ids'],
            'attention_mask_2': encoding_2['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }


class DSTC12PairTrainSampler:
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        학습을 위한 샘플러 클래스
        
        Args:
            dataset: DSTC12PairsDataset 인스턴스
            batch_size: 배치 크기
            shuffle: 셔플 여부
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        # 배치 단위로 인덱스 반환
        batch_indices = []
        for idx in self.indices:
            batch_indices.append(idx)
            
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        
        # 마지막 배치가 있으면 반환
        if batch_indices:
            yield batch_indices
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def dstc12_pairs_loader(args):
    """
    DSTC12 데이터에서 preference pairs를 사용하는 데이터로더를 생성합니다.
    
    Args:
        args: 필요한 인수를 포함하는 객체
            args.preference_file: preference_pairs.json 파일 경로
            args.dataset_file: all.jsonl 데이터셋 파일 경로
            args.model_name: 사용할 모델 이름
            args.max_length: 최대 시퀀스 길이
            args.batch_size: 배치 크기
            
    Returns:
        train_loader: 학습용 데이터로더
    """
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 데이터셋 생성
    dataset = DSTC12PairsDataset(
        preference_file=args.preference_file,
        dataset_file=args.dataset_file,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # 데이터로더 생성
    sampler = DSTC12PairTrainSampler(dataset, args.batch_size)
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0
    )
    
    return train_loader


if __name__ == "__main__":
    # 테스트 코드
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference_file", type=str, default="./dstc12-data/AppenBanking/preference_pairs.json",
                        help="preference_pairs.json 파일 경로")
    parser.add_argument("--dataset_file", type=str, default="./dstc12-data/AppenBanking/all.jsonl",
                        help="all.jsonl 데이터셋 파일 경로")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="사용할 모델 이름")
    parser.add_argument("--max_length", type=int, default=100, help="최대 시퀀스 길이")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    
    args = parser.parse_args()
    
    # 데이터로더 테스트
    train_loader = dstc12_pairs_loader(args)
    
    # 첫 번째 배치 출력
    batch = next(iter(train_loader))
    print("배치 예시:")
    print("Text 1:", batch["text_1"][0])
    print("Text 2:", batch["text_2"][0])
    print("Label:", batch["label"][0].item())
    print("입력 ID 1 크기:", batch["input_ids_1"].shape)
    print("입력 ID 2 크기:", batch["input_ids_2"].shape) 
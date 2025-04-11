import torch
import json
import argparse
import os
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from transformers import AutoTokenizer
import string
from tqdm import tqdm

# 필요한 NLTK 데이터 다운로드
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)

class UtteranceUtils:
    def __init__(self):
        pretrained_lm = 'sentence-transformers/gtr-t5-base'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {k:v for v, k in vocab.items()}
    
    def encode(self, docs, max_len=512):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks
    
    def extract_utterances(self, dataset_file):
        """
        all.jsonl 파일에서 모든 utterance를 추출합니다.
        """
        utterances = []
        theme_utterances = []
        utterance_map = {}  # utterance_id -> text 매핑
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Extracting utterances from dataset"):
                conversation = json.loads(line)
                for turn in conversation["turns"]:
                    utterance_text = turn.get("utterance")
                    utterance_id = turn.get("utterance_id")
                    if utterance_text:
                        utterances.append(utterance_text)
                    if utterance_id and utterance_text:
                        utterance_map[utterance_id] = utterance_text
                        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Extracting theme utterances from dataset"):
                conversation = json.loads(line)
                for turn in conversation["turns"]:
                    if turn.get('theme_label') is not None:
                        theme_utterances_text = turn.get("utterance")
                        if theme_utterances_text:
                            theme_utterances.append(theme_utterances_text)
        
        # 중복 제거 및 정렬
        print(f"총 추출된 unique utterance: {len(utterances)}개")
        print(f"총 추출된 unique theme utterance: {len(theme_utterances)}개")
        print(f"총 utterance ID 매핑: {len(utterance_map)}개")
        return utterances, theme_utterances, utterance_map
    
    def load_preference_pairs(self, preference_file, utterance_map):
        """
        preference_pairs.json 파일에서 positive/negative 쌍을 로드합니다.
        """
        with open(preference_file, 'r', encoding='utf-8') as f:
            preference_pairs = json.load(f)
        
        positive_pairs = []  # should_link
        negative_pairs = []  # cannot_link
        
        # should_link 쌍 로드 (positive)
        for pair in preference_pairs.get("should_link", []):
            if pair[0] in utterance_map and pair[1] in utterance_map:
                positive_pairs.append((utterance_map[pair[0]], utterance_map[pair[1]]))
                
        # cannot_link 쌍 로드 (negative)
        for pair in preference_pairs.get("cannot_link", []):
            if pair[0] in utterance_map and pair[1] in utterance_map:
                negative_pairs.append((utterance_map[pair[0]], utterance_map[pair[1]]))
        
        print(f"유효한 positive(should_link) 쌍: {len(positive_pairs)}개")
        print(f"유효한 negative(cannot_link) 쌍: {len(negative_pairs)}개")
        
        return positive_pairs, negative_pairs
    
    def save_texts_txt(self, docs, theme_docs, output_dir):
        """
        추출된 텍스트를 texts.txt 파일로 저장합니다.
        """
        output_file = os.path.join(output_dir, "texts.txt")
        theme_output_file = os.path.join(output_dir, "theme_texts.txt")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving texts to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in docs:
                f.write(f"{doc}\n")
        with open(theme_output_file, 'w', encoding='utf-8') as f:
            for doc in theme_docs:
                f.write(f"{doc}\n")
        print(f"텍스트가 {output_file}에 성공적으로 저장되었습니다.")
        print(f"텍스트가 {theme_output_file}에 성공적으로 저장되었습니다.")
    
    def _create_filter_idx(self):
        """스톱워드와 부적절한 품사 토큰 필터링 인덱스 생성"""
        stop_words = set(stopwords.words('english'))
        filter_idx = []
        valid_pos = ["NOUN", "VERB", "ADJ"]
        
        for i in self.inv_vocab:
            token = self.inv_vocab[i]
            if token in stop_words or token.startswith('##') \
            or token in string.punctuation or token.startswith('[') \
            or pos_tag([token], tagset='universal')[0][-1] not in valid_pos:
                filter_idx.append(i)
        
        return filter_idx
    
    def create_dataset(self, dataset_dir, text_file, loader_name, max_len=512):
        """
        texts.txt 파일을 읽어 PT 파일로 저장합니다.
        """
        os.makedirs(dataset_dir, exist_ok=True)
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
            return data
        
        print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
        with open(os.path.join(dataset_dir, text_file), encoding="utf-8") as corpus:
            docs = [doc.strip() for doc in corpus.readlines()]
        
        print(f"Converting texts into tensors.")
        input_ids, attention_masks = self.encode(docs, max_len)
        
        print(f"Saving encoded texts into {loader_file}")
        filter_idx = self._create_filter_idx()
        
        valid_pos = attention_masks.clone()
        for i in filter_idx:
            valid_pos[input_ids == i] = 0
        
        data = {"input_ids": input_ids, "attention_masks": attention_masks, "valid_pos": valid_pos}
        torch.save(data, loader_file)
        
        return data
    
    def create_preference_dataset(self, dataset_dir, positive_pairs, negative_pairs, loader_name, max_len=512):
        """
        Preference pair 데이터를 PT 파일로 저장합니다.
        """
        os.makedirs(dataset_dir, exist_ok=True)
        loader_file = os.path.join(dataset_dir, loader_name)
        
        if os.path.exists(loader_file):
            print(f"Loading encoded preference pairs from {loader_file}")
            data = torch.load(loader_file)
            return data
        
        # 모든 쌍을 하나의 리스트로 결합
        all_pairs = []
        for pair in positive_pairs:
            all_pairs.append((pair[0], pair[1], 1))  # 1은 positive를 나타냄
            
        for pair in negative_pairs:
            all_pairs.append((pair[0], pair[1], -1))  # -1은 negative를 나타냄
        
        print(f"총 preference pair: {len(all_pairs)}개")
        
        # 필터링 인덱스 생성
        filter_idx = self._create_filter_idx()
        
        # 모든 쌍을 처리하여 텐서로 변환
        input_ids1_list = []
        attention_mask1_list = []
        valid_pos1_list = []
        input_ids2_list = []
        attention_mask2_list = []
        valid_pos2_list = []
        relation_list = []
        
        for text1, text2, relation in tqdm(all_pairs, desc="인코딩 preference pairs"):
            # 첫 번째 텍스트 인코딩
            encoding1 = self.tokenizer.batch_encode_plus(
                [text1], 
                max_length=max_len, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            
            # 두 번째 텍스트 인코딩
            encoding2 = self.tokenizer.batch_encode_plus(
                [text2], 
                max_length=max_len, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            
            # 유효한 위치 마스크 생성
            valid_pos1 = encoding1['attention_mask'].clone()
            for i in filter_idx:
                valid_pos1[encoding1['input_ids'] == i] = 0
                
            valid_pos2 = encoding2['attention_mask'].clone()
            for i in filter_idx:
                valid_pos2[encoding2['input_ids'] == i] = 0
            
            # 배열에 추가
            input_ids1_list.append(encoding1['input_ids'])
            attention_mask1_list.append(encoding1['attention_mask'])
            valid_pos1_list.append(valid_pos1)
            input_ids2_list.append(encoding2['input_ids'])
            attention_mask2_list.append(encoding2['attention_mask'])
            valid_pos2_list.append(valid_pos2)
            relation_list.append(torch.tensor([[relation]], dtype=torch.long))
        
        # 텐서로 쌓기
        input_ids1 = torch.cat(input_ids1_list, dim=0)
        attention_mask1 = torch.cat(attention_mask1_list, dim=0)
        valid_pos1 = torch.cat(valid_pos1_list, dim=0)
        input_ids2 = torch.cat(input_ids2_list, dim=0)
        attention_mask2 = torch.cat(attention_mask2_list, dim=0)
        valid_pos2 = torch.cat(valid_pos2_list, dim=0)
        relation = torch.cat(relation_list, dim=0)
        
        # 데이터 준비 및 저장
        data = {
            "input_ids1": input_ids1, 
            "attention_mask1": attention_mask1, 
            "valid_pos1": valid_pos1,
            "input_ids2": input_ids2, 
            "attention_mask2": attention_mask2, 
            "valid_pos2": valid_pos2,
            "relation": relation
        }
        
        print(f"Saving encoded preference pairs into {loader_file}")
        torch.save(data, loader_file)
        
        return data
    
    def process_all_jsonl(self, dataset_file, preference_file, output_dir, max_len=512):
        """
        all.jsonl 파일과 preference_pairs.json 파일을 처리하여 텍스트 파일과 PT 파일을 생성합니다.
        """
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # utterance 추출
        docs, theme_docs, utterance_map = self.extract_utterances(dataset_file)
        
        # preference pairs 로드
        positive_pairs, negative_pairs = self.load_preference_pairs(preference_file, utterance_map)
        
        # texts.txt 저장
        self.save_texts_txt(docs, theme_docs, output_dir)
        
        # text.pt 생성
        data = self.create_dataset(output_dir, "texts.txt", "text.pt", max_len)
        theme_data = self.create_dataset(output_dir, "theme_texts.txt", "theme_text.pt", max_len)
        
        # preference.pt 생성
        preference_data = self.create_preference_dataset(output_dir, positive_pairs, negative_pairs, "preference.pt", max_len)
        
        return data, theme_data, preference_data

def main():
    parser = argparse.ArgumentParser(description="Extract utterances from all.jsonl and save as texts.txt and text.pt files")
    parser.add_argument('--dataset_file', type=str, default='./dstc12-data/AppenBanking/all.jsonl',
                       help="all.jsonl 데이터셋 파일 경로")
    parser.add_argument('--preference_file', type=str, default='./dstc12-data/AppenBanking/preference_pairs.json',
                      help="preference_pairs.json 파일 경로")
    parser.add_argument('--output_dir', type=str, default='./dstc12-data/AppenBanking/topclus',
                       help="처리된 파일을 저장할 디렉토리 경로")
    parser.add_argument('--max_len', type=int, default=512,
                       help="토큰화 시 최대 길이")
    
    args = parser.parse_args()
    
    utils = UtteranceUtils()
    data, theme_data, preference_data = utils.process_all_jsonl(
        args.dataset_file, args.preference_file, args.output_dir, args.max_len
    )
    
    print(f"데이터 크기: {data['input_ids'].shape}")
    print(f"테마 데이터 크기: {theme_data['input_ids'].shape}")
    print(f"선호도 데이터 크기: {preference_data['input_ids1'].shape}")
    print(f"처리가 완료되었습니다. 파일들이 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
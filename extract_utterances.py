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
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Extracting utterances from dataset"):
                conversation = json.loads(line)
                for turn in conversation["turns"]:
                    utterance_text = turn.get("utterance")
                    if utterance_text:
                        utterances.append(utterance_text)
        
        # 중복 제거 및 정렬
        print(utterances)
        print(f"총 추출된 unique utterance: {len(utterances)}개")
        return utterances
    
    def save_texts_txt(self, docs, output_dir):
        """
        추출된 텍스트를 texts.txt 파일로 저장합니다.
        """
        output_file = os.path.join(output_dir, "texts.txt")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving texts to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in docs:
                f.write(f"{doc}\n")
        
        print(f"텍스트가 {output_file}에 성공적으로 저장되었습니다.")
    
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
        stop_words = set(stopwords.words('english'))
        filter_idx = []
        valid_pos = ["NOUN", "VERB", "ADJ"]
        
        for i in self.inv_vocab:
            token = self.inv_vocab[i]
            if token in stop_words or token.startswith('##') \
            or token in string.punctuation or token.startswith('[') \
            or pos_tag([token], tagset='universal')[0][-1] not in valid_pos:
                filter_idx.append(i)
        
        valid_pos = attention_masks.clone()
        for i in filter_idx:
            valid_pos[input_ids == i] = 0
        
        data = {"input_ids": input_ids, "attention_masks": attention_masks, "valid_pos": valid_pos}
        torch.save(data, loader_file)
        
        return data
    
    def process_all_jsonl(self, dataset_file, output_dir, max_len=512):
        """
        all.jsonl 파일을 처리하여 texts.txt와 text.pt 파일을 생성합니다.
        """
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # utterance 추출
        docs = self.extract_utterances(dataset_file)
        
        # texts.txt 저장
        self.save_texts_txt(docs, output_dir)
        
        # text.pt 생성
        data = self.create_dataset(output_dir, "texts.txt", "text.pt", max_len)
        
        return data

def main():
    parser = argparse.ArgumentParser(description="Extract utterances from all.jsonl and save as texts.txt and text.pt files")
    parser.add_argument('--dataset_file', type=str, default='./dstc12-data/AppenBanking/all.jsonl',
                       help="all.jsonl 데이터셋 파일 경로")
    parser.add_argument('--output_dir', type=str, default='./dstc12-data/AppenBanking/topclus',
                       help="처리된 파일을 저장할 디렉토리 경로")
    parser.add_argument('--max_len', type=int, default=512,
                       help="토큰화 시 최대 길이")
    
    args = parser.parse_args()
    
    utils = UtteranceUtils()
    data = utils.process_all_jsonl(args.dataset_file, args.output_dir, args.max_len)
    print(f"데이터 크기: {data['input_ids'].shape}")
    print(f"처리가 완료되었습니다. 파일들이 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
import json
import argparse
from tqdm import tqdm

def load_preference_pairs(preference_file):
    """
    preference_pairs.json 파일에서 should_link와 cannot_link 쌍을 로드합니다.
    """
    with open(preference_file, 'r', encoding='utf-8') as f:
        preference_pairs = json.load(f)
    
    return preference_pairs["should_link"], preference_pairs["cannot_link"]

def create_utterance_id_to_text_map(dataset_file):
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

def map_utterances_to_text(preference_file, dataset_file, output_file):
    """
    preference_pairs.json 파일의 utterance ID를 all.jsonl 파일에서 찾아 텍스트로 매핑하고 결과를 저장합니다.
    """
    # preference_pairs.json 파일에서 쌍 로드
    should_link_pairs, cannot_link_pairs = load_preference_pairs(preference_file)
    
    print(f"총 should_link 쌍: {len(should_link_pairs)}")
    print(f"총 cannot_link 쌍: {len(cannot_link_pairs)}")
    
    # 데이터셋에서 utterance ID -> 텍스트 매핑 생성
    utterance_map = create_utterance_id_to_text_map(dataset_file)
    
    print(f"총 매핑된 utterance: {len(utterance_map)}")
    
    # 매핑된 결과 저장을 위한 딕셔너리 생성
    mapped_data = {
        "should_link": [],
        "cannot_link": []
    }
    
    # should_link 쌍 매핑
    for pair in tqdm(should_link_pairs, desc="Mapping should_link pairs"):
        text_1 = utterance_map.get(pair[0], "")
        text_2 = utterance_map.get(pair[1], "")
        
        if text_1 and text_2:
            mapped_data["should_link"].append({
                "text_1": text_1,
                "text_2": text_2,
                "utterance_id_1": pair[0],
                "utterance_id_2": pair[1]
            })
        else:
            print(f"Warning: Missing utterance for pair {pair}")
    
    # cannot_link 쌍 매핑
    for pair in tqdm(cannot_link_pairs, desc="Mapping cannot_link pairs"):
        text_1 = utterance_map.get(pair[0], "")
        text_2 = utterance_map.get(pair[1], "")
        
        if text_1 and text_2:
            mapped_data["cannot_link"].append({
                "text_1": text_1,
                "text_2": text_2,
                "utterance_id_1": pair[0],
                "utterance_id_2": pair[1]
            })
        else:
            print(f"Warning: Missing utterance for pair {pair}")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapped_data, f, ensure_ascii=False, indent=2)
    
    print(f"매핑 완료: {len(mapped_data['should_link'])} should_link 쌍, {len(mapped_data['cannot_link'])} cannot_link 쌍")
    print(f"결과가 {output_file}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="utterance ID를 텍스트로 매핑합니다.")
    parser.add_argument("--preference_file", type=str, default="./dstc12-data/AppenBanking/preference_pairs.json",
                        help="preference_pairs.json 파일 경로")
    parser.add_argument("--dataset_file", type=str, default="./dstc12-data/AppenBanking/all.jsonl",
                        help="all.jsonl 데이터셋 파일 경로")
    parser.add_argument("--output_file", type=str, default="./mapped_utterances.json",
                        help="매핑 결과를 저장할 파일 경로")
    
    args = parser.parse_args()
    
    map_utterances_to_text(args.preference_file, args.dataset_file, args.output_file)

if __name__ == "__main__":
    main() 
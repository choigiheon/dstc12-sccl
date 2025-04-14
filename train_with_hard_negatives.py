import os
import torch
import argparse
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataloader_with_pairs import dstc12_pairs_loader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sccl.models.Transformers import TransformerSCCL

def train_with_hard_negatives(args):
    """
    SCCL 모델을 하드 네거티브를 사용하여 학습합니다.
    
    Args:
        args: 명령줄 인수
    """
    # 장치 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 
                             'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"사용 장치: {device}")
    
    # 데이터 로더 생성
    train_loader = dstc12_pairs_loader(args)
    print(f"데이터 로더 생성 완료, 배치 수: {len(train_loader)}")
    
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # SCCL 모델 생성
    model = TransformerSCCL(args.model_name, tokenizer=tokenizer)
    model.to(device)
    
    # 최적화기 설정
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # 학습률 스케줄러 설정
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps
    )
    
    # 학습 루프
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # 일반적으로 학습 데이터는 두 개의 positive 쌍입니다.
            # 한 배치에서 두 개의 positive 쌍으로 분리하여 하드 네거티브를 적용합니다.
            batch_size = batch['input_ids_1'].size(0)
            half_batch = batch_size // 2
            
            # 데이터를 장치로 이동
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            
            # 배치를 두 개의 파트로 나누기
            input_ids_1_1 = input_ids_1[:half_batch]
            attention_mask_1_1 = attention_mask_1[:half_batch]
            input_ids_1_2 = input_ids_2[:half_batch]
            attention_mask_1_2 = attention_mask_2[:half_batch]
            
            input_ids_2_1 = input_ids_1[half_batch:]
            attention_mask_2_1 = attention_mask_1[half_batch:]
            input_ids_2_2 = input_ids_2[half_batch:]
            attention_mask_2_2 = attention_mask_2[half_batch:]
            
            # 인코딩
            with torch.no_grad():
                # 인코딩
                embeddings_1_1 = model.encoder(
                    input_ids=input_ids_1_1,
                    attention_mask=attention_mask_1_1
                ).last_hidden_state[:, 0]  # [CLS] 토큰 사용
                
                embeddings_1_2 = model.encoder(
                    input_ids=input_ids_1_2,
                    attention_mask=attention_mask_1_2
                ).last_hidden_state[:, 0]
                
                embeddings_2_1 = model.encoder(
                    input_ids=input_ids_2_1,
                    attention_mask=attention_mask_2_1
                ).last_hidden_state[:, 0]
                
                embeddings_2_2 = model.encoder(
                    input_ids=input_ids_2_2,
                    attention_mask=attention_mask_2_2
                ).last_hidden_state[:, 0]
            
            # 순전파: 하드 네거티브를 사용한 대조 학습
            optimizer.zero_grad()
            
            # contrast_logits_negative 함수는 직접 손실을 반환합니다
            loss = model.contrast_logits_negative(
                embeddings_1_1, embeddings_1_2, 
                embeddings_2_1, embeddings_2_2
            )
            
            # 역전파
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 진행 상황 출력
            if (step + 1) % args.print_freq == 0:
                print(f"에폭 {epoch+1}/{args.epochs}, 스텝 {step+1}/{len(train_loader)}, 손실: {loss.item():.4f}")
        
        # 에폭 평균 손실 계산
        avg_loss = total_loss / len(train_loader)
        print(f"에폭 {epoch+1} 완료, 평균 손실: {avg_loss:.4f}")
        
        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            
            torch.save(model.state_dict(), f"{args.save_dir}/sccl_hard_neg_model_best.pt")
            print(f"새로운 최고 모델 저장: 손실 {best_loss:.4f}")
    
    print(f"학습 완료. 최저 손실: {best_loss:.4f}")
    
    # 최종 모델 저장
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    torch.save(model.state_dict(), f"{args.save_dir}/sccl_hard_neg_model_final.pt")
    print(f"최종 모델 저장 완료: {args.save_dir}/sccl_hard_neg_model_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="하드 네거티브를 사용한 SCCL 모델 학습")
    
    # 데이터 관련 인수
    parser.add_argument("--preference_file", type=str, default="./dstc12-data/AppenBanking/preference_pairs.json",
                        help="preference_pairs.json 파일 경로")
    parser.add_argument("--dataset_file", type=str, default="./dstc12-data/AppenBanking/all.jsonl",
                        help="all.jsonl 데이터셋 파일 경로")
    parser.add_argument("--max_length", type=int, default=100, help="최대 시퀀스 길이")
    
    # 모델 관련 인수
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="사용할 모델 이름")
    parser.add_argument("--device", type=str, default="auto", 
                        help="학습에 사용할 장치 (auto, cuda, mps, cpu)")
    
    # 학습 관련 인수
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=5e-5, help="학습률")
    parser.add_argument("--epochs", type=int, default=5, help="학습 에폭 수")
    parser.add_argument("--print_freq", type=int, default=10, help="로그 출력 주기")
    
    # 저장 관련 인수
    parser.add_argument("--save_dir", type=str, default="./saved_models",
                        help="모델 저장 디렉토리")
    
    args = parser.parse_args()
    
    # 배치 크기가 짝수인지 확인
    if args.batch_size % 2 != 0:
        args.batch_size += 1
        print(f"배치 크기가 홀수입니다. 하드 네거티브 학습을 위해 {args.batch_size}로 조정합니다.")
    
    train_with_hard_negatives(args) 
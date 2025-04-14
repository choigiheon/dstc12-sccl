import os
import torch
import argparse
import numpy as np
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from dataloader_with_pairs import dstc12_pairs_loader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SCCLPairsModel(torch.nn.Module):
    def __init__(self, model_name, device):
        """
        SCCL 모델에 대한 간단한 구현
        
        Args:
            model_name (str): 사용할 사전학습 모델 이름
            device: 장치 (cuda, mps, cpu 등)
        """
        super(SCCLPairsModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.device = device
        
        # 벡터 유사도 계산을 위한 코사인 유사도
        self.cos = torch.nn.CosineSimilarity(dim=1)
        
        # 이진 분류를 위한 선형 층
        hidden_size = self.encoder.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 3, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, 2)
        )
    
    def mean_pooling(self, model_output, attention_mask):
        """
        토큰 임베딩의 평균을 계산합니다 (평균 풀링)
        
        Args:
            model_output: 인코더 모델의 출력
            attention_mask: 어텐션 마스크
            
        Returns:
            pooled_output: 평균 풀링된 임베딩
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        두 텍스트의 관계를 분류합니다
        
        Args:
            input_ids_1: 첫 번째 텍스트의 입력 ID
            attention_mask_1: 첫 번째 텍스트의 어텐션 마스크
            input_ids_2: 두 번째 텍스트의 입력 ID
            attention_mask_2: 두 번째 텍스트의 어텐션 마스크
            
        Returns:
            logits: 분류 로짓 (should_link, cannot_link에 대한 점수)
            similarity: 텍스트 쌍의 코사인 유사도
        """
        # 인코딩
        outputs_1 = self.encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
        outputs_2 = self.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)
        
        # 평균 풀링
        embeddings_1 = self.mean_pooling(outputs_1, attention_mask_1)
        embeddings_2 = self.mean_pooling(outputs_2, attention_mask_2)
        
        # 코사인 유사도 계산
        similarity = self.cos(embeddings_1, embeddings_2)
        
        # 분류를 위한 특성 결합
        # 두 임베딩과 요소별 곱을 결합
        combined = torch.cat([
            embeddings_1, 
            embeddings_2, 
            embeddings_1 * embeddings_2
        ], dim=1)
        
        # 분류
        logits = self.classifier(combined)
        
        return logits, similarity


def train(args):
    """
    SCCL 모델 학습
    
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
    
    # 모델 생성
    model = SCCLPairsModel(args.model_name, device)
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
    
    # 손실 함수 설정
    criterion = torch.nn.CrossEntropyLoss()
    
    # 학습 루프
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for step, batch in enumerate(train_loader):
            # 데이터를 장치로 이동
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['label'].to(device)
            
            # 순전파
            logits, similarity = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            
            # 손실 계산
            loss = criterion(logits, labels)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 예측 결과 수집
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # 진행 상황 출력
            if (step + 1) % args.print_freq == 0:
                print(f"에폭 {epoch+1}/{args.epochs}, 스텝 {step+1}/{len(train_loader)}, 손실: {loss.item():.4f}")
        
        # 에폭 평균 손실 및 평가 지표 계산
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"에폭 {epoch+1} 완료, 평균 손실: {avg_loss:.4f}, 정확도: {accuracy:.4f}, "
              f"정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}")
        
        # 최고 성능 모델 저장
        if accuracy > best_acc:
            best_acc = accuracy
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            
            torch.save(model.state_dict(), f"{args.save_dir}/sccl_pairs_model_best.pt")
            print(f"새로운 최고 모델 저장: 정확도 {best_acc:.4f}")
    
    print(f"학습 완료. 최고 정확도: {best_acc:.4f}")
    
    # 최종 모델 저장
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    torch.save(model.state_dict(), f"{args.save_dir}/sccl_pairs_model_final.pt")
    print(f"최종 모델 저장 완료: {args.save_dir}/sccl_pairs_model_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCCL 모델 학습")
    
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
    
    train(args) 
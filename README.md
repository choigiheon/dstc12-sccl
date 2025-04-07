# DSTC12
본 repository는 dstc11에서 best paper를 수상한 "A Two-Stage Progressive Intent Clustering for Task-Oriented Dialogue"를 구현한 repository 입니다.


SCCL Repo를 포킹한 후, SimCSE와 Two Stage Learning, Progressive K-Means 기능을 추가했습니다.

포킹한 폴더는 sccl/... 입니다.


## 사용법
모든 명령어는 root 디렉토리(dstc12)에서 실행시켜야 합니다.

잘 작동되는지 테스트용으로 inference_model은 Qwen/Qwen2.5-3B, 데이터셋은 샘플링된 데이터셋(AppenBanking/all_sampled.jsonl)을 사용하하도록 sh 파일이 미리 세팅되어 있습니다.

실제로 사용하려면 sh 파일 내에서 파라미터를 수정하여 model과 dataset을 바꾸세요.

sh 파일은 run_sccl.sh/run_theme_detection.sh/run_evaluation.sh 총 3개 있습니다.

### Clustering
다음의 명령어를 실행시키면, cluster_label_map.json 생성

제대로 클러스터링 됐는지 확인하려면, cluster_label_map.json 에 들어가기
```
$ sh run_sccl.sh
```

### Theme Generation
cluster_label_map.json의 결과를 바탕으로 라벨 생성
```
$ sh run_theme_detection.sh
```

### Evaluation
```
$ sh run_evaulation.sh
```
or
```
DSTC12 Repo와 동일한 명령어 입력
```

## 주의사항
- 모든 코드를 실행하기 전에는 dstc12 official repo와 마찬가지로 . ./set_paths.sh 를 실행시키세요.
- 가급적이면, parser.add_argument의 파라미터를 바꾸는 것이 아니라 sh 파일을 바꾸기를 권장합니다.
- sh 파일에서 따로 빼놓은 파라미터는 자주 바뀌는 파라미터입니다. 그것만 수정해도 하이퍼파라미터 탐색을 하는데 지장이 없습니다.
- 파라미터 *n_clusters와 dataset_file*는 모든 sh 파일에서 일치해야 합니다.
- get_llm()에서 bfloat16이 float16으로 바뀌었으니, 환경에 따라서 바꿔쓰세요.
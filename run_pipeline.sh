#!/usr/bin/env bash
# 처음부터 끝까지 다시 돌리는 순서 (클라우드용). 각 단계는 결과 파일이 있으면 건너뛴다.
# 필요: 환경변수 DATA_GO_KR_KEY (클라우드 환경의 비밀값으로 설정. 파일이나 커밋에 넣지 않는다)
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA_GO_KR_KEY:?DATA_GO_KR_KEY 환경변수가 필요합니다}"

pip install -q -r requirements.txt

# 1) 수집 (30일 구간 약 70개, API 호출 약 500회, 중단 후 다시 실행하면 이어 받음)
#    --end를 2026-10-19로 고정해 로컬과 같은 범위를 받는다. 최신 데이터가 필요하면 바꾼다
[ -f abandoned_dogs.csv ] || python -u collect_abandoned.py --start 20210101 --end 20261019 --out abandoned_dogs.csv

# 2) 문장 임베딩 (누수 구절 제거 후 글 기준. CPU 14스레드에서 약 35분, GPU면 몇 분)
mkdir -p emb_cache
[ -f emb_cache/e5_base_v2.npz ] || python -u embed_marks.py --csv abandoned_dogs.csv \
    --model intfloat/multilingual-e5-base --prefix "query: " --out emb_cache/e5_base_v2.npz

# 3) 모델 계열 비교 (계열별 결과를 model_search/에 저장, 다시 실행하면 끝난 계열은 건너뜀)
python -u model_search.py --csv abandoned_dogs.csv --emb emb_cache/e5_base_v2.npz --out model_search

# speech-kws-rag

ColBERT-style audio→keyword 리트리벌 POC.
키워드 토큰 뱅크 + 멀티라벨 BCE로 10–20초 음성에서 보험 도메인 키워드(100–200개)를 top-k로 추출합니다.

## Prerequisites

* Python 3.10+
* PyTorch (CUDA 사용 시, 환경에 맞는 휠 설치 권장)
  ```bash
  # GPU 예시 (CUDA 12.1)
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
* OS: Linux/macOS 권장 (Windows도 가능하나 오디오 툴 설치 이슈 주의)

## Quickstart

```bash
# 가상환경 생성 및 활성화
python -m venv .venv && source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 1) 키워드 정리(캐노니컬만)
python scripts/prepare_glossary.py --in data/raw/keywords.txt --out data/glossary.jsonl

# 2) 키워드 토큰 뱅크 + IDF 생성 (E5 frozen)
python -m src.indexing.build_text_index \
  --glossary data/glossary.jsonl \
  --out data/token_bank.json \
  --add_jamo

# 3) 학습 (ColBERT-style late interaction, BCE)
python -m src.training.train_mil \
  --train data/train.jsonl \
  --bank data/token_bank.json \
  --config configs/base.yaml

# 4) 추론 데모 (윈도우별 top-k)
python -m src.inference.stream_kws \
  --bank data/token_bank.json \
  --audio data/processed/sample.wav
```

## Data formats

`data/glossary.jsonl` (캐노니컬만)
```json
{"term_id":"피보험자","canonical":"피보험자"}
{"term_id":"손해보험","canonical":"손해보험"}
```

`data/train.jsonl` (text는 라벨 추출용/옵션, positives는 멀티라벨)
```json
{"wav_path":"data/processed/sample.wav","text":"피보험자 변경을 신청합니다.","positives":["피보험자"]}
{"wav_path":"data/processed/sample.wav","text":"손해보험 약관에 따르면...","positives":["손해보험"]}
```

## Folder structure
```
speech-kws-rag/
├─ README.md
├─ configs/
│  └─ base.yaml
├─ data/
│  ├─ raw/            # keywords.txt, 원천 오디오 등
│  └─ processed/      # 16k wav, train/val jsonl, 산출물
├─ docs/              # 설계(gemini.md), howto, reference, troubleshooting, …
├─ scripts/           # 준비/실행 스크립트
├─ src/
│  ├─ indexing/       # 토큰뱅크 + IDF 생성
│  ├─ models/         # 오디오 MLP 어댑터, 텍스트 임베더
│  ├─ training/       # BCE 학습
│  ├─ inference/      # 스트리밍 데모
│  └─ utils/          # 자모/IDF/정규화
└─ tests/             # 단위 테스트
```

## Validate & Tests

```bash
# 기본 단위 테스트
pytest -q

# 간단 검증 체크(예시)
python - << \'PY\'
import json
bank = [json.loads(l) for l in open("data/token_bank.json","r",encoding="utf-8")]
assert len(bank) > 0 and all(len(b["tokens"])>0 for b in bank)
print("OK: token bank sane")
PY
```

## Key ideas

* **Late interaction (ColBERT)**: 프레임×토큰 코사인 max, IDF로 흔한 토큰 영향↓
* **Keyword Token Bank**: 캐노니컬 서브토큰(옵션: 자모 1세트) 임베딩을 **텍스트 인코더(frozen)**로 사전 구축
* **Loss**: 멀티라벨 BCE + 하드 네거티브(보험자/피보험자 등)

## Troubleshooting (짧게)

* **토큰뱅크 비어있음** → `keywords.txt` 공백/특수문자 정규화, 템플릿에서 키워드 span 매핑 확인
* **모두 “보험”만 상위** → IDF drop 비율 올리기(`idf_drop_df_ratio`), 흔한 토큰 드롭
* **R@k 낮음** → 하드네거티브 샘플 증가, α/γ 재캘리브레이션, 오디오 MLP 어댑터 사용 확인

## Docs

* 상세 설계·규칙: `docs/gemini.md`
* 단계별 절차/레시피: `docs/howto/*`
* 스키마/포맷/지표: `docs/reference/*`
* 장애 대응: `docs/troubleshooting.md`

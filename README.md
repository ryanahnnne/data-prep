# Data Preparation Pipeline

This repository implements an end-to-end data preparation pipeline that combines image downloading, exploratory data analysis, text extraction via OCR, and stratified dataset splitting.

## Project Structure

```
data-prep/
├── pipeline.py          # 파이프라인 진입점
├── cfg/
│   └── config.yaml      # 설정 파일
├── tasks/
│   ├── __init__.py
│   ├── utils.py          # 공유 유틸리티
│   ├── downloader.py     # Task 1: 이미지 다운로드
│   ├── eda.py            # Task 2: EDA 분석
│   ├── text_extractor.py # Task 3: OCR 텍스트 추출
│   └── splitter.py       # Task 4: 데이터셋 분할
├── csv/                  # 입력 CSV 파일
└── requirements.txt
```

## Usage

```bash
# 기본 설정으로 실행
python pipeline.py

# 커스텀 config 사용
python pipeline.py --config cfg/config.yaml

# 설정 오버라이드
python pipeline.py download.enabled=true split.enabled=false
```

## Pipeline

```
CSV (이미지 URL + 라벨)
  ↓
[Task 1] Download     → images/ + download_results.csv
  ↓
[Task 2] EDA          → 시각화 차트 + eda_report.md
  ↓
[Task 3] OCR          → text_info 컬럼이 추가된 CSV
  ↓
[Task 4] Split        → train.csv, val.csv, test.csv + 검증 리포트
```

각 Task는 `config.yaml`의 `enabled` 플래그로 독립적으로 on/off 할 수 있습니다.

## Configuration

`cfg/config.yaml`에서 전체 파이프라인을 설정합니다.

| 섹션 | 설명 |
|---|---|
| `target` | 프로젝트명, 클래스명, 입력 CSV 경로 |
| `paths` | 이미지/출력 디렉토리 (변수 치환 지원) |
| `download` | 재시도 횟수, 워커 수, 타임아웃, 출력 포맷 |
| `eda` | 시각화 옵션, 이상치 탐지 기준, 리포트 설정 |
| `text_extraction` | OCR 엔진 선택 (Vision API / GLM-OCR) |
| `split` | train:val:test 비율, labeler confidence 가중치 |

### 클래스 변경 시

`target` 섹션만 수정하면 됩니다:

```yaml
target:
  class_name: "PS"
  class_name_kr: "문제_해결"
  csv_file: "csv/gt_problem_solution_v0127_0129.csv"
```

## Tasks

### Task 1: Image Downloader

AWS S3 URL에서 이미지를 다운로드합니다.

- 멀티스레드 병렬 다운로드 (max_workers 설정)
- 지수 백오프 재시도 (max_retries 설정)
- JPG/PNG/원본 포맷 변환
- RGBA → RGB 자동 변환

### Task 2: EDA (Exploratory Data Analysis)

데이터셋에 대한 종합 분석을 수행합니다.

- 라벨 분포, labeler별 통계
- 이미지 해상도/파일 크기/종횡비 분석
- 이상치 탐지 (IQR + Z-score + 도메인 기반 하드 범위)
- 시각화 차트 4종 + 마크다운 리포트

### Task 3: Text Extraction (OCR)

이미지에서 텍스트를 추출합니다.

- **Vision API**: Google Cloud Vision (confidence/높이 필터링)
- **GLM-OCR**: 오픈소스 트랜스포머 모델
- 텍스트 클리닝 (이메일, URL, 전화번호, 날짜, 특수문자 제거)

### Task 4: Dataset Splitter

Stratified train/val/test 분할을 수행합니다.

- 라벨 분포를 유지하는 계층화 분할
- Labeler confidence 가중치 기반 test set 구성
- 분할 검증 리포트 자동 생성

## Installation

```bash
pip install -r requirements.txt
```

OCR 사용 시 추가 설정:
- **Vision API**: Google Cloud 인증 설정 (`GOOGLE_APPLICATION_CREDENTIALS`)
- **GLM-OCR**: GPU 환경 권장

## Output

파이프라인 실행 후 `output_dir`에 생성되는 파일:

```
output/
├── eda_report.md                    # EDA 분석 리포트
├── split_verification_report.md     # 분할 검증 리포트
├── label_distribution.png           # 라벨 분포 차트
├── labeler_analysis.png             # Labeler 분석 차트
├── image_properties.png             # 이미지 속성 차트
├── label_vs_image_properties.png    # 라벨별 이미지 속성
├── train.csv                        # 학습 데이터
├── val.csv                          # 검증 데이터
└── test.csv                         # 테스트 데이터
```

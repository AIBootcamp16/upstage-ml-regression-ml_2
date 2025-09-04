# House Price Prediction | 아파트 실거래가 예측
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김소은](https://github.com/oriori88)             |            [김재록](https://github.com/UpstageAILab)             |            [김종화](https://github.com/UpstageAILab)             |            [최보경](https://github.com/UpstageAILab)             |            [황은혜](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
**House Price Prediction 경진대회**는 주어진 데이터를 활용하여 서울 아파트 실거래가를 예측하는 모델을 개발하는 대회입니다.  
본 레포지토리는 **2조 KAH**의 프로젝트 산출물을 관리하기 위한 저장소입니다.

### Environment
- Python 3.10
- Jupyter Notebook / VS Code (Remote-SSH)
- GPU 서버 (NVIDIA Tesla V100)

### Requirements
```bash
numpy
pandas
scikit-learn
lightgbm
xgboost
catboost
matplotlib
seaborn
plotly
geopandas
shap
````

---

## 1. Competition Info

### Overview

* Input: 아파트 특징 및 거래 정보 (국토교통부 실거래가 + 서울시 교통 데이터)
* Output: 거래금액(만원 단위)
* 목표: 정확하고 일반화된 모델 개발을 통한 아파트 시장 동향 예측

### Timeline

* 2025년 9월 1일 – 대회 시작
* 2025년 9월 10일 – 최종 제출 마감

---

## 2. Components

### Directory

```
house-price-prediction/
│
├── data/                # 데이터 저장
│   ├── raw/             # 원본 데이터 (절대 수정 X)
│   ├── processed/       # 전처리된 데이터
│   └── external/        # 외부 공공데이터 (예: 행정구역, 학군, 교통)
│
├── notebooks/           # 실험용 주피터 노트북
│   ├── 01_eda.ipynb     # (예시 파일들)
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── src/                 # 실제 코드 (함수화/모듈화)
│   ├── __init__.py
│   ├── preprocessing.py # 결측치, 이상치 처리
│   ├── features.py      # Feature Engineering
│   ├── train.py         # 모델 학습 코드
│   ├── evaluate.py      # 평가 코드
│   └── inference.py     # 추론 코드
│
├── models/              # 학습된 모델(pkl, h5 등)
│   └── random_forest.pkl
│
├── outputs/             # 결과물 저장
│   ├── figures/         # 시각화 이미지
│   ├── logs/            # 로그 파일
│   └── submissions/     # 제출용 csv
│
├── members/             # 개인별 작업 폴더
│   ├── soeun/           # (김소은)
│   │   └── notebooks/   # 개인 노트북, 시도해본 코드
│   ├── jaerok/             # (김재록)
│   │   └── notebooks/
│   ├── jonghwa/         # (김종화)
│   │   └── notebooks/
│   ├── bokyoung/           # (최보경)
│   │   └── notebooks/
│   └── eunhye/            # (황은혜)
│       └── notebooks/
│
├── tests/               # 테스트 코드
│
├── requirements.txt     # 의존성 패키지 리스트
├── README.md            # 프로젝트 개요
└── .gitignore           # 버전관리 제외 파일

```

---

## 3. Data Description

### Dataset Overview

* **국토부 실거래가 데이터**: 단지명, 전용면적, 층수, 건축년도, 거래금액 등
* **서울시 교통 데이터**: 지하철역/버스정류장 위치 및 노선 정보
* **평가 데이터**: 최종 성능 검증용 (blind test set)

### EDA

* 거래금액 분포 확인 → 로그 변환 필요성 파악
* 건축년도 → 건축연식 변수화
* 층수 분포/전용면적 분포 → 이상치 처리 기준 설정
* 지역별 평균 거래금액 차이 확인 (강남/서초/송파 vs 기타 지역)

### Data Processing

* 결측치 처리: 그룹 평균/중앙값 대체
* 이상치 처리: 거래금액 quantile 기반 클리핑
* 파생변수 생성:

  * 면적당가, 건축연식, 층구간
  * 단지/지역 평균가
  * 지하철·버스 접근성 지수

---

## 4. Modeling

### Model Description

* **Baseline**: Ridge, Lasso, RandomForest
* **Boosting 계열**: LightGBM, XGBoost, CatBoost
* **앙상블**: 스태킹/블렌딩

### Modeling Process

1. 데이터 전처리 → 파생변수 생성
2. 기본 모델 학습 → RMSE 기준 성능 측정
3. 부스팅 계열 모델 확장 → 하이퍼파라미터 튜닝
4. 교차 검증 (TimeSeriesSplit + 지역 기반 검증)
5. 스태킹/블렌딩을 통한 최종 성능 개선

---

## 5. Result

### Leader Board

* Public LB RMSE: **XXXX**
* Private LB RMSE: **XXXX**
* 최종 순위: **X위**

### Presentation

* [최종 발표자료 PDF]() (추가 예정)

---

## etc

### Meeting Log

* [Notion 회의록]() (링크 추가 예정)

### Reference

* 국토교통부 실거래가 공개시스템
* 서울열린데이터광장 (지하철/버스 정보)
* Scikit-learn, LightGBM, XGBoost, CatBoost 공식 문서

```


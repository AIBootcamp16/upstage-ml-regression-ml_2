# 🏠 House Price Prediction | 아파트 실거래가 예측
#### 서울시 아파트 실거래가 매매 데이터를 기반으로 아파트 가격을 예측하는 대회
<div>
  <a href="https://github.com/yourusername/yourrepo">
    <img src="https://img.shields.io/github/stars/yourusername/yourrepo?style=social" alt="GitHub stars"/>
  </a>
  <a href="https://github.com/yourusername/yourrepo">
    <img src="https://img.shields.io/github/forks/yourusername/yourrepo?style=social" alt="GitHub forks"/>
  </a>
</div>

## Team

| ![김소은](https://avatars.githubusercontent.com/u/11532528?v=4) | ![김재록](https://avatars.githubusercontent.com/u/47843619?v=4) | ![김종화](https://avatars.githubusercontent.com/u/221108223?v=4) | ![최보경](https://avatars.githubusercontent.com/u/110219144?v=4) | ![황은혜](https://avatars.githubusercontent.com/u/100017750?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김소은 (팀장)](https://github.com/oriori88)             |            [김재록](https://github.com/Raprok612)             |            [김종화](https://github.com/JHKIM-ItG)             |            [최보경](https://github.com/bekky1016)             |            [황은혜](https://github.com/eeunhyee)             |
| 전처리 <br> feature 추가 <br> 모델링 <br> 작업환경 구성 및 문서 정리 | 데이터 정제 <br> 문서 정리 | 전처리 <br> feature 추가 <br> 모델링 <br> 문서 정리 | feature 추가 <br> 모델링 <br> 문서 정리 | feature 추가 <br> 모델링 <br> 문서 정리 |

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
House Price Prediction 경진대회는 주어진 데이터를 활용하여 서울의 아파트 실거래가를 효과적으로 예측하는 모델을 개발하는 대회입니다. 

부동산은 의식주에서의 주로 중요한 요소 중 하나입니다. 이러한 부동산은 아파트 자체의 가치도 중요하고, 주변 요소 (강, 공원, 백화점 등)에 의해서도 영향을 받아 시간에 따라 가격이 많이 변동합니다. 개인에 입장에서는 더 싼 가격에 좋은 집을 찾고 싶고, 판매자의 입장에서는 적절한 가격에 집을 판매하기를 원합니다. 부동산 실거래가의 예측은 이러한 시세를 예측하여 적정한 가격에 구매와 판매를 도와주게 합니다. 그리고, 정부의 입장에서는 비정상적으로 시세가 이상한 부분을 체크하여 이상 신호를 파악하거나, 업거래 다운거래 등 부정한 거래를 하는 사람들을 잡아낼 수도 있습니다. 

저희는 이러한 목적 하에서 다양한 부동산 관련 의사결정을 돕고자 하는 부동산 실거래가를 예측하는 모델을 개발하는 것입니다. 특히, 가장 중요한 서울시로 한정해서 서울시의 아파트 가격을 예측하려고합니다.

<img width="698" height="285" alt="image" src="https://github.com/user-attachments/assets/bd8dce14-7af0-4634-b59e-c03c2ff93649" />


참가자들은 대회에서 제공된 데이터셋을 기반으로 모델을 학습하고, 서울시 각 지역의 아파트 매매 실거래가를 예측하는데 중점을 둡니다. 이를 위해 선형 회귀, 결정 트리, 랜덤 포레스트, 혹은 딥 러닝과 같은 다양한 regression 알고리즘을 사용할 수 있습니다.


제공되는 데이터셋은 총 네가지입니다. 첫번째는 국토교통부에서 제공하는 아파트 실거래가 데이터로 아파트의 위치, 크기, 건축 연도, 주변 시설 및 교통 편의성과 같은 다양한 특징들을 포함하고 있습니다. 두번째와 세번째 데이터는 추가 데이터로, 서울시에서 제공하는 지하철역과 버스정류장에 대한 다양한 정보들을 포함하고 있습니다. 마지막 네번째 데이터는 평가 데이터로, 최종 모델성능에 대한 검증을 위해 사용됩니다.

참가자들은 이러한 다양한 변수와 데이터를 고려하여 모델을 훈련하고, 아파트의 실거래가에 대한 예측 성능을 높이기 위한 최적의 방법을 찾아야 합니다.

<img width="873" height="359" alt="image" src="https://github.com/user-attachments/assets/0d10ec04-f21f-4325-b771-d9c39fb05dc8" />


경진대회의 목표는 정확하고 일반화된 모델을 개발하여 아파트 시장의 동향을 미리 예측하는 것입니다. 이를 통해 부동산 관련 의사 결정을 돕고, 효율적인 거래를 촉진할 수 있습니다. 또한, 참가자들은 모델의 성능을 평가하고 다양한 특성 간의 상관 관계를 심층적으로 이해함으로써 데이터 과학과 머신 러닝 분야에서의 실전 경험을 쌓을 수 있습니다.

* Input: 아파트 특징 및 거래 정보 (국토교통부 실거래가 + 서울시 교통 데이터)
* Output: 거래금액(만원 단위)
* 목표: 정확하고 일반화된 모델 개발을 통한 아파트 시장 동향 예측

### Timeline

* 2025년 9월 1일 – 대회 시작
* 2025년 9월 11일 – 최종 제출 마감

---

## 2. Components

### Directory

```
upstage-ml-regression-ml_2/
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
* **Geo Coding 위도, 경도 데이터**: 위도, 경도 결측치 보간을 위하여 주소 기준으로 위도 경도 불러옴
* **학교 데이터**: 근거리 초,중,고등학교 데이터
* **구별 평균 임금 데이터**: 연말 정산 데이터 기반 서울시 구별 평균 임금
* **금리 데이터**: 2006~2023년 금리 데이터
* **물가상승률 데이터**: 2006~2023년 물가상승률 데이터

### Data Processing

* 결측치 처리:
* 1) Geo Coding 데이터로 위도/경도 결측치 보간
  2) 불필요해보이는 변수들 과감히 삭제 (전화번호 등)
  3) 범주형 변수 : null로 보간
  4) 연속형 변수 : 단지 키 생성 후 같은 단지 내 값들 중 median값으로 보간
* 이상치 처리: 전용면적 boxplot에서 면적이 큰 데이터 삭제할 수 있지만 일부러 삭제 안 함
* 파생변수 생성:
  * 버스 feature : 최근접 정류장, 반경별 정류장 수
	 * 지하철 feature : 최근접 거리/역
	 * 학교 feature : 최근접 학교, 반경별 학교 수
  * 금리 feature : 증감률, 차이, MA3/MA12, Vol6, 상승여부
	 * 평균 임금 feature : 연도 내 평균/표준편차, 모멘텀/변동성 추세
	 * 물가상승률 feature : 전년대비 변화량, 비율 변화, 이동평균, 표준편차
	 * 강남여부, 한강여부 : 강남x전용면적, 한강x전용면적 상호작용도 함께 추가
	 * 신축 여부

---

## 4. Modeling

### Model Description

* **Baseline**: RandomForest
* **Boosting 계열**: LightGBM, XGBoost
* **앙상블**: 스태킹

### Modeling Process

1. 데이터 전처리 → 파생변수 생성
2. 기본 모델 학습 → RMSE 기준 성능 측정
3. 부스팅 계열 모델 확장 → 하이퍼파라미터 튜닝
4. 교차 검증 (TimeSeriesSplit + 지역 기반 검증)
5. 스태킹을 통한 최종 성능 개선

---

## 5. Result

### Leader Board

* Public LB RMSE: 15426.7103
* Private LB RMSE: 13682.2369
* 리더보드 중간 순위: **3위**
* 최종 순위: **5위**

중간 순위 13682.2369 3위에서 최종 순위는 15426.7103 5위로 결정됨

---

## etc
### Reference

* 국토교통부 실거래가 공개시스템
* 서울열린데이터광장 (지하철/버스 정보)
* Scikit-learn, LightGBM, XGBoost 공식 문서

```


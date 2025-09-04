# 🏠House Price Prediction 
서울시 아파트 실거래가 매매 데이터를 기반으로 아파트 가격을 예측하는 대회

---

## 1. Library Import  

---

## 2. Data Load  
- train과 test data 살펴보기  

---

## 3. Data Preprocessing  
- train/test 구분을 위한 칼럼 만듦  
- train과 test 데이터 합침  
- 컬럼 이름 정리  

### 3.1 결측치 확인 
- 딱 봐도 의미 없어보이는 컬럼은 np.nan으로 채움  
- 결측치 plot으로 그려서 확인 후 100만개 이하인 컬럼들만 의미있다고 판단, 새로운 객체에 저장  
- 연속형 변수와 범주형 변수 분리  
- 보간 진행 (범주형 : NULL 처리, 연속형 : 선형 보간) => 보간을 다른 방법으로 해봐도 좋을듯  

### 3.2 이상치 처리
- boxplot으로 분포 확인  
- IQR로 이상치 제거 (다른 방법으로 이상치 제거해보기)  

---

## 4. Feature Engineering  
- 시군구, 계약년월 등의 변수 분할  
- 강남/강북 여부 분리 (다른 방법으로도 분리해보기)  
- 신축/구축 여부 분리  
- **여기서 외부 공공 데이터를 이용해 변수 추가 가능할듯**  

---

## 5. Model Traning  

### 5.1 범주형 변수 encoding  
- 범주형 변수들 numeric하게 바꿔줌  

### 5.2 Training  
- target과 독립변수 분리  
- 학습과 테스트 데이터셋 분리 holdout 방법 적용 (그 외 다른 방법으로도 시도해보기)  
- RandomForest로 회귀 모델 적합 (하이퍼파라미터 지정해보기)  
- 변수 중요도 확인  
- 학습된 모델 저장 (피클)  

### 5.3 Feature selection  
- permutation importance로 변수 중요도를 측정  

### 5.4 valid prediction 분석  
- 검증 dataset에 target과 pred값(예측값)을 세팅  
- RMSE 계산  
- Error값이 큰 순서대로 sorting  
- 예측을 잘 하지 못한 top100, 예측을 잘 한 top100 추출  
- 레이블 인코딩 변수 복원  
- boxplot으로 분포 비교  
- 전용면적 분포 비교 (다른 변수로 분포 비교 해봐도 될듯)  

---

## 6. Inference (추론)  
- Test dataset에 대한 inference를 진행  
- 예측값 확인  

---

## 7. 결과 파일 도출  

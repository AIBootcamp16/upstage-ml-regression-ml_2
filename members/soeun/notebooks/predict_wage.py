import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def predict_by_district_linear(df: pd.DataFrame,
                               years_to_predict=(2022, 2023),
                               district_col: str | None = None,
                               save_path: str | None = None) -> pd.DataFrame:
    """
    Wide 형식(행=구, 열=연도 숫자 문자열)의 임금 데이터에 대해
    구별 선형회귀를 적합하고 years_to_predict 값을 예측해 열로 추가.

    Parameters
    ----------
    df : DataFrame
        첫 컬럼이 구 이름이거나, 인덱스가 구 이름인 데이터프레임.
    years_to_predict : tuple
        예측할 연도들.
    district_col : str | None
        구 이름이 들어있는 컬럼명(있으면 지정). 없으면 인덱스를 사용.
    save_path : str | None
        저장 경로를 주면 CSV로 저장.

    Returns
    -------
    DataFrame
        예측 열이 추가된 DataFrame.
    """
    # 1) 구 이름을 index로 정리
    if district_col and district_col in df.columns:
        df = df.set_index(district_col)
    else:
        # 첫 열이 구 이름(문자)이고 연도는 숫자열일 수 있으니 그대로 둠
        pass

    # 2) 연도 컬럼만 선별(숫자만으로 된 컬럼명)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    # 숫자형으로 강제 변환(쉼표·공백 등 제거)
    df[year_cols] = (df[year_cols]
                     .replace({',': ''}, regex=True)
                     .apply(pd.to_numeric, errors='coerce'))

    # 3) 학습에 사용할 연도(과거치)와 값
    #    (이미 존재하는 미래연도는 학습에서 제외)
    train_years = sorted([int(y) for y in year_cols if int(y) < min(years_to_predict)])
    X_train = np.array(train_years).reshape(-1, 1)  # 공통 X

    # 4) 각 구별로 회귀 후 미래연도 예측
    preds = {str(y): [] for y in years_to_predict}
    for gu, row in df[map(str, train_years)].iterrows():
        y_train = row.values.astype(float)

        # 결측이 모두인 구는 통과
        if np.all(np.isnan(y_train)):
            for y in years_to_predict:
                preds[str(y)].append(np.nan)
            continue

        # 부분 결측이 있으면 해당 연도만 사용
        mask = ~np.isnan(y_train)
        if mask.sum() < 2:
            # 점 하나만 있으면 상수 모델로 취급
            const_val = np.nanmean(y_train)
            for y in years_to_predict:
                preds[str(y)].append(const_val)
            continue

        model = LinearRegression()
        model.fit(X_train[mask], y_train[mask])

        for y in years_to_predict:
            p = float(model.predict(np.array([[y]])))
            # 음수 등 비현실값 방지(하한 0)
            p = max(0.0, p)
            preds[str(y)].append(round(p))

    # 5) 예측 열 추가
    for y in years_to_predict:
        df[str(y)] = preds[str(y)]

    # 6) 저장 옵션
    if save_path:
        df.to_csv(save_path, encoding='utf-8-sig')

    return df

# ==== 사용 예시 =======================================================
# 1) CSV로부터 읽는 경우 (첫 열이 '구' 같은 이름이거나 인덱스여도 됨)
# wages = pd.read_csv("seoul_wages.csv")  # 파일명 예시
# out = predict_by_district_linear(wages, years_to_predict=(2022, 2023),
#                                  district_col=None,  # 구 컬럼명이 있으면 '구' 등으로 지정
#                                  save_path="seoul_wages_with_2022_2023.csv")
# print(out.head())

# 2) 엑셀(xlsx)에서 읽는 경우
# wages = pd.read_excel("seoul_wages.xlsx", sheet_name=0)
# out = predict_by_district_linear(wages, (2022, 2023), district_col=None,
#                                  save_path="seoul_wages_with_2022_2023.csv")

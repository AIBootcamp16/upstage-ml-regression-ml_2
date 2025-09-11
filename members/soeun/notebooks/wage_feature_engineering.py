# income_features.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def _norm_gu(s: pd.Series) -> pd.Series:
    # "강남구" 등만 있다고 가정. 공백 제거만 수행.
    return s.astype(str).str.replace(" ", "", regex=False)

def _rolling_beta3_series(series: pd.Series) -> pd.Series:
    """
    각 그룹(구)별 값 series에 대해 길이 3 롤링의 선형회귀 기울기(beta)를 계산.
    transform로 호출되므로 반드시 입력/출력 인덱스 동일하게 반환.
    """
    idx = series.index
    values = series.to_numpy()
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    # 길이 3 창마다 회귀
    for i in range(2, n):
        y = values[i-2:i+1]
        if np.isnan(y).any():
            continue
        x = np.arange(3).reshape(-1, 1)
        coef = LinearRegression().fit(x, y).coef_[0]
        out[i] = float(coef)
    return pd.Series(out, index=idx)

def add_income_derived_features_simple(
    df: pd.DataFrame,          # concat_select (이미 '구','동','계약년','계약월' 존재)
    wage_wide: pd.DataFrame,   # ['구','2016',...,'2024'] 형태의 임금 wide표
    gu_col: str = "구",
    year_col: str = "계약년",
    year_min: int = 2016,
    year_max: int = 2024
) -> pd.DataFrame:

    out = df.copy()

    # 키 준비
    out["구_std"] = _norm_gu(out[gu_col])
    out["year"]   = out[year_col].astype(int)
    out["year_clip"] = out["year"].clip(year_min, year_max)

    # 임금 wide -> long
    w = wage_wide.copy()
    w["구_std"] = _norm_gu(w["구"] if "구" in w.columns else w.iloc[:, 0])
    year_cols = [c for c in w.columns if str(c).isdigit()]
    w_long = (w.melt(id_vars=["구_std"], value_vars=year_cols,
                     var_name="year", value_name="wage")
                .astype({"year": int}))
    w_long = w_long[(w_long["year"] >= year_min) & (w_long["year"] <= year_max)]

    # 정렬(그룹-연도 순) & 인덱스 깔끔히
    w_long = w_long.sort_values(["구_std", "year"]).reset_index(drop=True)

    # 연도 내 평균/표준편차 (transform 사용으로 인덱스 보존)
    seoul_mean = w_long.groupby("year")["wage"].transform("mean")
    seoul_std  = w_long.groupby("year")["wage"].transform("std").replace(0, np.nan)

    w_long["seoul_mean"] = seoul_mean
    w_long["seoul_std"]  = seoul_std
    w_long["rel_gap"]    = w_long["wage"] / seoul_mean - 1.0
    w_long["z_score"]    = (w_long["wage"] - seoul_mean) / seoul_std
    w_long["rank_pct"]   = w_long.groupby("year")["wage"].rank(pct=True)

    # 모멘텀/변동성/추세 — transform만 사용
    yoy = w_long.groupby("구_std")["wage"].transform(lambda s: s.pct_change())
    cagr3 = w_long.groupby("구_std")["wage"].transform(
        lambda s: (s / s.shift(3))**(1/3) - 1
    )
    accel = yoy.groupby(w_long["구_std"]).transform(lambda s: s.diff())
    vol3 = yoy.groupby(w_long["구_std"]).transform(lambda s: s.rolling(3, min_periods=3).std())
    beta3 = w_long.groupby("구_std")["wage"].transform(_rolling_beta3_series)

    w_long["yoy"] = yoy
    w_long["cagr3"] = cagr3
    w_long["accel"] = accel
    w_long["vol3"] = vol3
    w_long["trend_beta3"] = beta3

    # 로그 안정화
    w_long["wage_log1p"] = np.log1p(w_long["wage"])

    # 병합 (키: 구_std + year_clip)
    feat_cols = [
        "wage","wage_log1p","rel_gap","z_score","rank_pct",
        "yoy","cagr3","accel","vol3","trend_beta3"
    ]
    featmap = w_long[["구_std","year"] + feat_cols].rename(columns={"year": "year_clip"})

    merged = out.merge(
        featmap,
        how="left",
        on=["구_std", "year_clip"]
    )

    # 결측치 보수(중립값)
    merged["rel_gap"]  = merged["rel_gap"].fillna(0.0)
    merged["z_score"]  = merged["z_score"].fillna(0.0)
    merged["rank_pct"] = merged["rank_pct"].fillna(0.5)
    for c in ["yoy","cagr3","accel","vol3","trend_beta3"]:
        merged[c] = merged[c].fillna(0.0)
    if "wage" in merged.columns:
        # 명목값 없으면 같은 해 서울평균(있으면)으로 채움
        # seoul_mean은 featmap에 안 붙였으므로 보수 어려우면 0으로 두고 log1p 안정화
        merged["wage"] = merged["wage"].fillna(0.0)
        merged["wage_log1p"] = np.log1p(merged["wage"])

    return merged

# simple_pipeline.py

import numpy as np
import pandas as pd
import lightgbm as lgb

# 1) 파생 피처 최소셋
def add_min_area_features_inplace(X, area_col="전용면적", cut=135):
    if area_col not in X.columns:
        raise KeyError(f"'{area_col}' not in X")
    a = X[area_col].astype(float)
    X["area_over"] = (a - cut).clip(lower=0)
    # 선택적 상호작용 2개 (있을 때만)
    if "강남" in X.columns:
        X["강남*area_over"] = X["강남"] * X["area_over"]
    if "한강" in X.columns:
        X["한강*area_over"] = X["한강"] * X["area_over"]
    return X

# 2) 단일 모델로 학습(로그 타깃)
def fit_lgb_simple(X_train, y_train, X_valid=None, y_valid=None, params=None):
    params = params or {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": 42,
    }

    y_tr_log = np.log1p(y_train)
    dtr = lgb.Dataset(X_train, label=y_tr_log)

    valid_sets = []
    if X_valid is not None and y_valid is not None:
        y_va_log = np.log1p(y_valid)
        dva = lgb.Dataset(X_valid, label=y_va_log, reference=dtr)
        valid_sets = [dva]
    else:
        y_va_log = None

    model = lgb.train(
        params, dtr,
        valid_sets=valid_sets if valid_sets else None,
        num_boost_round=1000,
        early_stopping_rounds=100 if valid_sets else None,
        verbose_eval=False
    )

    # 검증셋 있으면 편향보정용 분산 추정
    sigma2 = None
    if X_valid is not None and y_valid is not None:
        pred_va_log = model.predict(X_valid, num_iteration=getattr(model, "best_iteration", None))
        sigma2 = float(np.var(np.log1p(y_valid) - pred_va_log))
    return model, sigma2

# 3) 예측(편향보정 옵션)
def predict_price(model, X_new, sigma2=None):
    pred_log = model.predict(X_new, num_iteration=getattr(model, "best_iteration", None))
    if sigma2 is None:
        return np.expm1(pred_log)
    return np.expm1(pred_log + 0.5*float(sigma2))


import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_R = 6_371_000.0  # meters

# ---------- 공통 유틸 ----------
def _to_rad(lat, lon):
    return np.deg2rad(np.c_[lat, lon])

def _build_tree(lat_deg, lon_deg):
    return BallTree(_to_rad(lat_deg, lon_deg), metric="haversine")

def _dist_to_nearest(tree, q_rad):
    d, idx = tree.query(q_rad, k=1)           # d: radians
    return d[:, 0] * EARTH_R, idx[:, 0]       # meters, indices

def _count_within(tree, q_rad, meters):
    r = meters / EARTH_R
    ind = tree.query_radius(q_rad, r=r)
    return np.array([len(ix) for ix in ind], dtype=int)

def _apply_with_mask(df, lat_col, lon_col, fn):
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
    mask = (~np.isnan(lat)) & (~np.isnan(lon))
    q_rad = _to_rad(lat[mask], lon[mask])
    return fn(q_rad, mask)

# ---------- (지하철 전용) ----------
def build_subway_features(
    df,
    sub_df,
    lat_col="lat", lon_col="lon",
    sub_lat_col="위도", sub_lon_col="경도",
    count_radii=(500, 1000),
    station_name_col="역사명",
    line_col="호선",
    transfer_min_lines=2,
):
    s_lat = pd.to_numeric(sub_df[sub_lat_col], errors="coerce").to_numpy()
    s_lon = pd.to_numeric(sub_df[sub_lon_col], errors="coerce").to_numpy()
    valid = (~np.isnan(s_lat)) & (~np.isnan(s_lon))
    base = df.reset_index(drop=True).copy()

    feats = pd.DataFrame(index=df.index)
    if valid.sum() == 0:
        feats["dist_subway_min_m"] = np.nan
        for r in count_radii:
            feats[f"sub_ct_{r}m"] = 0
        if station_name_col and line_col and (station_name_col in sub_df.columns) and (line_col in sub_df.columns):
            feats["nearest_subway_line_count"] = np.nan
            feats["dist_transfer_subway_min_m"] = np.nan
        return pd.concat([df.reset_index(drop=True), feats], axis=1)

    tree = _build_tree(s_lat[valid], s_lon[valid])

    have_line = (station_name_col in sub_df.columns) and (line_col in sub_df.columns)
    xfer_tree = None
    sub_with_cnt = None
    if have_line:
        tmp = sub_df.loc[valid, [station_name_col, line_col]].copy()
        tmp[line_col] = tmp[line_col].astype(str)
        line_count_map = tmp.groupby(station_name_col)[line_col].nunique()
        sub_with_cnt = tmp.copy()
        sub_with_cnt["line_count"] = sub_with_cnt[station_name_col].map(line_count_map).fillna(1).astype(int)
        xfer_mask = sub_with_cnt["line_count"].to_numpy() >= transfer_min_lines
        if xfer_mask.any():
            xfer_tree = _build_tree(s_lat[valid][xfer_mask], s_lon[valid][xfer_mask])

    def _fn(q_rad, mask):
        res = pd.DataFrame(index=df.index)

        dmin, idx = _dist_to_nearest(tree, q_rad)
        dfull = np.full(len(df), np.nan); dfull[mask] = dmin
        res["dist_subway_min_m"] = dfull

        for r in count_radii:
            cnt = _count_within(tree, q_rad, r)
            cfull = np.zeros(len(df), dtype=int); cfull[mask] = cnt
            res[f"sub_ct_{r}m"] = cfull

        if have_line:
            lc = sub_with_cnt["line_count"].to_numpy()[idx]
            lcfull = np.full(len(df), np.nan); lcfull[mask] = lc
            res["nearest_subway_line_count"] = lcfull

            if xfer_tree is not None:
                dx, _ = _dist_to_nearest(xfer_tree, q_rad)
                xfull = np.full(len(df), np.nan); xfull[mask] = dx
                res["dist_transfer_subway_min_m"] = xfull
            else:
                res["dist_transfer_subway_min_m"] = np.nan
        return res

    # feats = _apply_with_mask(df, lat_col, lon_col, _fn)
    feats = _apply_with_mask(base, lat_col, lon_col, _fn)
    feats = feats.reset_index(drop=True)

    # 원본 df와 병합해서 반환
    return pd.concat([base, feats], axis=1)

# ---------- (버스 전용) ----------
def build_bus_features(
    df,
    bus_df,
    lat_col="lat", lon_col="lon",          # df의 위도/경도 컬럼명
    bus_lat_col="위도", bus_lon_col="경도", # bus_df의 위도/경도 컬럼명
    count_radii=(200, 400, 800),           # 반경 내 정류장 개수(m)
    central_type_col="정류소_타입",         # 중앙차로 판단 컬럼 (없으면 None)
    central_radius_m=400,                  # 중앙차로 비율 계산 반경(m)
):
    # 유효 버스 좌표만 사용
    b_lat = pd.to_numeric(bus_df[bus_lat_col], errors="coerce").to_numpy()
    b_lon = pd.to_numeric(bus_df[bus_lon_col], errors="coerce").to_numpy()
    valid = (~np.isnan(b_lat)) & (~np.isnan(b_lon))
    base = df.reset_index(drop=True).copy()

    feats = pd.DataFrame(index=df.index)
    if valid.sum() == 0:
        # 버스 포인트 없으면 안전한 기본값으로 반환
        feats["dist_bus_min_m"] = np.nan
        for r in count_radii:
            feats[f"bus_ct_{r}m"] = 0
        if central_type_col is not None:
            feats[f"bus_central_ratio_{central_radius_m}m"] = np.nan
        return pd.concat([df.reset_index(drop=True), feats], axis=1)

    tree = _build_tree(b_lat[valid], b_lon[valid])

    # 중앙차로 여부 벡터 (옵션)
    is_central = None
    if central_type_col and (central_type_col in bus_df.columns):
        is_central = bus_df.loc[valid, central_type_col].astype(str).str.contains("중앙", na=False).to_numpy()

    def _fn(q_rad, mask):
        res = pd.DataFrame(index=df.index)

        # 1) 최근접 정류장 거리
        dmin, _ = _dist_to_nearest(tree, q_rad)
        dfull = np.full(len(df), np.nan); dfull[mask] = dmin
        res["dist_bus_min_m"] = dfull

        # 2) 반경 내 정류장 개수
        for r in count_radii:
            cnt = _count_within(tree, q_rad, r)
            cfull = np.zeros(len(df), dtype=int); cfull[mask] = cnt
            res[f"bus_ct_{r}m"] = cfull

        # 3) 중앙차로 비율 (있을 때만)
        if is_central is not None:
            rr = central_radius_m / EARTH_R
            inds = tree.query_radius(q_rad, r=rr)
            ratio_sub = np.array([(is_central[ix].mean() if len(ix) else 0.0) for ix in inds], dtype=float)
            rfull = np.full(len(df), np.nan); rfull[mask] = ratio_sub
            res[f"bus_central_ratio_{central_radius_m}m"] = rfull

        return res

    # feats = _apply_with_mask(df, lat_col, lon_col, _fn)
    feats = _apply_with_mask(base, lat_col, lon_col, _fn)
    feats = feats.reset_index(drop=True)

    # 원본 df와 병합해서 반환
    return pd.concat([base, feats], axis=1)
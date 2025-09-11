import numpy as np, pandas as pd
from sklearn.neighbors import BallTree

EARTH_R = 6371000.0  # meters

def _to_rad(lat, lon):
    return np.radians(np.c_[lat.astype(float), lon.astype(float)])

def _build_tree(subway_df, lat_col="위도", lon_col="경도"):
    pts = _to_rad(subway_df[lat_col], subway_df[lon_col])
    tree = BallTree(pts, metric="haversine")
    return tree, pts

def add_subway_features(df, subway_df,
                        lat_col="위도", lon_col="경도",
                        sub_lat_col="위도", sub_lon_col="경도",
                        sub_name_col="역명",
                        radii_m=(300, 500, 1000),
                        walk_m_per_min=80,
                        out_csv=None):
    # 준비
    tree, sub_pts = _build_tree(subway_df, sub_lat_col, sub_lon_col)
    qmask = df[lat_col].notna() & df[lon_col].notna()
    q = _to_rad(df.loc[qmask, lat_col], df.loc[qmask, lon_col])

    # 최근접 거리/역
    d, idx = tree.query(q, k=1)
    dist_m = (d[:, 0] * EARTH_R)
    nearest_idx = idx[:, 0]

    out = pd.DataFrame(index=df.index)
    out.loc[qmask, "subway_dist_m"] = dist_m
    out["subway_walk_min"] = out["subway_dist_m"] / walk_m_per_min

    if sub_name_col in subway_df.columns:
        names = subway_df.iloc[nearest_idx][sub_name_col].values
        out.loc[qmask, "subway_nearest_name"] = names

    for r in radii_m:
        rr = r / EARTH_R
        ind = tree.query_radius(q, r=rr)
        counts = np.array([len(ix) for ix in ind])
        out.loc[qmask, f"subway_cnt_{r}m"] = counts

    # 합치기
    out = out.reindex(df.index)
    result = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    # 저장 옵션
    if out_csv:
        result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return result

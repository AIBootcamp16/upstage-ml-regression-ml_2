import numpy as np, pandas as pd, time
from sklearn.neighbors import BallTree

EARTH_R = 6371000.0  # meters

def _to_rad(lat, lon):
    return np.radians(np.c_[lat.astype(float), lon.astype(float)])

def _build_tree_park(park_df, lat_col="위도", lon_col="경도"):
    t0 = time.perf_counter()
    pts = _to_rad(park_df[lat_col], park_df[lon_col])
    tree = BallTree(pts, metric="haversine")
    print(f"[park] tree built in {time.perf_counter()-t0:.2f}s  (n={len(pts)})")
    return tree

def add_park_features(df, park_df,
                     lat_col="위도", lon_col="경도",
                     park_lat_col="위도", park_lon_col="경도",
                     park_name_col="한강공원이름",
                     radii_m=(500, 1000, 1500),
                     walk_m_per_min=80,
                     out_csv=None,
                     verbose=True):
    t_start = time.perf_counter()
    tree = _build_tree_park(park_df, park_lat_col, park_lon_col)

    qmask = df[lat_col].notna() & df[lon_col].notna()
    q = _to_rad(df.loc[qmask, lat_col], df.loc[qmask, lon_col])
    if verbose: print(f"[park] query points: {q.shape[0]}/{len(df)}")

    # 최근접 학교
    t0 = time.perf_counter()
    d, idx = tree.query(q, k=1)  # radians
    if verbose: print(f"[park] nearest query: {time.perf_counter()-t0:.2f}s")

    dist_m = d[:, 0] * EARTH_R
    nearest_idx = idx[:, 0]

    out = pd.DataFrame(index=df.index)
    out.loc[qmask, "park_dist_m"] = dist_m
    out["park_walk_min"] = out["park_dist_m"] / walk_m_per_min

    if park_name_col in park_df.columns:
        out.loc[qmask, "park_nearest_name"] = park_df.iloc[nearest_idx][park_name_col].values

    # 반경별 학교 수
    for r in radii_m:
        t0 = time.perf_counter()
        rr = r / EARTH_R
        ind = tree.query_radius(q, r=rr)
        out.loc[qmask, f"park_cnt_{r}m"] = np.fromiter((len(ix) for ix in ind), dtype=int, count=len(ind))
        if verbose: print(f"[park] radius {r}m: {time.perf_counter()-t0:.2f}s")

    result = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    if out_csv:
        t0 = time.perf_counter()
        result.to_csv(out_csv, index=False, encoding="utf-8-sig")
        if verbose: print(f"[park] saved '{out_csv}' in {time.perf_counter()-t0:.2f}s")

    if verbose: print(f"[park] total {time.perf_counter()-t_start:.2f}s")
    return result

import os, time, requests
import numpy as np
import pandas as pd
from tqdm import tqdm

def _norm_bunji(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if s == "" or s == "0": return None
    return s

def _geocode_one(addr, session, api_key, timeout=5):
    url = "https://api.vworld.kr/req/address"
    params = {
        "service":"address", "request":"getCoord", "version":"2.0",
        "crs":"epsg:4326", "type":"parcel", "address":addr, "key":api_key
    }
    try:
        r = session.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return (np.nan, np.nan)
        data = r.json()
        if data.get("response", {}).get("status") != "OK":
            return (np.nan, np.nan)
        p = data["response"]["result"]["point"]
        return float(p["y"]), float(p["x"])  # (lat, lon)
    except Exception:
        return (np.nan, np.nan)

def geocode_concat_and_save_fast(
    df: pd.DataFrame,
    sigungu_col="시군구",
    bunji_col="번지",
    output_csv="concat_with_geo.csv",
    sleep_sec=0.12,       # QPS 보호(약 8~9 QPS). 주소가 적으면 더 줄여도 됨(0.08~0.10)
    timeout=5
):
    """
    df (예: concat_select2)를 입력받아:
      1) 시군구+번지 문자열 만들기
      2) 고유 주소만 지오코딩
      3) 결과를 원본 df에 매핑하여 geo_lat/geo_lon 추가
      4) 새 CSV로 저장
    """
    api_key = "F95CB06B-91FF-3BE2-BDA2-F4C02EB9EFEC" #os.getenv("VWORLD_KEY")
    if not api_key:
        raise RuntimeError("VWORLD_KEY 환경변수가 설정되어 있지 않습니다.")

    # 주소 문자열 구성
    sig = df[sigungu_col].astype(str).str.strip()
    bun = df[bunji_col].apply(_norm_bunji)
    addr = pd.Series(np.where(bun.isna(), None, (sig + " " + bun.astype(str))), index=df.index)
    # 고유 주소만 추출
    uniq = sorted({a for a in addr.dropna().unique()})

    results = {}
    sess = requests.Session()
    for a in tqdm(uniq, desc=f"Geocoding unique addrs ({len(uniq):,})"):
        lat, lon = _geocode_one(a, sess, api_key, timeout=timeout)
        results[a] = (lat, lon)
        time.sleep(sleep_sec)

    # 원본에 매핑
    lat_series = addr.map(lambda a: results.get(a, (np.nan, np.nan))[0] if a else np.nan)
    lon_series = addr.map(lambda a: results.get(a, (np.nan, np.nan))[1] if a else np.nan)

    out = df.copy()
    out["위도"] = lat_series
    out["경도"] = lon_series

    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved → {output_csv}")
    return out

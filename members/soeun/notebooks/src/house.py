# !pip install eli5==0.13.0

# # 한글 폰트 사용을 위한 라이브러리입니다.
# !apt-get install -y fonts-nanum

# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')
import seaborn as sns

# utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import warnings;warnings.filterwarnings('ignore')

# Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import eli5
from eli5.sklearn import PermutationImportance\

import build_transit_features as btf
import nearest_subway  as ns
import geocode_concat as gc

train_path = "~/apt-price/data/train.csv"
test_path = "~/apt-price/data/test.csv"
bus_path = "~/apt-price/data/bus_feature.csv"
subway_path = "~/apt-price/data/subway_feature.csv"

dt = pd.read_csv(train_path)
dt_test = pd.read_csv(test_path)
bus_df = pd.read_csv(bus_path)
subway_df = pd.read_csv(subway_path)

# 데이터 shape 확인
print('Train data shape : ', dt.shape, 'Test data shape : ', dt_test.shape)
print('Bus data shape : ', bus_df.shape, 'Subway data shape : ', subway_df.shape)

# train과 test 구분을 위한 컬럼을 만들어주고 하나의 데이터로 합쳐줌
dt['is_test'] = 0
dt_test['is_test'] = 1
concat = pd.concat([dt, dt_test]) 
concat['is_test'].value_counts()

# 어려운 컬럼 이름 바꿔줌
concat = concat.rename(columns={'전용면적(㎡)':'전용면적', '좌표X':'경도', '좌표Y':'위도'})

# 집값과 전혀 상관 없을 것 같은 아래 변수들은 과감히 지워줌
concat.drop(columns=["해제사유발생일",
                    "등기신청일자",
                    "거래유형",
                    "중개사소재지",
                    "k-전화번호",
                    "k-팩스번호",
                    "단지소개기존clob",
                    "k-홈페이지",
                    "k-등록일자",
                    "k-수정일자",
                    "고용보험관리번호",
                    "단지승인일",
                    "관리비 업로드",
                    "단지신청일",
                    'k-사용검사일-사용승인일',
                    '사용허가여부',
                    '기타/의무/임대/임의=1/2/3/4',
                    '청소비관리형태',
                    '세대전기계약방법',
                    'k-관리방식',
                    'k-단지분류(아파트,주상복합등등)',
                    'k-시행사',
                    'k-복도유형',
                    'k-세대타입(분양형태)'], inplace=True)

# 본번, 부번 str 형태로 수정
concat_select = concat
concat['본번'] = concat['본번'].astype('str')
concat['부번'] = concat['부번'].astype('str')

# 지하철 역세권인가
# concat_select2 = btf.build_subway_features(concat_select, subway_df, lat_col="위도", lon_col="경도")


# concat_select2 = ns.geocode_df_with_cache(
#     concat_select,
#     sido_col="시도", sigungu_col="시군구", beopdong_col="법정동",
#     bunji_col="번지", bonbun_col="본번", bubun_col="부번",
#     out_lat_col="geo_lat", out_lon_col="geo_lon",
#     cache_path="../cache/geocode_cache.json",              # 캐시 파일
#     checkpoint_path="../cache/geocode_checkpoint.parquet", # 중간/최종 저장
#     log_every=10000, checkpoint_every=1000
# )      

concat_select2 = gc.geocode_concat_and_save(
    df=concat_select,
    sigungu_col="시군구",   # concat_select2 안 컬럼명 맞게 바꿔야 함
    bunji_col="번지",
    output_csv="concat_select2_with_geo.csv"
)



# print(concat_select2.head())
print(concat_select2.head())


# 버스 정류장 가까운가
# concat_select2 = btf.build_bus_features(concat_select, subway_df, lat_col="위도", lon_col="경도", sub_lat_col="위도", sub_lon_col="경도")



# # 버스: X=경도, Y=위도 → 숫자화 후 NaN 제거
# bus_df["경도"] = pd.to_numeric(bus_df["X좌표"], errors="coerce")
# bus_df["위도"] = pd.to_numeric(bus_df["Y좌표"], errors="coerce")
# bus_df = bus_df.dropna(subset=["위도", "경도"])

# # 지하철: 이미 위도/경도라 가정
# subway_df["위도"] = pd.to_numeric(subway_df["위도"], errors="coerce")
# subway_df["경도"] = pd.to_numeric(subway_df["경도"], errors="coerce")
# subway_df = subway_df.dropna(subset=["위도", "경도"])

# # 쿼리 데이터(거래/아파트)
# concat_select["위도"] = pd.to_numeric(concat_select["위도"], errors="coerce")
# concat_select["경도"] = pd.to_numeric(concat_select["경도"], errors="coerce")





# # 연속형 변수와 범주형 변수 분리
# continuous_columns = []
# categorical_columns = []

# for column in concat_select.columns:
#     if pd.api.types.is_numeric_dtype(concat_select[column]):
#         continuous_columns.append(column)
#     else:
#         categorical_columns.append(column)

# print("연속형 변수:", continuous_columns)
# print("범주형 변수:", categorical_columns)

# # 결측치 처리
# # 범주형 변수는 null을 채워서 보간해줌
# concat_select[categorical_columns] = concat_select[categorical_columns].fillna('NULL')

# # 연속형 변수
# # 단지 키(시군구+도로명+아파트명+건축년도)를 만들고 같은 단지 내 관측들에서 median 값으로 채우기
# concat_select["complex_key"] = (
#     concat_select["시군구"].astype(str).str.strip() + "|" +
#     concat_select["도로명"].astype(str).str.strip() + "|" +
#     concat_select["아파트명"].astype(str).str.strip() + "|" +
#     concat_select["건축년도"].astype(str).str.strip()
# )

# concat_select[continuous_columns].isna().mean().sort_values(ascending=False)


# def _fill_by_group_median(df, col, group_cols):
#     # group_cols 순서대로 내려가며 채우기
#     s = df[col]
#     for gcols in group_cols:
#         med = df.groupby(gcols)[col].transform("median")
#         s = s.fillna(med)
#     return s

# group_levels = [
#     ["complex_key"],                # 같은 단지
#     ["시군구","아파트명"],             # 같은 시군구의 같은 아파트명
#     ["시군구"],                      # 같은 시군구
# ]

# for col in continuous_columns:
#     if col not in concat_select.columns: 
#         continue
#     was_na = concat_select[col].isna()
#     concat_select[col] = _fill_by_group_median(concat_select, col, group_levels)
#     # 전역 중앙값으로 최종 백업
#     concat_select[col] = concat_select[col].fillna(concat_select[col].median())
    

# print(concat_select.info())






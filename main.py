# main.py — NEXT SCENE: 최근 개봉작의 관객 예측 리포트
from datetime import datetime
from io import BytesIO
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
BASE = ["first_scrn", "first_show", "peak"]
EXTRA = {"배우 수": "actors_n", "주연 수": "lead_n", "상영시간": "showTm",
         "청소년관람불가": "adult", "대형 배급사": "bigco"}
LABELS = dict(zip(BASE, ["첫 관측일 스크린 수", "첫 관측일 상영 횟수", "성수기 개봉"]))
LABELS.update({v: k for k, v in EXTRA.items()})
NEEDED = ["actors_n", "lead_n", "showTm", "watchGrade", "company"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(url):
    with urlopen(url, timeout=25) as response:
        return response.read()


def prepare(movies, people, daily, today):
    # '최근'은 달력과 실제 데이터 마지막 날짜를 모두 확인한다.
    daily = daily.copy()
    daily["관측일"] = pd.to_datetime(daily["날짜"].astype(str), format="%Y%m%d", errors="coerce")
    daily = daily[daily["관측일"].notna() & (daily["관측일"] <= today)]
    if daily.empty:
        raise ValueError("오늘까지의 관측 기록이 없습니다.")
    asof = daily["관측일"].max()
    if movies["movieCd"].duplicated().any() or people["movieCd"].duplicated().any():
        raise ValueError("영화 코드가 중복되어 있습니다. 데이터 갱신 후 다시 확인하세요.")
    df = movies.merge(people[["movieCd", "director"] + NEEDED], on="movieCd", how="inner", validate="one_to_one")
    merged_n = len(df)
    for col in BASE + ["actors_n", "lead_n", "showTm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["watchGrade", "company"]:
        df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)
    missing = df[BASE + NEEDED].isna().sum()
    df["개봉일"] = pd.to_datetime(df["openDt"].astype(str), format="%Y%m%d", errors="coerce")
    df["첫 관측일"] = pd.to_datetime(df["first_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=BASE + NEEDED + ["개봉일", "첫 관측일"])
    df = df[(df["first_scrn"] > 0) & (df["first_show"] > 0) & (df["showTm"] > 0)
            & (df["첫 관측일"] <= asof) & (df["개봉일"] <= asof)].copy()
    df["adult"] = (df["watchGrade"] == "청소년관람불가").astype(int)
    # 관측일이 확인되는 일별 표의 마지막 누적관객을 사용한다.
    latest = daily.sort_values("관측일").drop_duplicates("영화코드", keep="last")
    latest = latest[["영화코드", "관측일", "누적관객"]].rename(columns={
        "영화코드": "movieCd", "관측일": "마지막 관측일", "누적관객": "관측 누적관객"})
    latest["관측 누적관객"] = pd.to_numeric(latest["관측 누적관객"], errors="coerce")
    df = df.merge(latest, on="movieCd", how="inner", validate="one_to_one")
    df = df.dropna(subset=["관측 누적관객"])
    df = df[df["관측 누적관객"] > 0].copy()
    df["total_audi"] = df["관측 누적관객"]
    # 최근 30일 개봉작은 예측 전용. 최소 60일 지난 작품만 훈련·테스트에 쓴다.
    cutoff = asof - pd.Timedelta(days=60)
    history = df[df["개봉일"] <= cutoff].sort_values("movieCd").reset_index(drop=True)
    recent = df[df["개봉일"] > asof - pd.Timedelta(days=30)].copy()
    mask = history.index % 10 < 3
    train, test = history[~mask].copy(), history[mask].copy()
    if len(train) < 20 or len(test) < 5:
        raise ValueError("훈련·테스트에 쓸 과거 영화가 부족합니다.")
    # 배급사 상위 5곳도 훈련 데이터로만 정한다.
    big = train["company"].value_counts().head(5).index
    for part in [train, test, recent]:
        part["bigco"] = part["company"].isin(big).astype(int)
    recent = recent.sort_values(["개봉일", "first_scrn"], ascending=[False, False])
    return train, test, recent, asof, missing, merged_n


def evaluate(model, data, cols):
    pred = model.predict(data[cols])
    r2 = r2_score(data["total_audi"], pred) if data["total_audi"].nunique() > 1 else float("nan")
    return {"영화 수": len(data), "R²": r2,
            "MAE(명)": mean_absolute_error(data["total_audi"], pred)}


def score(train, test, cols):
    model = LinearRegression().fit(train[cols], train["total_audi"])
    return model, evaluate(model, train, cols), evaluate(model, test, cols)


st.set_page_config(page_title="NEXT SCENE · 관객 예측", page_icon="🎬", layout="wide")
st.title("NEXT SCENE")
st.caption("속성을 더 모으면 예측이 달라질까 · 최근 개봉작 관객 예측 리포트")
try:
    movies = pd.read_csv(BytesIO(fetch(ROOT + "kobis_movies.csv")), encoding="utf-8-sig", dtype={"movieCd": str})
    people = pd.read_csv(BytesIO(fetch(ROOT + "kobis_people.csv")), encoding="utf-8-sig", dtype={"movieCd": str})
    daily = pd.read_csv(BytesIO(fetch(ROOT + "kobis_daily.csv")), encoding="utf-8-sig", dtype={"영화코드": str, "날짜": str})
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Seoul")).date())
    train, test, recent, asof, missing, merged_n = prepare(movies, people, daily, today)
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다. 잠시 뒤 다시 실행하세요. ({type(error).__name__}: {error})")
    st.stop()

st.info(f"데이터 기준일 {asof:%Y.%m.%d} · 최근 30일 개봉작 {len(recent)}편 · 훈련 {len(train)}편 / 테스트 {len(test)}편")
if (today - asof).days > 2:
    st.warning(f"데이터가 {(today-asof).days}일 전 기록입니다. 현재 상영작 전체와 다를 수 있습니다.")
st.caption("최근 개봉작은 학습·평가에 넣지 않습니다. 과거 영화의 마지막 관측 누적관객을 예측 목표로 삼으며, 최종 관객 수가 확정되었다는 뜻은 아닙니다.")
with st.expander("합친 데이터 확인"):
    st.write(f"영화 표 {len(movies)}편 · 인물 표 {len(people)}편 · 합친 표 {merged_n}편")
    st.write("열별 결측 편수", missing)
    st.dataframe(train[["movieNm", "director", "first_scrn", "actors_n", "showTm", "total_audi"]].head(10))
    st.plotly_chart(px.scatter(train, x="showTm", y="total_audi", hover_name="movieNm", log_y=True,
                              labels={"showTm": "상영시간(분)", "total_audi": "관측 누적관객(명)"}), key="duration")
    grades = train.groupby("watchGrade")["total_audi"].agg(중앙값="median", 편수="size").reset_index()
    st.plotly_chart(px.bar(grades, x="watchGrade", y="중앙값", text="편수"), key="grades")

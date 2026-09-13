# main.py — 영화 흥행 예측기 (심화 탐구 프로젝트)
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

MOVIES = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
PEOPLE = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_people.csv"

BASE = ["first_scrn", "first_show", "peak"]
EXTRA = {
    "배우 수": "actors_n",
    "주연 수": "lead_n",
    "상영시간": "showTm",
    "청소년관람불가": "adult",
    "대형 배급사": "bigco",
}
NEEDED = ["actors_n", "lead_n", "showTm", "watchGrade", "company"]   # 하나라도 결측이면 뺀다

st.title("영화 흥행 예측기")


@st.cache_data
def load():
    movies = pd.read_csv(MOVIES, encoding="utf-8-sig", dtype={"movieCd": str})
    people = pd.read_csv(PEOPLE, encoding="utf-8-sig", dtype={"movieCd": str})
    people = people[["movieCd", "director"] + NEEDED]
    df = movies.merge(people, on="movieCd", how="inner")
    결측 = df[NEEDED].isna().sum()                  # 열마다 비어 있는 영화가 몇 편인가
    합친편수 = len(df)
    df = df.dropna(subset=NEEDED)                   # 다섯 열 가운데 하나라도 결측이면 뺀다
    df = df[(df["total_audi"] > 0) & (df["first_scrn"] > 0) & (df["showTm"] > 0)]
    df = df.sort_values("movieCd").reset_index(drop=True)
    df["adult"] = (df["watchGrade"] == "청소년관람불가").astype(int)
    big = df["company"].value_counts().head(5).index
    df["bigco"] = df["company"].isin(big).astype(int)
    return df, 결측, 합친편수


def split(df):
    test = (df.index % 10) < 3          # 열 편마다 앞의 세 편이 테스트용
    return df[~test], df[test]


def score(train, test, cols):
    model = LinearRegression().fit(train[cols], train["total_audi"])
    pred = model.predict(test[cols])
    return r2_score(test["total_audi"], pred), mean_absolute_error(test["total_audi"], pred), model

df, 결측, 합친편수 = load()
train, test = split(df)
st.write(f"합친 표 {합친편수}편 · 결측 등을 뺀 뒤 {len(df)}편 · 훈련용 {len(train)}편 · 테스트용 {len(test)}편")
st.write("열별 결측 편수")
st.write(결측)
st.dataframe(df[["movieNm", "director", "first_scrn", "actors_n", "showTm", "total_audi"]].head(10))

st.subheader("넣기 전에 먼저 그려 본다")
st.plotly_chart(px.scatter(df, x="showTm", y="total_audi", log_y=True, hover_name="movieNm",
                           labels={"showTm": "상영시간(분)", "total_audi": "총 관객 수"}))
등급 = df.groupby("watchGrade")["total_audi"].agg(중앙값="median", 편수="size").reset_index()
st.plotly_chart(px.bar(등급, x="watchGrade", y="중앙값", text="편수",
                       labels={"watchGrade": "관람등급", "중앙값": "총 관객 수 중앙값"}))

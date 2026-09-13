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

st.header("고른 속성으로 다시 학습한다")
picked = [col for name, col in EXTRA.items() if st.checkbox(name, value=False)]
cols = BASE + picked
r2, mae, model = score(train, test, cols)
c1, c2 = st.columns(2)
c1.metric("테스트용 데이터의 R²", f"{r2:.3f}")
c2.metric("테스트용 평균절대오차", f"{mae:,.0f}명")
weight = pd.DataFrame({"속성": cols, "가중치": model.coef_})
st.plotly_chart(px.bar(weight, x="가중치", y="속성", orientation="h"))
st.caption("단위가 다르면 가중치의 크기를 그대로 비교할 수 없습니다.")
look = test.assign(예측=model.predict(test[cols]))
st.plotly_chart(px.scatter(look, x="total_audi", y="예측", hover_name="movieNm", log_x=True, log_y=True))

st.header("내가 고른 영화를 예측한다")
name = st.text_input("영화 이름", "", placeholder="영화 이름을 적으세요")
값 = {}
값["first_scrn"] = st.number_input("첫 관측일 스크린수", value=800)
값["first_show"] = st.number_input("첫 관측일 상영횟수", value=3000)
값["peak"] = 1 if st.checkbox("성수기 개봉", value=True) else 0
for col in picked:
    값[col] = st.number_input(col, value=float(df[col].median()))
mine = pd.DataFrame([[값[c] for c in cols]], columns=cols)
pred = float(model.predict(mine)[0])
st.metric(f"{name}의 예측 총 관객 수", f"{pred:,.0f}명")
near = df.assign(차이=(df["total_audi"] - pred).abs()).nsmallest(5, "차이")
st.dataframe(near[["movieNm", "total_audi"]])

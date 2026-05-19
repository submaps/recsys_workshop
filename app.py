import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Настройка страницы Streamlit
st.set_page_config(page_title="IMDb Movie Recommender", layout="wide")


# 1. Загрузка данных и кэширование, чтобы интерфейс не зависал
@st.cache_data
def load_data():
    data_path = './imdb_top_1000_img.csv'
    df = pd.read_csv(data_path)

    df.rename(columns={
                'Series_Title': 'Title',
                'Released_Year': 'year',
                'IMDB_Rating': 'Rate',
                'Overview': 'Info'
               }, inplace=True)
    df['Description'] = df['Info'].tolist()
    df = df.drop_duplicates('Title', keep='first').reset_index(drop=True)
    df["content_soup"] = df['Title'] + ' ' + df['Description']
    return df

# 2. Вычисление матрицы сходства (Кэшируем трансформированные матрицы)
@st.cache_resource
def calculate_similarity_matrix(_df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(_df["content_soup"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


df = load_data()
cosine_sim = calculate_similarity_matrix(df)


def get_recommendations(title, cosine_sim=cosine_sim, num_rec=5):
    try:
        idx = df[df["Title"] == title].index[0]
    except IndexError:
        return pd.DataFrame()

    # Считаем попарное сходство всех фильмов с выбранным
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Сортируем по убыванию скора сходства
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Берем топ-N схожих фильмов (пропуская самый первый — он является самим фильмом)
    sim_scores = sim_scores[1 : num_rec + 1]

    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

def get_recommendations_multi(titles, cosine_sim=cosine_sim, num_rec=5):
    rec_list = []
    for title in titles:
        try:
            idx = df.query('Title == @title').index[0]
        except IndexError:
            return pd.DataFrame()

        # Считаем попарное сходство всех фильмов с выбранным
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Сортируем по убыванию скора сходства
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Берем топ-N схожих фильмов (пропуская самый первый — он является самим фильмом)
        sim_scores = sim_scores[1 : num_rec + 1]

        movie_indices = [i[0] for i in sim_scores]
        rec_cands = df.iloc[movie_indices]
        rec_list.append(rec_cands)
    res_df = pd.concat(rec_list, ignore_index=True)
    res_df = res_df.query('Title not in @titles')
    return res_df.reset_index(drop=True)


# --- ИНТЕРФЕЙС STREAMLIT ---

st.title("🎬 Рекомендательная система")
st.write(
    "Выберите фильм, который вам нравится, и мы подберем картины со схожим жанром, сюжетом и атмосферой."
)

st.markdown("---")

# Создаем разметку на две колонки (Сайдбар слева и контент справа)
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Настройки")
    selected_movies = st.multiselect(
        "Ваши любимые фильмы:", df["Title"].values,
    )

    # Слайдер количества рекомендаций
    num_recommendations = st.slider(
        "Сколько фильмов порекомендовать?",
        min_value=1,
        max_value=5,
        value=3,
    )

    generate_btn = st.button("Найти похожие", type="primary")

with col2:
    st.header("Выбранные фильмы")
    if len(selected_movies) == 1:
        movie_info = df.query('Title in @selected_movies').iloc[0]
        # movie_info = df[df["Title"] == selected_movie].iloc[0]

        # Красиво выводим инфо о текущем фильме
        st.subheader(f"{movie_info['Title']} ({movie_info['year']})")
        st.caption(
            f"⭐: **{movie_info['Rate']}** | {movie_info['Genre']}"
        )
        st.image(movie_info['Poster_Link'])
        st.write(f"*Описание:* {movie_info['Info']}")
    else:
        st.write(f"{selected_movies}")
    st.markdown("---")

    # Логика выдачи рекомендаций
    if generate_btn:
        st.subheader("🍿 Рекомендуем посмотреть:")
        recommendations = get_recommendations_multi(
            selected_movies, num_rec=num_recommendations
        )

        # recommendations = get_recommendations(
        #     selected_movie, num_rec=num_recommendations
        # )
        print(recommendations)

        if not recommendations.empty:
            # Выводим карточки рекомендаций горизонтально (каждая в своей мини-колонке)
            n = len(recommendations)
            print('n:', n)
            cols = min(n, 3)
            rows = n // 4 + 1
            print('cols:', cols, 'rows:', rows)

            rec_cols = st.columns(cols)
            for i in range(rows):
                for j in range(cols):
                    index = i + j - 1
                    if index < len(recommendations):
                        cell = recommendations.iloc[index]
                        with rec_cols[j]:
                            # Оформляем карточку фильма с помощью markdown-контейнера
                            st.markdown(
                                f"""
                                <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; min-height:220px; border-left: 5px solid #ff4b4b;">
                                    <h4 style="margin:0 0 10px 0; color:#262730;">{cell['Title']} ({cell['year']})</h4>
                                    <p style="font-size:14px; margin:0;"><b>⭐</b> {cell['Rate']}</p>
                                    <p style="font-size:13px; color:#555; margin:5px 0;">{cell['Genre']}</p>
                                    <img src="{cell['Poster_Link']}"/>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
        else:
            st.error("Ошибка при генерации рекомендаций.")

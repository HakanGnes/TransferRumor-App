import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from player_segmentation import segment_players
from data_loader import load_players, get_available_seasons
import joblib

# Excel yerine artik SQLite DB'den (fetch_api_football_data.py tarafindan
# doldurulur) canli olarak okunuyor. DATA_PATH kaldirildi.
#
# NOT: yellowbrick'in KElbowVisualizer'i guncel scikit-learn surumleriyle
# uyumsuz hale geldi (estimator tip kontrolu hata veriyor: "not a clustering
# estimator"). Bu yuzden yellowbrick bagimliligi tamamen kaldirildi; elbow
# (dirsek) noktasi asagida birkac satirlik sade bir fonksiyonla (Kneedle
# yontemi - ilk/son noktayi birlestiren dogruya en uzak nokta) hesaplaniyor.


def _find_elbow_k(k_values, inertias):
    """Basit 'kneedle' yontemi: ilk-son noktayi birlestiren dogruya en uzak k."""
    if len(k_values) <= 2:
        return k_values[0]
    x1, y1 = k_values[0], inertias[0]
    x2, y2 = k_values[-1], inertias[-1]
    denom = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    if denom == 0:
        return k_values[0]
    distances = [
        abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        for x, y in zip(k_values, inertias)
    ]
    return k_values[int(np.argmax(distances))]


def _non_feature_columns(seasons):
    """KMeans'e sokulmayacak kimlik + segmentasyon metin sutunlari (sezona gore degisir)."""
    cols = ["PLAYER", "CLUB", "NATION", "VALUE", "POSITION", "LEAGUE"]
    for s in seasons:
        cols += [
            f"SEGMENT_{s}",
            f"SALES_EXPECTATION_PRICE_{s}",
            f"RECOMMEND_FOR_ACTION_{s}",
        ]
    return cols


@st.cache_data
def load_data():
    """
    SQLite'taki (api_football_data.db) en guncel sezonu ana veri, DB'de
    bulunan TUM sezonlari da segmentasyon icin kullanarak birlestirilmis
    DataFrame'i olusturur.
    """
    seasons = get_available_seasons()
    if not seasons:
        st.error(
            "SQLite veritabaninda (api_football_data.db) hic veri bulunamadi. "
            "Once fetch_api_football_data.py scriptini calistirip veri cek."
        )
        st.stop()

    latest_season = seasons[0]
    df_main = load_players(latest_season)
    if df_main.empty:
        st.error(f"En guncel sezon ({latest_season}) icin oyuncu verisi bulunamadi.")
        st.stop()
    df_main = df_main.drop_duplicates(subset=["PLAYER", "CLUB"])

    df = df_main
    for season in seasons:
        df_season = segment_players(season)
        if df_season.empty:
            continue
        df = pd.merge(df, df_season, on="PLAYER", how="left")

    return df, seasons


def get_kmeans_model(position, dataframe, non_feature_columns):
    model_path = f"kmeans_{position.lower()}.joblib"
    try:
        kmeans = joblib.load(model_path)
    except FileNotFoundError:
        data = dataframe[dataframe["POSITION"] == position]
        X = data.drop(non_feature_columns, axis=1, errors="ignore").values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        max_k = min(20, len(X_scaled) - 1)
        if max_k < 3:
            # Cok az oyuncu var (nadir durum); sabit kucuk bir k kullan.
            best_k = max(2, min(max_k, 2))
        else:
            k_values = list(range(2, max_k + 1))
            inertias = []
            for k in k_values:
                km = KMeans(n_clusters=k, random_state=17, n_init=10).fit(X_scaled)
                inertias.append(km.inertia_)
            best_k = _find_elbow_k(k_values, inertias)

        kmeans = KMeans(n_clusters=best_k, random_state=17, n_init=10).fit(X_scaled)
        joblib.dump(kmeans, model_path)
    return kmeans


def ilgilenilebilecek_oyuncular(dataframe, non_feature_columns):
    st.header("Transfer Player Prediction")
    takim = st.sidebar.selectbox("Team:", dataframe["CLUB"].unique())
    pozisyon = st.sidebar.selectbox("Position:", dataframe["POSITION"].unique())
    yas = st.sidebar.slider("Age Range:", min_value=16, max_value=40, key="yas_slider")

    value_available = dataframe["VALUE"].notna().any()
    if value_available:
        deger = st.sidebar.slider("Value Range:", min_value=0, max_value=150000000, step=100000, key="deger_slider")
    else:
        st.sidebar.info(
            "Piyasa degeri (Value) bilgisi API-Football ucretsiz planinda bulunmuyor, "
            "bu filtre devre disi."
        )
        deger = None

    if st.sidebar.button("Get Predictions🔍"):
        kmeans = get_kmeans_model(pozisyon, dataframe, non_feature_columns)

        position_df = dataframe[dataframe["POSITION"] == pozisyon].copy()

        X = position_df.drop(non_feature_columns, axis=1, errors="ignore").values

        position_df["CLUSTER"] = kmeans.predict(StandardScaler().fit_transform(X))
        position_df["CLUSTER"] = position_df["CLUSTER"] + 1

        target_cluster = round(position_df.loc[position_df["CLUB"] == takim, "CLUSTER"].mean())
        mask = (
            (position_df["POSITION"] == pozisyon)
            & (position_df["AGE"] <= yas)
            & (position_df["CLUB"] != takim)
            & (position_df["CLUSTER"] == target_cluster)
        )
        if value_available:
            mask &= position_df["VALUE"] <= deger

        transfer_edilebilecekler = position_df.loc[mask]

        st.write(transfer_edilebilecekler[["PLAYER", "CLUB", "POSITION", "AGE", "VALUE"]])


def oyuncu_kazanc_beklentisi(dataframe, seasons):
    st.header("Sales Expectation and Performance Analysis")
    takim2 = st.sidebar.selectbox("Team: ", dataframe["CLUB"].unique()).upper()
    season = st.sidebar.selectbox("Season:", [str(s) for s in seasons])

    if st.sidebar.button("Get Predictions🔍"):
        segment_col = f"SEGMENT_{season}"
        performance_score_col = f"PERFORMANCE_SCORE_{season}"
        sales_exp_col = f"SALES_EXPECTATION_PRICE_{season}"

        oyuncu_sonuc = dataframe[dataframe["CLUB"] == takim2][["PLAYER", sales_exp_col, segment_col, performance_score_col]]
        st.write(oyuncu_sonuc)


def oyunculara_göre_aksiyon_tavsiyesi(dataframe, seasons):
    st.header("Recommendation for Action")
    takim2 = st.sidebar.selectbox("Team: ", dataframe["CLUB"].unique()).upper()
    season = st.sidebar.selectbox("Season:", [str(s) for s in seasons])

    if st.sidebar.button("Get Recommendations🔍"):
        recommendation_col = f"RECOMMEND_FOR_ACTION_{season}"
        takim_df = dataframe.loc[(dataframe["CLUB"] == takim2), ["PLAYER", "CLUB", "AGE", "POSITION", recommendation_col]]
        st.write(takim_df)


def main():
    new_title = '<p style="font-family:algerian; color:White; font-size: 55px;">TRANSFER RUMOR⚽️</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    df, seasons = load_data()
    non_feature_columns = _non_feature_columns(seasons)

    selected_option = st.sidebar.radio(
        "Choose Your Action:",
        ("Transfer Player Prediction", "Sales Expectation and Performance Analysis", "Recommendation for Action"),
    )

    if selected_option == "Transfer Player Prediction":
        ilgilenilebilecek_oyuncular(df, non_feature_columns)
    elif selected_option == "Sales Expectation and Performance Analysis":
        oyuncu_kazanc_beklentisi(df, seasons)
    else:
        oyunculara_göre_aksiyon_tavsiyesi(df, seasons)


if __name__ == "__main__":
    main()

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.hdqwalls.com/wallpapers/football-ground-sun-rays-4k-ev.jpg");
background-size: 110%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
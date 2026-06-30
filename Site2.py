import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from player_segmentation import segment_players
import joblib

DATA_PATH = "./no_nans_data1.xlsx"

@st.cache_data
def load_data():
    """
    Loads all the necessary data for the application.
    """
    df_main = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
    df_main = df_main.drop_duplicates()
    df_main.columns = df_main.columns.str.upper()
    df_main[['CLUB', 'POSITION']] = df_main[['CLUB', 'POSITION']].applymap(lambda x: x.upper())

    df_19_20 = segment_players("19/20")
    df_20_21 = segment_players("20/21")

    # Merge the dataframes
    df = pd.merge(df_main, df_19_20, on="PLAYER", how="left")
    df = pd.merge(df, df_20_21, on="PLAYER", how="left")

    return df

def get_kmeans_model(position, dataframe):
    model_path = f"kmeans_{position.lower()}.joblib"
    try:
        kmeans = joblib.load(model_path)
    except FileNotFoundError:
        if position == "ATTACK":
            data = dataframe[dataframe['POSITION'] == 'ATTACK']
            X = data.drop(["PLAYER", "CLUB", "NATION", "VALUE", "POSITION", "LEAGUE"], axis=1, errors='ignore').values
        elif position == "MIDFIELD":
            data = dataframe[dataframe['POSITION'] == 'MIDFIELD']
            X = data.drop(["PLAYER", "CLUB", "NATION", "VALUE", "POSITION", "LEAGUE"], axis=1, errors='ignore').values
        else:
            data = dataframe[dataframe['POSITION'] == 'DEFENDER']
            X = data.drop(["PLAYER", "CLUB", "NATION", "VALUE", "POSITION", "LEAGUE"], axis=1, errors='ignore').values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(random_state=17)
        elbow = KElbowVisualizer(kmeans, k=(2, 20))
        elbow.fit(X_scaled)
        kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=17).fit(X_scaled)
        joblib.dump(kmeans, model_path)
    return kmeans

def ilgilenilebilecek_oyuncular(dataframe):
    st.header("Transfer Player Prediction")
    takim = st.sidebar.selectbox("Team:", dataframe["CLUB"].unique())
    pozisyon = st.sidebar.selectbox("Position:", dataframe["POSITION"].unique())
    yas = st.sidebar.slider("Age Range:", min_value=16, max_value=40, key="yas_slider")
    deger = st.sidebar.slider("Value Range:",min_value=0, max_value=150000000,step=100000,key="deger_slider")

    if st.sidebar.button("Get Predictions🔍"):
        kmeans = get_kmeans_model(pozisyon, dataframe)
        
        position_df = dataframe[dataframe['POSITION'] == pozisyon].copy()
        
        X = position_df.drop(["PLAYER", "CLUB", "NATION", "VALUE", "POSITION", "LEAGUE"], axis=1, errors='ignore').values
        
        position_df["CLUSTER"] = kmeans.predict(StandardScaler().fit_transform(X))
        position_df["CLUSTER"] = position_df["CLUSTER"] + 1
        
        transfer_edilebilecekler = position_df.loc[(position_df["POSITION"] == pozisyon) & (position_df["AGE"] <= yas) & (position_df["VALUE"] <= deger) & (position_df["CLUB"] != takim) & (position_df["CLUSTER"] == round(position_df.loc[position_df["CLUB"] == takim, "CLUSTER"].mean()))]

        st.write(transfer_edilebilecekler[["PLAYER", "CLUB", "POSITION", "AGE", "VALUE"]])


def oyuncu_kazanc_beklentisi(dataframe):
    st.header("Sales Expectation and Performance Analysis")
    takim2 = st.sidebar.selectbox("Team: ", dataframe["CLUB"].unique()).upper()
    season = st.sidebar.selectbox("Season:", ["20/21", "19/20"])
    
    if st.sidebar.button("Get Predictions🔍"):
        segment_col = f"SEGMENT{season.replace('/', '_')}"
        performance_score_col = f"PERFORMANCE_SCORE_{season.replace('/', '_')}"
        sales_exp_col = "SALES_EXPECTATION_PRICE_x" if season == "20/21" else "SALES_EXPECTATION_PRICE_y"
        
        oyuncu_sonuc = dataframe[dataframe["CLUB"] == takim2][["PLAYER", sales_exp_col, segment_col, performance_score_col]]
        st.write(oyuncu_sonuc)


def oyunculara_göre_aksiyon_tavsiyesi(dataframe):
    st.header("Recommendation for Action")
    takim2 = st.sidebar.selectbox("Team: ", dataframe["CLUB"].unique())
    season = st.sidebar.selectbox("Season:", ["20/21", "19/20"])
    
    if st.sidebar.button("Get Recommendations🔍"):
        recommendation_col = "RECOMMEND_FOR_ACTION_x" if season == "20/21" else "RECOMMEND_FOR_ACTION_y"
        takim_df = dataframe.loc[(dataframe["CLUB"] == takim2), ["PLAYER", "CLUB", "AGE", "POSITION", recommendation_col]]
        st.write(takim_df)


def main():
    new_title = '<p style="font-family:algerian; color:White; font-size: 55px;">TRANSFER RUMOR⚽️</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    df = load_data()

    selected_option = st.sidebar.radio("Choose Your Action:", ("Transfer Player Prediction", "Sales Expectation and Performance Analysis", "Recommendation for Action"))

    if selected_option == "Transfer Player Prediction":
        ilgilenilebilecek_oyuncular(df)
    elif selected_option == "Sales Expectation and Performance Analysis":
        oyuncu_kazanc_beklentisi(df)
    else:
        oyunculara_göre_aksiyon_tavsiyesi(df)


if __name__ == '__main__':
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

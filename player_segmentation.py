"""
player_segmentation.py
========================
API-Football'dan cekilen ham istatistiklerle (bkz. data_loader.py) her sezon
icin DINAMIK olarak oyuncu segmentasyonu hesaplar. Excel/precomputed veriye
bagimlilik yok; SQLite DB'de veri oldugu surece herhangi bir sezon icin
calisir.

METODOLOJI (orijinal projedeki mantigin API-Football'da mevcut olan daha
az sayidaki istatistige uyarlanmis hali):
    1. Her pozisyon (ATTACK/MIDFIELD/DEFENDER) icin ilgili metrikler
       yuzdelik dilime (percentile) gore 1-5 arasi puanlanir.
    2. Puanlar toplanip Total_Score elde edilir.
    3. Total_Score'un kendisi de percentile'a gore 5 esit dilime bolunerek
       Performance kategorisi (Under_expected .. Flawless) belirlenir.
       (Sabit bin sinirlari yerine percentile kullanilmasinin sebebi:
       farkli sezon/lig/veri kaynagi kombinasyonlarinda skor araligi
       degisebilir, percentile bu degisime otomatik uyum saglar.)
    4. Age_Cat (Young/Experienced/Mature/End_Of_Career) + Performance
       birlestirilip Segment string'i olusturulur (orn. "Young_Flawless").
    5. Segment string'inden SALES_EXPECTATION_PRICE ve RECOMMEND_FOR_ACTION
       sabit lookup tablolariyla turetilir (bu kisim orijinal projeyle ayni).
"""

import warnings
import numpy as np
import pandas as pd

from data_loader import load_players, get_available_seasons  # noqa: F401 (get_available_seasons Site2.py icin disari aciliyor)


# --------------------------------------------------------------------------
# Sales Expectation / Recommend For Action lookup tablolari
# (segment string -> deger). Bu kisim veri kaynagindan bagimsiz, oldugu gibi
# korunuyor.
# --------------------------------------------------------------------------

_SALES_EXPECTATION_MAP = {
    "Young_Under_expected": "Low",
    "Young_Open_to_development": "Low - Mid",
    "Young_Player_with_high_potential": "Mid",
    "Young_High_performance": "High",
    "Young_Flawless": "Very_High",
    "Experienced_Under_expected": "Low",
    "Experienced_Open_to_development": "Low - Mid",
    "Experienced_Player_with_high_potential": "Mid",
    "Experienced_High_performance": "Mid - High",
    "Experienced_Flawless": "Very_High",
    "Mature_Under_expected": "Very_Low",
    "Mature_Open_to_development": "Low",
    "Mature_Player_with_high_potential": "Mid",
    "Mature_High_performance": "Mid - High",
    "Mature_Flawless": "High",
    "End_Of_Career_Under_expected": "Very_Low",
    "End_Of_Career_Open_to_development": "Low",
    "End_Of_Career_Player_with_high_potential": "Low",
    "End_Of_Career_High_performance": "Mid",
    "End_Of_Career_Flawless": "Mid - High",
}

_RECOMMEND_FOR_ACTION_MAP = {
    "Young_Under_expected": "Considering the potential of these young players, encourage them to improve their performance with extra work and training. By taking a long-term approach, support their development and show patience.",
    "Young_Open_to_development": "Create special training programs to maximize the potential of these young players. Help them gain experience by regularly giving them chances in first team matches.",
    "Young_Player_with_high_potential": "Young and high-potential players can be an important part of the team in the future. Focusing on opportunities for these players to develop their skills and gain experience can greatly benefit in the long run.",
    "Young_High_performance": "Help these young talents improve their physical condition and technical skills so that they can maintain their high performance. Encourage them to show that they are ready to give leadership roles within the team.",
    "Young_Flawless": "Help these young talents maximize their physical and technical abilities so that they can maintain their excellent performance. Encourage them to evaluate media and marketing opportunities to gain more visibility.",
    "Experienced_Under_expected": "Create individual training plans for experienced players to improve their performance and increase their motivation. Based on their past accomplishments, allow them to focus more on the team leadership role.",
    "Experienced_Open_to_development": "Take a long-term approach to preparing these experienced players for the future. Encourage them to build mentoring relationships with young players and share their knowledge.",
    "Experienced_Player_with_high_potential": "Experienced and high-potential players can make an immediate contribution to the team. Thanks to the experience they have, these players can lead in tough matches and guide young players. At the same time, making a special effort to further develop the potential of these players can increase the team's chances of success",
    "Experienced_High_performance": "Support experienced players to maintain a high level of performance. Consider increasing their leadership as one of the key players on the team, giving them more responsibility within the team.",
    "Experienced_Flawless": "Support experienced players to maintain flawless performances and strengthen their leadership roles. Strategically engage them to enable them to take on more responsibility within the team.",
    "Mature_Under_expected": "Develop a special rehabilitation and training program to help mature players return to their jerseys. Set new goals to boost their motivation and rekindle their desire to improve their performance.",
    "Mature_Open_to_development": "Create individual training and training plans to enable these mature players to develop further. Consider giving leadership roles within the team and encourage them to share their experiences with younger players.",
    "Mature_Player_with_high_potential": "Create a strategic plan to maximize the high potential of these mature players. Ensure that they maintain the proper balance of training and rest so that they can maintain their performance.",
    "Mature_High_performance": "Help mature players maintain their high level of performance and encourage them to make more impact within the team by increasing their leadership. Encourage young players to share their experiences by mentoring them.",
    "Mature_Flawless": "Help mature players maintain flawless performances and encourage them to make more impact within the team by increasing their leadership. Encourage young players to share their experiences by mentoring them",
    "End_Of_Career_Under_expected": "Provide specific support and motivation for players nearing the end of their careers to rotate their jerseys. Consider mentoring roles or assistant coaching positions to allow the team to benefit from their experience.",
    "End_Of_Career_Open_to_development": "Help players nearing the end of their careers prepare their final stages for the future of the team. Support players in thinking about their own post-career plans and training.",
    "End_Of_Career_Player_with_high_potential": "Players who are nearing the end of their careers but still have high potential can be a valuable asset to teams. Thanks to their experience, these players can give advice to young talents and increase  the morale of the team with their leadership on the field. The future contributions of these players must be carefully evaluated and aligned with the team's overall strategy.",
    "End_Of_Career_High_performance": "Provide physical and psychological support to help these players maintain their performance as they approach the end of their careers. Take full advantage of their experience by increasing their leadership role within the team.",
    "End_Of_Career_Flawless": "Help players near the end of their careers maintain flawless performances and strengthen their leadership roles. Prepare players for mentoring or managerial roles to support their post-career plans.",
}

_PERFORMANCE_LABELS = {
    1: "Under_expected",
    2: "Open_to_development",
    3: "Player_with_high_potential",
    4: "High_performance",
    5: "Flawless",
}

_AGE_BINS = [15, 22, 27, 32, 45]
_AGE_LABELS = ["Young", "Experienced", "Mature", "End_Of_Career"]

# Pozisyona gore hangi metrikler kullanilsin ve yon (True: yuksek daha iyi,
# False: dusuk daha iyi). Bir metrik listede birden fazla kez gecerse o
# metrigin toplam skordaki agirligi artar (RATING gibi genel/ozet bir
# metrigi -defansif detay istatistigi eksik oldugu icin- daha agirlikli
# kullaniyoruz).
_POSITION_METRICS = {
    "ATTACK": [
        ("GOALS", True),
        ("ASSISTS", True),
        ("RATING", True),
        ("RATING", True),
        ("MINUTES", True),
        ("APPEARANCES", True),
        ("CARDS", False),
    ],
    "MIDFIELD": [
        ("ASSISTS", True),
        ("GOALS", True),
        ("RATING", True),
        ("RATING", True),
        ("MINUTES", True),
        ("APPEARANCES", True),
        ("CARDS", False),
    ],
    "DEFENDER": [
        ("RATING", True),
        ("RATING", True),
        ("RATING", True),
        ("MINUTES", True),
        ("APPEARANCES", True),
        ("CARDS", False),
    ],
}


def _percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Bir sayisal seriyi 1-5 arasi tam sayi skora cevirir (percentile tabanli).
    pd.qcut'un aksine, tekrar eden degerlerde ('bin edges must be unique')
    hata vermez ve tek oyunculu gruplarda bile calisir.
    """
    if series.nunique(dropna=True) <= 1:
        # Herkes ayni degere sahipse ayirt edici degildir, notr skor (3) verilir.
        return pd.Series(3, index=series.index)

    pct = series.rank(pct=True, ascending=ascending, method="average")
    score = np.ceil(pct * 5).clip(lower=1, upper=5)
    return score.astype(int)


def _compute_segment_for_position(df_pos: pd.DataFrame, position: str) -> pd.DataFrame:
    df_pos = df_pos.copy()
    df_pos["CARDS"] = df_pos["YELLOW_CARDS"].fillna(0) + df_pos["RED_CARDS"].fillna(0) * 2

    total_score = pd.Series(0, index=df_pos.index)
    for metric, ascending in _POSITION_METRICS[position]:
        total_score = total_score + _percentile_score(df_pos[metric], ascending=ascending)

    df_pos["TOTAL_SCORE"] = total_score
    performance_score = _percentile_score(df_pos["TOTAL_SCORE"], ascending=True)
    df_pos["PERFORMANCE_LABEL"] = performance_score.map(_PERFORMANCE_LABELS)
    df_pos["PERFORMANCE_SCORE"] = performance_score.astype(str)

    age_cat = pd.cut(df_pos["AGE"], bins=_AGE_BINS, labels=_AGE_LABELS, include_lowest=True)
    age_cat = age_cat.astype(str).replace("nan", "Mature")  # aralik disi yaslar icin guvenli varsayilan

    df_pos["SEGMENT"] = age_cat.astype(str) + "_" + df_pos["PERFORMANCE_LABEL"].astype(str)
    return df_pos


def segment_players(season):
    """
    Verilen sezon (API-Football sezon yili, orn. 2024) icin dinamik
    segmentasyon hesaplar.

    Donen sutunlar: PLAYER, SEGMENT_<season>, PERFORMANCE_SCORE_<season>,
    SALES_EXPECTATION_PRICE_<season>, RECOMMEND_FOR_ACTION_<season>
    """
    try:
        season_int = int(str(season).replace("/", "").strip()[:4]) if not isinstance(season, int) else season
    except ValueError:
        warnings.warn(f"Gecersiz sezon degeri: {season}")
        return pd.DataFrame()

    df = load_players(season_int)
    if df.empty:
        warnings.warn(
            f"Sezon {season_int} icin DB'de veri bulunamadi. "
            f"Once fetch_api_football_data.py'yi bu sezon icin calistirdigindan emin ol."
        )
        return pd.DataFrame()

    segmented_parts = []
    for position in ["ATTACK", "MIDFIELD", "DEFENDER"]:
        df_pos = df[df["POSITION"] == position]
        if df_pos.empty:
            continue
        segmented_parts.append(_compute_segment_for_position(df_pos, position))

    if not segmented_parts:
        return pd.DataFrame()

    result = pd.concat(segmented_parts, axis=0)

    season_key = str(season_int)
    segment_col = f"SEGMENT_{season_key}"
    performance_score_col = f"PERFORMANCE_SCORE_{season_key}"
    sales_exp_col = f"SALES_EXPECTATION_PRICE_{season_key}"
    recommendation_col = f"RECOMMEND_FOR_ACTION_{season_key}"

    result[segment_col] = result["SEGMENT"]
    result[performance_score_col] = result["PERFORMANCE_SCORE"]
    result[sales_exp_col] = result[segment_col].map(_SALES_EXPECTATION_MAP)
    result[recommendation_col] = result[segment_col].map(_RECOMMEND_FOR_ACTION_MAP)

    final_total_df = result[
        ["PLAYER", segment_col, performance_score_col, sales_exp_col, recommendation_col]
    ].copy()
    return final_total_df
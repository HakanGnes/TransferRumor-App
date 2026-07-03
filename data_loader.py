"""
data_loader.py
================
api_football_data.db (fetch_api_football_data.py tarafindan olusturulur)
icindeki teams/players tablolarini okuyup, Site2.py ve player_segmentation.py
icin kullanilabilir bir pandas DataFrame'e cevirir.

API-Football UCRETSIZ PLANIN GETIRMEDIGI VERI: piyasa degeri (market value).
Bu yuzden VALUE sutunu NaN olarak doner; Site2.py bunu buna gore ele alir.
"""

import sqlite3
import pandas as pd

DB_PATH = "api_football_data.db"

# fetch_api_football_data.py'deki TARGET_LEAGUES ile birebir ayni tutulmali
LEAGUE_NAMES = {
    39: "PREMIER LEAGUE",
    140: "LA LIGA",
    135: "SERIE A",
    78: "BUNDESLIGA",
    61: "LIGUE 1",
}

# API-Football pozisyon stringlerini projenin kullandigi 3 kategoriye esler.
# Kaleciler orijinal projede de desteklenmiyordu, o yuzden disarida birakiliyor.
POSITION_MAP = {
    "Attacker": "ATTACK",
    "Midfielder": "MIDFIELD",
    "Defender": "DEFENDER",
    "Goalkeeper": None,
}


def get_available_seasons(db_path: str = DB_PATH) -> list:
    """DB'de veri bulunan sezonlari (en yeniden en eskiye) dondurur."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT season FROM players ORDER BY season DESC"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def load_players(season: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Belirli bir sezon icin teams+players tablolarini birlestirip
    Site2.py'nin bekledigi sekilde uppercase sutun isimleriyle dondurur.

    Donen sutunlar:
        PLAYER, CLUB, LEAGUE, NATION, AGE, POSITION, VALUE (NaN),
        APPEARANCES, MINUTES, GOALS, ASSISTS, YELLOW_CARDS, RED_CARDS, RATING
    """
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT
                p.player_id, p.name AS player, p.age, p.nationality,
                p.position, p.appearances, p.minutes, p.goals, p.assists,
                p.yellow_cards, p.red_cards, p.rating,
                t.name AS club, t.league_id
            FROM players p
            JOIN teams t ON p.team_id = t.team_id AND p.season = t.season
            WHERE p.season = ?
        """
        df = pd.read_sql_query(query, conn, params=(season,))
    finally:
        conn.close()

    if df.empty:
        return df

    df["position"] = df["position"].map(POSITION_MAP)
    df = df.dropna(subset=["position"])  # kaleciler ve bilinmeyen pozisyonlar disarida

    df["league"] = df["league_id"].map(LEAGUE_NAMES)
    df["value"] = pd.NA  # API-Football ucretsiz planda piyasa degeri yok

    df = df.rename(
        columns={
            "player": "PLAYER",
            "club": "CLUB",
            "league": "LEAGUE",
            "nationality": "NATION",
            "age": "AGE",
            "position": "POSITION",
            "value": "VALUE",
            "appearances": "APPEARANCES",
            "minutes": "MINUTES",
            "goals": "GOALS",
            "assists": "ASSISTS",
            "yellow_cards": "YELLOW_CARDS",
            "red_cards": "RED_CARDS",
            "rating": "RATING",
        }
    )

    df["CLUB"] = df["CLUB"].str.upper()
    df["NATION"] = df["NATION"].str.upper()

    keep_cols = [
        "PLAYER", "CLUB", "LEAGUE", "NATION", "AGE", "POSITION", "VALUE",
        "APPEARANCES", "MINUTES", "GOALS", "ASSISTS", "YELLOW_CARDS",
        "RED_CARDS", "RATING",
    ]
    df = df[keep_cols].drop_duplicates(subset=["PLAYER", "CLUB"]).reset_index(drop=True)

    # Eksik sayisal degerler (rating gibi) medyanla dolduruluyor ki qcut
    # tabanli skorlama NaN yuzunden patlamasin.
    numeric_cols = ["AGE", "APPEARANCES", "MINUTES", "GOALS", "ASSISTS", "YELLOW_CARDS", "RED_CARDS", "RATING"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df
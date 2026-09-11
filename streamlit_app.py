
from __future__ import annotations

import io
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="NFL Totals Lab — Auto Data",
    page_icon="🏈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# nflverse uses LA for the Rams.
ODDS_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
ABBR_TO_ODDS_NAME = {v: k for k, v in ODDS_NAME_TO_ABBR.items()}

# Stadium-area coordinates used directly for weather.
# This avoids city-name geocoding failures and ties the forecast to the venue area.
WEATHER_COORDS = {
    "ARI": (33.5276, -112.2626, "State Farm Stadium"),
    "ATL": (33.7554, -84.4008, "Mercedes-Benz Stadium"),
    "BAL": (39.2780, -76.6227, "M&T Bank Stadium"),
    "BUF": (42.7738, -78.7868, "Highmark Stadium"),
    "CAR": (35.2258, -80.8528, "Bank of America Stadium"),
    "CHI": (41.8623, -87.6167, "Soldier Field"),
    "CIN": (39.0955, -84.5161, "Paycor Stadium"),
    "CLE": (41.5061, -81.6995, "Huntington Bank Field"),
    "DAL": (32.7473, -97.0945, "AT&T Stadium"),
    "DEN": (39.7439, -105.0201, "Empower Field at Mile High"),
    "DET": (42.3400, -83.0456, "Ford Field"),
    "GB": (44.5013, -88.0622, "Lambeau Field"),
    "HOU": (29.6847, -95.4107, "NRG Stadium"),
    "IND": (39.7601, -86.1639, "Lucas Oil Stadium"),
    "JAX": (30.3239, -81.6373, "EverBank Stadium"),
    "KC": (39.0489, -94.4839, "Arrowhead Stadium"),
    "LV": (36.0908, -115.1830, "Allegiant Stadium"),
    "LAC": (33.9535, -118.3392, "SoFi Stadium"),
    "LA": (33.9535, -118.3392, "SoFi Stadium"),
    "MIA": (25.9580, -80.2389, "Hard Rock Stadium"),
    "MIN": (44.9738, -93.2581, "U.S. Bank Stadium"),
    "NE": (42.0909, -71.2643, "Gillette Stadium"),
    "NO": (29.9511, -90.0812, "Caesars Superdome"),
    "NYG": (40.8135, -74.0745, "MetLife Stadium"),
    "NYJ": (40.8135, -74.0745, "MetLife Stadium"),
    "PHI": (39.9008, -75.1675, "Lincoln Financial Field"),
    "PIT": (40.4468, -80.0158, "Acrisure Stadium"),
    "SEA": (47.5952, -122.3316, "Lumen Field"),
    "SF": (37.4030, -121.9700, "Levi's Stadium"),
    "TB": (27.9759, -82.5033, "Raymond James Stadium"),
    "TEN": (36.1665, -86.7713, "Nissan Stadium"),
    "WAS": (38.9076, -76.8645, "Northwest Stadium"),
}

# Roof classification. Retractable-roof games still need a game-day open/closed choice.
FIXED_ROOF = {"DET", "MIN", "NO", "LV", "LA", "LAC"}
RETRACTABLE_ROOF = {"ARI", "ATL", "DAL", "HOU", "IND"}

FEATURES = [
    "EPA_Matchup",
    "Pass_EPA_Matchup",
    "Rush_EPA_Matchup",
    "Success_Matchup",
    "Explosive_Matchup",
    "Plays_Matchup",
    "Turnover_Env",
    "Sack_Env",
    "Wind",
    "Cold_Units",
    "Dome",
    "Short_Rest",
    "Div_Game",
]

PBP_COLS = [
    "game_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "pass",
    "rush",
    "epa",
    "yards_gained",
    "interception",
    "fumble_lost",
    "sack",
]

# -----------------------------
# Local model files
# -----------------------------
@st.cache_data
def load_model_files():
    coef_df = pd.read_csv(ROOT / "model_coefficients.csv")
    coefficients = dict(zip(coef_df["Feature"], coef_df["Coefficient"].astype(float)))
    holdout = pd.read_csv(ROOT / "backtest_summary.csv").iloc[0].to_dict()
    historical = pd.read_csv(ROOT / "historical_features.csv")
    return coefficients, holdout, historical

COEF, HOLDOUT, HISTORICAL = load_model_files()

# -----------------------------
# Network data loaders
# -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_schedule() -> pd.DataFrame:
    r = requests.get(SCHEDULE_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_pbp(season: int) -> pd.DataFrame:
    url = PBP_URL.format(season=season)
    # Reading from bytes gives clearer errors than handing pandas a redirecting URL.
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(
        io.BytesIO(r.content),
        compression="gzip",
        usecols=lambda c: c in PBP_COLS,
        low_memory=False,
    )
    missing = [c for c in PBP_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"nflverse play-by-play is missing expected columns: {missing}")
    return df

@st.cache_data(ttl=120, show_spinner=False)
def fetch_odds(_api_key: str):
    if not _api_key:
        return []
    params = {
        "apiKey": _api_key,
        "regions": "us",
        "markets": "totals,spreads",
        "oddsFormat": "american",
    }
    r = requests.get(ODDS_URL, params=params, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key. Check the key and try again.")
    if r.status_code == 429:
        raise RuntimeError("The Odds API usage limit has been reached.")
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(lat: float, lon: float, venue_name: str, kickoff_utc_iso: str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "forecast_days": 16,
    }
    r = requests.get(WEATHER_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    tz_name = payload.get("timezone") or "America/New_York"
    kickoff_utc = datetime.fromisoformat(kickoff_utc_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    local_dt = kickoff_utc.astimezone(ZoneInfo(tz_name))
    target = local_dt.replace(minute=0, second=0, microsecond=0)

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    naive_target = target.replace(tzinfo=None)
    idx = (hourly["time"] - pd.Timestamp(naive_target)).abs().idxmin()
    row = hourly.loc[idx]

    return {
        "resolved_name": venue_name,
        "timezone": tz_name,
        "kickoff_local": local_dt,
        "temperature": float(row["temperature_2m"]),
        "wind": float(row["wind_speed_10m"]),
        "gust": float(row["wind_gusts_10m"]),
        "precip": float(row["precipitation_probability"]),
    }

# -----------------------------
# Football metrics
# -----------------------------
@st.cache_data(show_spinner=False)
def aggregate_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    for c in ["pass", "rush", "epa", "yards_gained", "interception", "fumble_lost", "sack", "week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[
        (df["season_type"] == "REG")
        & df["posteam"].notna()
        & df["defteam"].notna()
        & df["epa"].notna()
        & ((df["pass"] == 1) | (df["rush"] == 1))
    ].copy()

    df["success"] = (df["epa"] > 0).astype(float)
    df["explosive"] = (df["yards_gained"].fillna(0) >= 20).astype(float)
    df["turnover"] = (
        (df["interception"].fillna(0) == 1) | (df["fumble_lost"].fillna(0) == 1)
    ).astype(float)
    df["pass_epa_value"] = df["epa"].where(df["pass"] == 1)
    df["rush_epa_value"] = df["epa"].where(df["rush"] == 1)

    g = (
        df.groupby(["game_id", "posteam", "defteam", "week"], as_index=False)
        .agg(
            epa=("epa", "mean"),
            pass_epa=("pass_epa_value", "mean"),
            rush_epa=("rush_epa_value", "mean"),
            success=("success", "mean"),
            explosive=("explosive", "mean"),
            plays=("epa", "size"),
            turnovers=("turnover", "sum"),
            pass_plays=("pass", "sum"),
            sacks=("sack", "sum"),
        )
    )
    g["turnover_rate"] = g["turnovers"] / g["plays"].clip(lower=1)
    g["sack_rate"] = g["sacks"] / g["pass_plays"].clip(lower=1)
    return g

def build_histories(metrics: pd.DataFrame, max_week: int | None = None):
    m = metrics.copy()
    if max_week is not None:
        m = m[m["week"] < max_week]
    m = m.sort_values(["week", "game_id"])
    states: dict[str, dict[str, list[dict]]] = {}

    for _, game in m.groupby("game_id", sort=False):
        rows = list(game.to_dict("records"))
        if len(rows) < 2:
            continue
        by_team = {r["posteam"]: r for r in rows}
        for r in rows:
            team = r["posteam"]
            opp = r["defteam"]
            if opp not in by_team:
                continue
            states.setdefault(team, {"off": [], "def": []})
            metric_keys = ["epa", "pass_epa", "rush_epa", "success", "explosive", "plays", "turnover_rate", "sack_rate"]
            off = {k: r.get(k) for k in metric_keys}
            deff = {k: by_team[opp].get(k) for k in metric_keys}
            states[team]["off"].append(off)
            states[team]["def"].append(deff)
    return states

def mean_valid(values):
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    return sum(vals) / len(vals) if vals else None

def blended_metric(history: list[dict], metric: str, season_weight=0.65, last4_weight=0.35):
    if not history:
        return None
    season = mean_valid([x.get(metric) for x in history])
    recent = mean_valid([x.get(metric) for x in history[-4:]])
    if season is None:
        return recent
    if recent is None:
        return season
    return season_weight * season + last4_weight * recent

def combine_state(prev_states, cur_states, team, strict: bool):
    current = cur_states.get(team, {"off": [], "def": []})
    if strict or len(current["off"]) >= 4:
        return current, False

    previous = prev_states.get(team, {"off": [], "def": []})
    # Experimental early-season carryover: use enough of the previous season's final
    # four games to reach at least four pregame games.
    needed = max(0, 4 - len(current["off"]))
    off = previous["off"][-needed:] + current["off"] if needed else current["off"]
    deff = previous["def"][-needed:] + current["def"] if needed else current["def"]
    return {"off": off, "def": deff}, needed > 0

def matchup_features(away: str, home: str, prev_states, cur_states, strict: bool):
    a, a_prior = combine_state(prev_states, cur_states, away, strict)
    h, h_prior = combine_state(prev_states, cur_states, home, strict)

    if len(a["off"]) < 4 or len(h["off"]) < 4:
        return None, {
            "away_games": len(cur_states.get(away, {"off": []})["off"]),
            "home_games": len(cur_states.get(home, {"off": []})["off"]),
            "used_prior": False,
        }

    def metric(state, side, name):
        return blended_metric(state[side], name)

    def mu(*values):
        x = mean_valid(values)
        return 0.0 if x is None else x

    def matchup(name):
        away_expected = mu(metric(a, "off", name), metric(h, "def", name))
        home_expected = mu(metric(h, "off", name), metric(a, "def", name))
        return away_expected + home_expected

    f = {
        "EPA_Matchup": matchup("epa"),
        "Pass_EPA_Matchup": matchup("pass_epa"),
        "Rush_EPA_Matchup": matchup("rush_epa"),
        "Success_Matchup": matchup("success"),
        "Explosive_Matchup": matchup("explosive"),
        "Plays_Matchup": mu(metric(a, "off", "plays")) + mu(metric(h, "off", "plays")),
        "Turnover_Env": matchup("turnover_rate"),
        "Sack_Env": matchup("sack_rate"),
    }
    meta = {
        "away_games": len(cur_states.get(away, {"off": []})["off"]),
        "home_games": len(cur_states.get(home, {"off": []})["off"]),
        "used_prior": a_prior or h_prior,
    }
    return f, meta

# -----------------------------
# Odds and model helpers
# -----------------------------
def to_float(v, default=None):
    try:
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default

def season_from_kickoff(kickoff: datetime) -> int:
    return kickoff.year if kickoff.month >= 7 else kickoff.year - 1

def schedule_kickoff_utc(game_row) -> datetime:
    gameday = pd.Timestamp(game_row["gameday"]).date()
    gt = str(game_row.get("gametime") or "13:00")
    try:
        hh, mm = [int(x) for x in gt.split(":")[:2]]
    except Exception:
        hh, mm = 13, 0
    eastern = datetime(gameday.year, gameday.month, gameday.day, hh, mm, tzinfo=ZoneInfo("America/New_York"))
    return eastern.astimezone(timezone.utc)

def upcoming_schedule(schedule: pd.DataFrame, days_ahead=10):
    today = pd.Timestamp(date.today())
    end = today + pd.Timedelta(days=days_ahead)
    x = schedule[
        (schedule["game_type"] == "REG")
        & (schedule["gameday"] >= today)
        & (schedule["gameday"] <= end)
    ].copy()
    return x.sort_values(["gameday", "gametime", "game_id"])

def match_odds_event(odds_events, away: str, home: str):
    for e in odds_events:
        if ODDS_NAME_TO_ABBR.get(e.get("away_team")) == away and ODDS_NAME_TO_ABBR.get(e.get("home_team")) == home:
            return e
    return None

def book_market_rows(event):
    rows = []
    if not event:
        return rows
    for b in event.get("bookmakers", []):
        total_market = next((m for m in b.get("markets", []) if m.get("key") == "totals"), None)
        spread_market = next((m for m in b.get("markets", []) if m.get("key") == "spreads"), None)
        if not total_market:
            continue
        over = next((o for o in total_market.get("outcomes", []) if o.get("name") == "Over"), None)
        under = next((o for o in total_market.get("outcomes", []) if o.get("name") == "Under"), None)
        if not over or not under:
            continue

        home_spread = None
        if spread_market:
            home_name = event.get("home_team")
            home_out = next((o for o in spread_market.get("outcomes", []) if o.get("name") == home_name), None)
            if home_out:
                home_spread = to_float(home_out.get("point"))

        rows.append({
            "key": b.get("key"),
            "book": b.get("title", b.get("key")),
            "total": to_float(over.get("point")),
            "over": to_float(over.get("price")),
            "under": to_float(under.get("price")),
            "home_spread": home_spread,
            "updated": b.get("last_update"),
        })
    return rows

def american_profit(odds):
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)

def fair_american(p):
    if not (0 < p < 1):
        return None
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p

def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def model_output(features, total, over_odds, under_odds):
    correction = float(COEF["Intercept"])
    for f in FEATURES:
        correction += float(COEF[f]) * float(features[f])
    model_total = total + correction
    sd = float(COEF["Residual_SD"])
    p_over = normal_cdf(correction / sd)
    p_under = 1 - p_over
    over_ev = p_over * american_profit(over_odds) - (1 - p_over)
    under_ev = p_under * american_profit(under_odds) - (1 - p_under)

    v1_signal = "PASS"
    if abs(correction) >= 2:
        if correction > 0 and p_over >= 0.54 and over_ev >= 0.025:
            v1_signal = "OVER"
        elif correction < 0 and p_under >= 0.54 and under_ev >= 0.025:
            v1_signal = "UNDER"

    return {
        "correction": correction,
        "model_total": model_total,
        "p_over": p_over,
        "p_under": p_under,
        "fair_over": fair_american(p_over),
        "fair_under": fair_american(p_under),
        "over_ev": over_ev,
        "under_ev": under_ev,
        "signal": v1_signal,
    }

def fmt_odds(x):
    if x is None:
        return "—"
    x = round(float(x))
    return f"+{x}" if x > 0 else str(x)

# -----------------------------
# UI
# -----------------------------
st.title("🏈 NFL Totals Lab — Auto Data")
st.caption("Schedule + team metrics + weather load automatically. Live sportsbook odds can load automatically with The Odds API.")


with st.sidebar:
    st.header("Live data settings")
    secret_key = ""
    try:
        secret_key = str(st.secrets.get("THE_ODDS_API_KEY", "") or "")
    except Exception:
        secret_key = ""

    entered_key = st.text_input(
        "The Odds API key",
        value="",
        type="password",
        help="Optional. Put the key in Streamlit Secrets for a deployed app, or paste it here for this session.",
    )
    api_key = entered_key.strip() or secret_key.strip()

    strict_mode = st.checkbox(
        "Strict V1 ratings",
        value=False,
        help="Strict mode requires at least four prior games in the CURRENT season. Turn it off to use an experimental prior-season carryover early in the year.",
    )
    st.caption("Odds key is never written into the app files.")

try:
    with st.spinner("Loading NFL schedule…"):
        schedule = load_schedule()
except Exception as exc:
    st.exception(exc)
    st.stop()

upcoming = upcoming_schedule(schedule, days_ahead=12)
if upcoming.empty:
    st.warning("No regular-season games were found in the next 12 days.")
    st.stop()

# Friendly game labels.
game_options = []
for idx, r in upcoming.iterrows():
    label = f"{pd.Timestamp(r['gameday']).strftime('%a %b %d')} — {r['away_team']} @ {r['home_team']} — Week {int(r['week'])}"
    game_options.append((idx, label))

chosen_label = st.selectbox("Choose game", [x[1] for x in game_options])
chosen_idx = next(x[0] for x in game_options if x[1] == chosen_label)
game = upcoming.loc[chosen_idx]

away = str(game["away_team"])
home = str(game["home_team"])
week = int(game["week"])
season = int(game["season"])
kickoff_utc = schedule_kickoff_utc(game)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Away", away)
c2.metric("Home", home)
c3.metric("Week", week)
c4.metric("Kickoff (ET)", kickoff_utc.astimezone(ZoneInfo("America/New_York")).strftime("%a %-I:%M %p"))

# -----------------------------
# Odds
# -----------------------------
odds_events = []
event = None
book_rows = []
odds_error = None
if api_key:
    try:
        with st.spinner("Refreshing sportsbook odds…"):
            odds_events = fetch_odds(api_key)
        event = match_odds_event(odds_events, away, home)
        book_rows = book_market_rows(event)
    except Exception as exc:
        odds_error = str(exc)

if odds_error:
    st.warning(f"Live odds unavailable: {odds_error}")

st.subheader("Market")
market_source = "Manual / nflverse snapshot"
selected_book_row = None

if book_rows:
    books_df = pd.DataFrame(book_rows)
    default_index = 0
    dk = books_df.index[books_df["key"].eq("draftkings")].tolist()
    if dk:
        default_index = dk[0]
    book_name = st.selectbox("Sportsbook", books_df["book"].tolist(), index=default_index)
    selected_book_row = books_df[books_df["book"] == book_name].iloc[0].to_dict()
    market_source = book_name

    with st.expander("Compare available sportsbook totals"):
        show = books_df[["book", "total", "over", "under", "home_spread", "updated"]].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)

fallback_total = to_float(game.get("total_line"), 44.5)
fallback_over = to_float(game.get("over_odds"), -110)
fallback_under = to_float(game.get("under_odds"), -110)
fallback_spread = to_float(game.get("spread_line"), 0.0)

if selected_book_row:
    default_total = selected_book_row["total"] or fallback_total
    default_over = selected_book_row["over"] or fallback_over
    default_under = selected_book_row["under"] or fallback_under
    default_spread = selected_book_row["home_spread"] if selected_book_row["home_spread"] is not None else fallback_spread
else:
    default_total, default_over, default_under, default_spread = fallback_total, fallback_over, fallback_under, fallback_spread

m1, m2, m3, m4 = st.columns(4)
market_total = m1.number_input("Total", value=float(default_total), step=0.5)
over_odds = m2.number_input("Over odds", value=int(round(default_over)), step=1)
under_odds = m3.number_input("Under odds", value=int(round(default_under)), step=1)
home_spread = m4.number_input("Home spread", value=float(default_spread), step=0.5)
st.caption(f"Market source: **{market_source}**. You can always overwrite the line or price before calculating.")

# -----------------------------
# Weather + roof
# -----------------------------
st.subheader("Weather")
weather = None
weather_error = None
weather_loc = WEATHER_COORDS.get(home)

try:
    if weather_loc:
        lat, lon, venue_name = weather_loc
        with st.spinner("Loading stadium-area forecast…"):
            weather = fetch_weather(lat, lon, venue_name, kickoff_utc.isoformat())
except Exception as exc:
    weather_error = str(exc)

roof_type = "fixed" if home in FIXED_ROOF else "retractable" if home in RETRACTABLE_ROOF else "outdoor"
schedule_roof = str(game.get("roof") or "").lower()

if roof_type == "fixed":
    dome = 1.0
    roof_label = "Fixed roof / covered"
elif roof_type == "outdoor":
    dome = 0.0
    roof_label = "Outdoor"
else:
    if schedule_roof in {"closed", "dome"}:
        default_roof = "Closed"
    elif schedule_roof == "open":
        default_roof = "Open"
    else:
        default_roof = "Unknown / assume open"
    roof_choice = st.selectbox(
        "Retractable roof status",
        ["Unknown / assume open", "Open", "Closed"],
        index=["Unknown / assume open", "Open", "Closed"].index(default_roof),
        help="Retractable-roof decisions may not be known until game day. Confirm this before relying on the weather input.",
    )
    dome = 1.0 if roof_choice == "Closed" else 0.0
    roof_label = f"Retractable — {roof_choice}"

if weather:
    outside_temp = weather["temperature"]
    outside_wind = weather["wind"]
    outside_gust = weather["gust"]
    precip = weather["precip"]
    if dome == 1:
        model_temp = 70.0
        model_wind = 0.0
    else:
        model_temp = outside_temp
        model_wind = outside_wind

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Roof", roof_label)
    w2.metric("Kickoff temp", f"{outside_temp:.0f}°F")
    w3.metric("Wind", f"{outside_wind:.0f} mph")
    w4.metric("Rain chance", f"{precip:.0f}%")
    st.caption(
        f"Forecast location: {weather['resolved_name']} · local kickoff "
        f"{weather['kickoff_local'].strftime('%a %-I:%M %p')} · gusts {outside_gust:.0f} mph. "
        + ("Indoor model inputs use 70°F / 0 mph." if dome == 1 else "Outdoor model inputs use the forecast values.")
    )
else:
    if weather_error:
        st.warning(f"Weather could not load: {weather_error}")
    model_temp = 70.0
    model_wind = 0.0 if dome == 1 else st.number_input("Manual wind mph", value=5.0)
    if dome == 0:
        model_temp = st.number_input("Manual temperature °F", value=70.0)

cold_units = 0.0 if dome == 1 else max(0.0, (40.0 - model_temp) / 10.0)

# -----------------------------
# Team ratings
# -----------------------------
st.subheader("Automatic team ratings")

try:
    with st.spinner(f"Loading {season} nflverse play-by-play…"):
        cur_pbp = load_pbp(season)
    with st.spinner(f"Loading {season-1} prior-season play-by-play…"):
        prev_pbp = load_pbp(season - 1)

    cur_metrics = aggregate_game_metrics(cur_pbp)
    prev_metrics = aggregate_game_metrics(prev_pbp)

    cur_states = build_histories(cur_metrics, max_week=week)
    prev_states = build_histories(prev_metrics, max_week=None)

    football_features, rating_meta = matchup_features(
        away, home, prev_states, cur_states, strict=strict_mode
    )
except Exception as exc:
    st.error(f"Could not build nflverse team ratings: {exc}")
    football_features = None
    rating_meta = {"away_games": 0, "home_games": 0, "used_prior": False}

r1, r2, r3 = st.columns(3)
r1.metric(f"{away} current-season games used", rating_meta["away_games"])
r2.metric(f"{home} current-season games used", rating_meta["home_games"])
r3.metric("Prior-season carryover", "Yes" if rating_meta["used_prior"] else "No")

if rating_meta["used_prior"]:
    st.warning(
        "Early-season mode is using prior-season games because one or both teams have fewer than four "
        "current-season games. This makes the output **experimental** and is not the same setup that was backtested."
    )

if football_features is None:
    st.warning(
        "Not enough pregame team data under the selected rating mode. Turn off Strict V1 early in the season "
        "to see the experimental prior-season carryover."
    )
    st.stop()

with st.expander("See the advanced inputs the app calculated"):
    feature_table = pd.DataFrame(
        [{"Feature": k, "Value": v} for k, v in football_features.items()]
    )
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

# Rest/division inputs from schedule.
away_rest = to_float(game.get("away_rest"), 7.0)
home_rest = to_float(game.get("home_rest"), 7.0)
short_rest = 1.0 if min(away_rest, home_rest) <= 5 else 0.0
div_raw = game.get("div_game")
try:
    div_game = 1.0 if float(div_raw) == 1 else 0.0
except Exception:
    div_game = 0.0

all_features = dict(football_features)
all_features.update(
    {
        "Wind": float(model_wind),
        "Cold_Units": float(cold_units),
        "Dome": float(dome),
        "Short_Rest": short_rest,
        "Div_Game": div_game,
    }
)

# -----------------------------
# Model result
# -----------------------------
st.subheader("Model output")
result = model_output(
    all_features,
    float(market_total),
    float(over_odds),
    float(under_odds),
)

a, b, c, d = st.columns(4)
a.metric("Sportsbook total", f"{market_total:.1f}")
b.metric("V1 model total", f"{result['model_total']:.1f}")
c.metric("Model edge", f"{result['correction']:+.2f}")
d.metric("V1 signal", result["signal"])

a, b, c, d = st.columns(4)
a.metric("Over probability", f"{result['p_over']:.1%}")
b.metric("Under probability", f"{result['p_under']:.1%}")
c.metric("Fair Over", fmt_odds(result["fair_over"]))
d.metric("Fair Under", fmt_odds(result["fair_under"]))

a, b, c = st.columns(3)
a.metric("Over EV", f"{result['over_ev']:+.1%}")
b.metric("Under EV", f"{result['under_ev']:+.1%}")
c.metric("Weather input", f"{model_temp:.0f}°F / {model_wind:.0f} mph")

if result["signal"] != "PASS":
    st.warning(
        f"V1 would flag **{result['signal']}**, but this is **not a validated live-bet recommendation**. "
        "V1 lost 18.2% ROI in the untouched 2025 holdout."
    )
else:
    st.info("V1 filter says PASS.")

st.divider()
with st.expander("Data sources & refresh behavior"):
    st.markdown(
        """
- **Schedule / game metadata:** nflverse `games.csv`
- **Team EPA / success / explosive / play volume / turnovers / sacks:** calculated from nflverse play-by-play
- **Weather:** Open-Meteo stadium-area forecast
- **Sportsbook totals / prices:** The Odds API when a key is configured
- **Fallback line:** nflverse schedule snapshot or manual override
- Schedule refresh cache: ~30 minutes
- Weather refresh cache: ~30 minutes
- Odds refresh cache: ~2 minutes
- Play-by-play refresh cache: ~1 hour
        """
    )

with st.expander("Model status"):
    st.caption(
        f"V1 historical holdout: {int(HOLDOUT['Wins'])}-{int(HOLDOUT['Losses'])}, "
        f"{float(HOLDOUT['Win_Pct']):.1%} wins, {float(HOLDOUT['ROI']):.1%} ROI."
    )


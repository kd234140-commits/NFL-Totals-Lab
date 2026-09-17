from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

PROP_MARKET_LABELS = {
    "player_pass_yds": "Passing yards",
    "player_pass_tds": "Passing TDs",
    "player_pass_completions": "Pass completions",
    "player_pass_attempts": "Pass attempts",
    "player_pass_interceptions": "Pass interceptions",
    "player_pass_rush_yds": "Pass + rush yards",
    "player_rush_yds": "Rushing yards",
    "player_rush_attempts": "Rush attempts",
    "player_receptions": "Receptions",
    "player_reception_yds": "Receiving yards",
    "player_rush_reception_yds": "Rush + receiving yards",
    "player_tds": "Touchdowns",
    "player_anytime_td": "Anytime TD",
}

CORE_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_completions",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_rush_attempts",
    "player_receptions",
    "player_reception_yds",
]


def american_to_decimal(odds):
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if not math.isfinite(o) or o == 0:
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o))


def implied_prob(odds):
    d = american_to_decimal(odds)
    return None if d is None or d <= 1 else 1.0 / d


def fair_american(p):
    if p is None or not (0 < float(p) < 1):
        return None
    p = float(p)
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def ev_from_prob(prob, odds):
    d = american_to_decimal(odds)
    if prob is None or d is None:
        return None
    return float(prob) * d - 1.0


def devig_two_way(odds_a, odds_b):
    pa, pb = implied_prob(odds_a), implied_prob(odds_b)
    if pa is None or pb is None or pa + pb <= 0:
        return None, None
    z = pa + pb
    return pa / z, pb / z


def best_offer(rows):
    """Return row with highest decimal price."""
    best = None
    best_d = -1.0
    for r in rows:
        d = american_to_decimal(r.get("Odds"))
        if d is not None and d > best_d:
            best_d, best = d, r
    return best


def complement_side(side: str):
    s = str(side).strip().lower()
    if s == "over":
        return "Under"
    if s == "under":
        return "Over"
    if s == "yes":
        return "No"
    if s == "no":
        return "Yes"
    return None


def parse_prop_event(payload: dict) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    matchup = f"{payload.get('away_team', '')} @ {payload.get('home_team', '')}".strip()
    rows = []
    for b in payload.get("bookmakers", []) or []:
        bkey = b.get("key")
        bname = b.get("title") or bkey
        for m in b.get("markets", []) or []:
            mkey = m.get("key")
            mlabel = PROP_MARKET_LABELS.get(mkey, mkey)
            updated = m.get("last_update") or b.get("last_update")
            for o in m.get("outcomes", []) or []:
                player = o.get("description")
                side = o.get("name")
                odds = o.get("price")
                point = o.get("point")
                if not player or side is None or odds is None:
                    continue
                try:
                    odds = float(odds)
                except Exception:
                    continue
                try:
                    point = float(point) if point is not None else None
                except Exception:
                    point = None
                rows.append({
                    "Event ID": payload.get("id"),
                    "Matchup": matchup,
                    "Commence": payload.get("commence_time"),
                    "Market Key": mkey,
                    "Market": mlabel,
                    "Player": str(player),
                    "Side": str(side),
                    "Line": point,
                    "Odds": odds,
                    "Book Key": bkey,
                    "Book": bname,
                    "Updated": updated,
                })
    return pd.DataFrame(rows)


def combine_prop_payloads(payloads) -> pd.DataFrame:
    frames = [parse_prop_event(x) for x in (payloads or [])]
    frames = [x for x in frames if not x.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prop_value_table(payloads, minimum_other_books: int = 1) -> pd.DataFrame:
    """
    Consensus EV at the EXACT same player/market/line.
    Best offered price is evaluated against de-vigged two-way probabilities from OTHER books.
    At least one other full two-way book is required, which avoids self-confirming EV.
    """
    offers = combine_prop_payloads(payloads)
    if offers.empty:
        return pd.DataFrame()

    key_cols = ["Event ID", "Matchup", "Market Key", "Market", "Player", "Line"]
    out = []
    for _, g in offers.groupby(key_cols, dropna=False):
        # Only markets with a recognized complementary side can be fairly de-vigged.
        for side in sorted(g["Side"].dropna().unique()):
            comp = complement_side(side)
            if comp is None:
                continue
            side_rows = g[g["Side"].str.lower().eq(str(side).lower())].to_dict("records")
            if not side_rows:
                continue
            candidate = best_offer(side_rows)
            if candidate is None:
                continue

            fair_probs = []
            comparison_books = []
            for bkey, bg in g.groupby("Book Key", dropna=False):
                if bkey == candidate.get("Book Key"):
                    continue
                a = bg[bg["Side"].str.lower().eq(str(side).lower())]
                b = bg[bg["Side"].str.lower().eq(comp.lower())]
                if a.empty or b.empty:
                    continue
                # Same exact line is already guaranteed by the outer group.
                oa = best_offer(a.to_dict("records"))
                ob = best_offer(b.to_dict("records"))
                if oa is None or ob is None:
                    continue
                pa, _ = devig_two_way(oa["Odds"], ob["Odds"])
                if pa is not None:
                    fair_probs.append(pa)
                    comparison_books.append(str(oa.get("Book")))

            if len(fair_probs) < max(1, int(minimum_other_books)):
                continue

            p = sum(fair_probs) / len(fair_probs)
            ev = ev_from_prob(p, candidate["Odds"])
            implied = implied_prob(candidate["Odds"])
            row = dict(candidate)
            row.update({
                "Fair Prob": p,
                "Fair Odds": fair_american(p),
                "Implied Prob": implied,
                "EV": ev,
                "Probability Edge": p - implied if implied is not None else None,
                "Consensus Books": len(fair_probs),
                "Consensus Sources": ", ".join(sorted(set(comparison_books))),
            })
            out.append(row)

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    # Keep one row per exact selection (the best offered price).
    sel = ["Event ID", "Market Key", "Player", "Line", "Side"]
    df["_decimal"] = df["Odds"].map(american_to_decimal)
    df = df.sort_values(["EV", "Consensus Books", "_decimal"], ascending=[False, False, False])
    df = df.drop_duplicates(sel, keep="first").drop(columns=["_decimal"])
    return df.reset_index(drop=True)


def _distinct_best_pair(a_rows, b_rows):
    """Best two-price combination, preferring different books."""
    combos = []
    for a in a_rows:
        da = american_to_decimal(a.get("Odds"))
        if da is None:
            continue
        for b in b_rows:
            db = american_to_decimal(b.get("Odds"))
            if db is None:
                continue
            if a.get("Book Key") == b.get("Book Key"):
                continue
            s = 1.0 / da + 1.0 / db
            combos.append((s, a, b))
    if not combos:
        # Same-book negative-hold is extremely unusual, but keep it visible if it exists.
        for a in a_rows:
            da = american_to_decimal(a.get("Odds"))
            if da is None:
                continue
            for b in b_rows:
                db = american_to_decimal(b.get("Odds"))
                if db is None:
                    continue
                s = 1.0 / da + 1.0 / db
                combos.append((s, a, b))
    return min(combos, key=lambda x: x[0]) if combos else None


def arb_math(odds_a, odds_b, total_stake=100.0):
    da, db = american_to_decimal(odds_a), american_to_decimal(odds_b)
    if da is None or db is None:
        return None
    inv = 1.0 / da + 1.0 / db
    if inv <= 0:
        return None
    stake_a = float(total_stake) * (1.0 / da) / inv
    stake_b = float(total_stake) * (1.0 / db) / inv
    payout = float(total_stake) / inv
    profit = payout - float(total_stake)
    return {
        "Implied Sum": inv,
        "ROI": 1.0 / inv - 1.0,
        "Stake A": stake_a,
        "Stake B": stake_b,
        "Guaranteed Payout": payout,
        "Guaranteed Profit": profit,
    }


def prop_arbitrage_table(payloads, total_stake=100.0, min_roi=0.0) -> pd.DataFrame:
    offers = combine_prop_payloads(payloads)
    if offers.empty:
        return pd.DataFrame()
    key_cols = ["Event ID", "Matchup", "Market Key", "Market", "Player", "Line"]
    rows = []
    for _, g in offers.groupby(key_cols, dropna=False):
        done = set()
        for side in g["Side"].dropna().unique():
            comp = complement_side(side)
            if comp is None:
                continue
            pair_key = tuple(sorted([side.lower(), comp.lower()]))
            if pair_key in done:
                continue
            done.add(pair_key)
            a = g[g["Side"].str.lower().eq(str(side).lower())].to_dict("records")
            b = g[g["Side"].str.lower().eq(comp.lower())].to_dict("records")
            pair = _distinct_best_pair(a, b)
            if pair is None:
                continue
            inv, ra, rb = pair
            if inv >= 1.0:
                continue
            calc = arb_math(ra["Odds"], rb["Odds"], total_stake)
            if calc is None or calc["ROI"] < float(min_roi):
                continue
            rows.append({
                "Matchup": ra["Matchup"], "Market": ra["Market"], "Player": ra["Player"], "Line": ra["Line"],
                "Side A": ra["Side"], "Odds A": ra["Odds"], "Book A": ra["Book"],
                "Side B": rb["Side"], "Odds B": rb["Odds"], "Book B": rb["Book"],
                **calc,
            })
    return pd.DataFrame(rows).sort_values("ROI", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def _core_book_rows(event):
    rows = []
    if not event:
        return rows
    home, away = event.get("home_team"), event.get("away_team")
    matchup = f"{away} @ {home}"
    for b in event.get("bookmakers", []) or []:
        bk, bn = b.get("key"), b.get("title") or b.get("key")
        for m in b.get("markets", []) or []:
            key = m.get("key")
            if key not in {"h2h", "spreads", "totals"}:
                continue
            for o in m.get("outcomes", []) or []:
                try:
                    price = float(o.get("price"))
                except Exception:
                    continue
                try:
                    point = float(o.get("point")) if o.get("point") is not None else None
                except Exception:
                    point = None
                rows.append({
                    "Event ID": event.get("id"), "Matchup": matchup, "Home": home, "Away": away,
                    "Market Key": key, "Side": o.get("name"), "Line": point, "Odds": price,
                    "Book Key": bk, "Book": bn, "Updated": m.get("last_update") or b.get("last_update"),
                })
    return rows


def core_best_lines(odds_events) -> pd.DataFrame:
    rows = []
    for event in odds_events or []:
        offers = pd.DataFrame(_core_book_rows(event))
        if offers.empty:
            continue
        home, away = event.get("home_team"), event.get("away_team")
        matchup = f"{away} @ {home}"

        # ML: price is all that matters.
        for team in [away, home]:
            x = offers[(offers["Market Key"] == "h2h") & (offers["Side"] == team)]
            if not x.empty:
                rr = best_offer(x.to_dict("records"))
                rows.append({"Matchup": matchup, "Market": "Moneyline", "Selection": team, "Line": None, "Odds": rr["Odds"], "Book": rr["Book"]})

        # Spread: most favorable point first, then best price at that point.
        for team in [away, home]:
            x = offers[(offers["Market Key"] == "spreads") & (offers["Side"] == team) & offers["Line"].notna()]
            if not x.empty:
                best_line = float(x["Line"].max())
                rr = best_offer(x[x["Line"] == best_line].to_dict("records"))
                rows.append({"Matchup": matchup, "Market": "Spread", "Selection": team, "Line": best_line, "Odds": rr["Odds"], "Book": rr["Book"]})

        # Totals: Over wants lowest number; Under wants highest number.
        for side, fn in [("Over", "min"), ("Under", "max")]:
            x = offers[(offers["Market Key"] == "totals") & (offers["Side"].str.lower() == side.lower()) & offers["Line"].notna()]
            if not x.empty:
                line = float(x["Line"].min() if fn == "min" else x["Line"].max())
                rr = best_offer(x[x["Line"] == line].to_dict("records"))
                rows.append({"Matchup": matchup, "Market": "Total", "Selection": side, "Line": line, "Odds": rr["Odds"], "Book": rr["Book"]})
    return pd.DataFrame(rows)


def core_arbitrage_table(odds_events, total_stake=100.0, min_roi=0.0) -> pd.DataFrame:
    arbs = []
    for event in odds_events or []:
        rows = _core_book_rows(event)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        home, away = event.get("home_team"), event.get("away_team")
        matchup = f"{away} @ {home}"

        # Moneyline
        a = df[(df["Market Key"] == "h2h") & (df["Side"] == home)].to_dict("records")
        b = df[(df["Market Key"] == "h2h") & (df["Side"] == away)].to_dict("records")
        pair = _distinct_best_pair(a, b)
        if pair and pair[0] < 1.0:
            calc = arb_math(pair[1]["Odds"], pair[2]["Odds"], total_stake)
            if calc and calc["ROI"] >= float(min_roi):
                arbs.append({"Matchup": matchup, "Market": "Moneyline", "Line": None,
                             "Side A": home, "Odds A": pair[1]["Odds"], "Book A": pair[1]["Book"],
                             "Side B": away, "Odds B": pair[2]["Odds"], "Book B": pair[2]["Book"], **calc})

        # Exact-line spreads
        home_lines = sorted(set(df[(df["Market Key"] == "spreads") & (df["Side"] == home)]["Line"].dropna()))
        for hl in home_lines:
            a = df[(df["Market Key"] == "spreads") & (df["Side"] == home) & (df["Line"] == hl)].to_dict("records")
            b = df[(df["Market Key"] == "spreads") & (df["Side"] == away) & (df["Line"] == -hl)].to_dict("records")
            pair = _distinct_best_pair(a, b)
            if pair and pair[0] < 1.0:
                calc = arb_math(pair[1]["Odds"], pair[2]["Odds"], total_stake)
                if calc and calc["ROI"] >= float(min_roi):
                    arbs.append({"Matchup": matchup, "Market": "Spread", "Line": float(hl),
                                 "Side A": f"{home} {hl:+.1f}", "Odds A": pair[1]["Odds"], "Book A": pair[1]["Book"],
                                 "Side B": f"{away} {-hl:+.1f}", "Odds B": pair[2]["Odds"], "Book B": pair[2]["Book"], **calc})

        # Exact-line totals
        totals = sorted(set(df[df["Market Key"] == "totals"]["Line"].dropna()))
        for line in totals:
            a = df[(df["Market Key"] == "totals") & (df["Side"].str.lower() == "over") & (df["Line"] == line)].to_dict("records")
            b = df[(df["Market Key"] == "totals") & (df["Side"].str.lower() == "under") & (df["Line"] == line)].to_dict("records")
            pair = _distinct_best_pair(a, b)
            if pair and pair[0] < 1.0:
                calc = arb_math(pair[1]["Odds"], pair[2]["Odds"], total_stake)
                if calc and calc["ROI"] >= float(min_roi):
                    arbs.append({"Matchup": matchup, "Market": "Total", "Line": float(line),
                                 "Side A": f"Over {line:.1f}", "Odds A": pair[1]["Odds"], "Book A": pair[1]["Book"],
                                 "Side B": f"Under {line:.1f}", "Odds B": pair[2]["Odds"], "Book B": pair[2]["Book"], **calc})

    return pd.DataFrame(arbs).sort_values("ROI", ascending=False).reset_index(drop=True) if arbs else pd.DataFrame()


def core_middle_table(odds_events) -> pd.DataFrame:
    rows = []
    for event in odds_events or []:
        df = pd.DataFrame(_core_book_rows(event))
        if df.empty:
            continue
        home, away = event.get("home_team"), event.get("away_team")
        matchup = f"{away} @ {home}"

        # Totals middle: Over lowest available total + Under highest available total.
        ov = df[(df["Market Key"] == "totals") & (df["Side"].str.lower() == "over") & df["Line"].notna()]
        un = df[(df["Market Key"] == "totals") & (df["Side"].str.lower() == "under") & df["Line"].notna()]
        if not ov.empty and not un.empty:
            lo, hi = float(ov["Line"].min()), float(un["Line"].max())
            if lo < hi:
                a = best_offer(ov[ov["Line"] == lo].to_dict("records"))
                b = best_offer(un[un["Line"] == hi].to_dict("records"))
                if a and b and a["Book Key"] != b["Book Key"]:
                    rows.append({"Matchup": matchup, "Type": "Total middle", "Window": f"{lo:.1f} to {hi:.1f}",
                                 "Bet A": f"Over {lo:.1f} ({int(a['Odds']):+d})", "Book A": a["Book"],
                                 "Bet B": f"Under {hi:.1f} ({int(b['Odds']):+d})", "Book B": b["Book"]})

        # Spread middle: highest home point + highest away point; overlap exists if their sum > 0.
        hs = df[(df["Market Key"] == "spreads") & (df["Side"] == home) & df["Line"].notna()]
        aws = df[(df["Market Key"] == "spreads") & (df["Side"] == away) & df["Line"].notna()]
        if not hs.empty and not aws.empty:
            hline, aline = float(hs["Line"].max()), float(aws["Line"].max())
            if hline + aline > 0:
                a = best_offer(hs[hs["Line"] == hline].to_dict("records"))
                b = best_offer(aws[aws["Line"] == aline].to_dict("records"))
                if a and b and a["Book Key"] != b["Book Key"]:
                    rows.append({"Matchup": matchup, "Type": "Spread middle", "Window": f"{hline + aline:.1f} pts overlap",
                                 "Bet A": f"{home} {hline:+.1f} ({int(a['Odds']):+d})", "Book A": a["Book"],
                                 "Bet B": f"{away} {aline:+.1f} ({int(b['Odds']):+d})", "Book B": b["Book"]})
    return pd.DataFrame(rows)

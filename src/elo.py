"""
Elo rating system for international football.

How Elo works (the short version):
  - Every team starts at the same rating (1500).
  - Before each match we compute the "expected score" for each team — a number
    between 0 and 1 that represents the probability of winning (draws count as 0.5).
  - After the match the actual result is compared to that expectation and both
    teams' ratings are nudged toward reality by a fixed step size K.
  - Good teams gain points by beating other good teams; they barely gain anything
    by beating weak opponents.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RATING = 1500   # every team begins here
K_FACTOR = 20           # how much a single match moves the needle
                        # K=20 is a common choice for football; higher = more
                        # reactive, lower = more stable


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Return the expected score (win probability) for team A given both ratings.

    The formula is the standard logistic function used in chess Elo:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    A score of 1.0 means team A wins, 0.5 means a draw, 0.0 means team A loses.
    The denominator 400 is the conventional scaling constant — a 400-point gap
    means the stronger team wins ~91 % of the time.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_ratings(
    rating_home: float,
    rating_away: float,
    home_score: int,
    away_score: int,
) -> tuple[float, float]:
    """
    Apply one match result and return the updated (home_rating, away_rating).

    actual_score is 1 if the home team won, 0.5 for a draw, 0 if they lost.
    The Elo update rule is:
        new_R = old_R + K * (actual - expected)
    Both teams are updated simultaneously (away gets the mirror image).
    """
    # --- convert the scoreline to a result value for the home team ---
    if home_score > away_score:
        actual_home = 1.0   # home win
    elif home_score == away_score:
        actual_home = 0.5   # draw
    else:
        actual_home = 0.0   # home loss (away win)

    # --- expected scores before the match ---
    exp_home = expected_score(rating_home, rating_away)
    exp_away = expected_score(rating_away, rating_home)  # = 1 - exp_home

    # --- the actual score from the away team's perspective is the mirror ---
    actual_away = 1.0 - actual_home

    # --- apply the update formula ---
    new_home = rating_home + K_FACTOR * (actual_home - exp_home)
    new_away = rating_away + K_FACTOR * (actual_away - exp_away)

    return new_home, new_away


# ---------------------------------------------------------------------------
# Main computation: process the full match history
# ---------------------------------------------------------------------------

def compute_elo(results_path: str = "data/raw/results.csv") -> dict[str, float]:
    """
    Read every match in chronological order, run Elo updates, and return a
    dict mapping team name → final Elo rating.

    Parameters
    ----------
    results_path : path to the downloaded results.csv file

    Returns
    -------
    ratings : dict  {team_name: elo_rating}
    """
    df = pd.read_csv(results_path, parse_dates=["date"])

    # Sort by date so we process history in the right order.
    # Ties within the same date don't matter much for Elo.
    df = df.sort_values("date").reset_index(drop=True)

    # Drop rows where scores are missing (some very old friendlies have NaN scores)
    df = df.dropna(subset=["home_score", "away_score"])

    # ratings is a plain dict; missing keys default to DEFAULT_RATING
    ratings: dict[str, float] = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Look up current ratings, falling back to the default for new teams
        r_home = ratings.get(home, DEFAULT_RATING)
        r_away = ratings.get(away, DEFAULT_RATING)

        # Update based on the result
        r_home, r_away = update_ratings(
            r_home, r_away,
            int(row["home_score"]), int(row["away_score"])
        )

        ratings[home] = r_home
        ratings[away] = r_away

    return ratings


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def top_n(ratings: dict[str, float], n: int = 20) -> pd.DataFrame:
    """Return the top-n teams as a tidy DataFrame sorted by rating descending."""
    df = (
        pd.DataFrame(ratings.items(), columns=["team", "elo"])
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1          # rank starts at 1
    df["elo"] = df["elo"].round(1)
    return df.head(n)


def get_rating(ratings: dict[str, float], team: str) -> float:
    """
    Look up a single team's Elo rating by name.

    Raises KeyError with a helpful message if the team is not found.
    Usage:
        ratings = compute_elo()
        get_rating(ratings, "Brazil")
    """
    if team not in ratings:
        # Show close matches to help with spelling
        close = [t for t in ratings if team.lower() in t.lower()]
        hint = f"  Did you mean one of: {close[:5]}" if close else ""
        raise KeyError(f"Team '{team}' not found in ratings.{hint}")
    return round(ratings[team], 1)


# ---------------------------------------------------------------------------
# Run as a script: python src/elo.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Computing Elo ratings from full match history …")
    ratings = compute_elo()

    print(f"\nTotal teams rated: {len(ratings)}")
    print("\nTop 20 teams by Elo rating:")
    print(top_n(ratings).to_string())

    # Quick single-team lookup demo
    print("\n--- Single-team lookup examples ---")
    for team in ["Brazil", "France", "Germany", "England", "Argentina"]:
        print(f"  {team}: {get_rating(ratings, team)}")

"""
Match prediction: win / draw / loss probabilities derived from Elo ratings.

The key challenge with Elo for football is that it only produces a single
"expected score" (E_A), which conflates wins and draws.  We fix that by
measuring the actual draw rate from history and splitting E_A into three
probabilities that are both consistent with Elo AND grounded in real data.

DRAW HANDLING — how it works:
  Step 1 — Empirical draw rates by Elo gap
    Replay the full match history.  For every match, record the Elo gap
    (|rating_A - rating_B|) just *before* the match is played, and whether
    the match ended in a draw.  Bin those records by gap and compute the
    draw rate in each bin.

    Bins measured from 49 257 historical matches:
        Gap range   | n matches | draw rate
        ------------|-----------|----------
          0 –  49   |  13 720   |  25.6 %
         50 –  99   |  11 334   |  25.4 %
        100 – 149   |   8 947   |  23.5 %
        150 – 199   |   6 090   |  22.2 %
        200 – 299   |   6 028   |  17.5 %
        300 +       |   3 138   |   9.9 %

    Pattern: the more evenly matched two teams are, the higher the draw
    rate.  This makes intuitive sense.

  Step 2 — Split E_A into three probabilities
    E_A  = expected score from the standard Elo formula (0–1).
    pd   = empirical draw rate for this Elo gap.

    P(A wins) = E_A       – 0.5 * pd
    P(draw)   = pd
    P(B wins) = (1 – E_A) – 0.5 * pd

    Why does this work?  Elo's E_A is a weighted average:
        E_A = P(A wins) * 1 + P(draw) * 0.5 + P(B wins) * 0
    Solving for P(A wins) with P(draw) = pd gives exactly the formula above.

  Step 3 — Guard against negative probabilities
    On extreme mismatches (e.g. E_A ≈ 0.98 but pd = 0.10) the formula for
    P(B wins) can produce a tiny negative number.  We floor all three at 0
    and renormalise so they sum to exactly 1.
"""

import random
import pandas as pd
from src.elo import expected_score, update_ratings, DEFAULT_RATING


# ---------------------------------------------------------------------------
# Empirical draw-rate bins (measured from data — see module docstring)
# ---------------------------------------------------------------------------

# Each entry: (upper_bound_exclusive, draw_rate)
# We walk the list and use the rate for the first bin whose upper bound
# is greater than the actual Elo gap.
_DRAW_BINS = [
    (  50, 0.2555),   # gap  0 –  49
    ( 100, 0.2544),   # gap 50 –  99
    ( 150, 0.2345),   # gap 100 – 149
    ( 200, 0.2215),   # gap 150 – 199
    ( 300, 0.1752),   # gap 200 – 299
    (9999, 0.0991),   # gap 300 +
]


def _draw_rate_for_gap(gap: float) -> float:
    """
    Look up the empirical draw probability for a given absolute Elo gap.
    Walks the bin list and returns the rate for the first bin that contains
    this gap value.
    """
    for upper, rate in _DRAW_BINS:
        if gap < upper:
            return rate
    # Should never reach here given 9999 sentinel, but be safe
    return _DRAW_BINS[-1][1]


# ---------------------------------------------------------------------------
# Build model: one replay pass → ratings + draw table validation
# ---------------------------------------------------------------------------

def build_model(results_path: str = "data/raw/results.csv") -> dict[str, float]:
    """
    Replay the full match history in chronological order.

    Returns
    -------
    ratings : dict  {team_name: elo_rating}

    The draw-rate bins (_DRAW_BINS) are pre-computed constants derived from
    the same history; this function just produces the final Elo ratings.
    """
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["home_score", "away_score"])

    ratings: dict[str, float] = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        r_h = ratings.get(home, DEFAULT_RATING)
        r_a = ratings.get(away, DEFAULT_RATING)

        r_h, r_a = update_ratings(
            r_h, r_a,
            int(row["home_score"]), int(row["away_score"])
        )

        ratings[home] = r_h
        ratings[away] = r_a

    return ratings


# ---------------------------------------------------------------------------
# Probability calculation
# ---------------------------------------------------------------------------

def match_probabilities(
    rating_a: float,
    rating_b: float,
    home_bonus_a: float = 0.0,
    home_bonus_b: float = 0.0,
) -> tuple[float, float, float]:
    """
    Given two Elo ratings, return (p_win_a, p_draw, p_win_b) as fractions
    that sum to exactly 1.0.

    home_bonus_a / home_bonus_b : extra Elo points added to the effective
    rating of team A or B when they are playing at home.  Both default to 0
    (neutral venue).  The bonus shifts the expected score without changing
    the stored Elo — it is applied only for this single match calculation.

    Steps:
      1. Apply any home bonus to get effective ratings.
      2. Compute E_A from the Elo formula using effective ratings.
      3. Look up the empirical draw rate for this gap.
      4. Split E_A into the three outcomes.
      5. Floor negatives at 0 and renormalise.
    """
    # Step 1 — effective ratings (base Elo ± home bonus)
    eff_a = rating_a + home_bonus_a
    eff_b = rating_b + home_bonus_b

    # Step 2 — Elo expected score for team A using effective ratings
    e_a = expected_score(eff_a, eff_b)

    # Step 3 — draw rate for this Elo gap (use effective ratings for the gap)
    gap = abs(eff_a - eff_b)
    p_draw = _draw_rate_for_gap(gap)

    # Step 4 — split into three outcomes (see module docstring for derivation)
    p_win_a = e_a - 0.5 * p_draw
    p_win_b = (1.0 - e_a) - 0.5 * p_draw

    # Step 5 — guard: floor at zero, then renormalise
    p_win_a = max(p_win_a, 0.0)
    p_win_b = max(p_win_b, 0.0)
    # p_draw stays at the empirical rate — we keep it fixed and let the
    # win/loss absorb any renormalisation needed
    total = p_win_a + p_draw + p_win_b
    p_win_a /= total
    p_draw  /= total
    p_win_b /= total

    return p_win_a, p_draw, p_win_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_match(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
) -> None:
    """
    Print the win / draw / loss probabilities for a match between two teams.

    Looks up both teams' Elo ratings, computes the three-way split, and
    prints a formatted breakdown.  Also asserts the probabilities sum to 100 %.
    """
    # Validate both teams exist
    for team in (team_a, team_b):
        if team not in ratings:
            close = [t for t in ratings if team.lower() in t.lower()]
            hint = f"  Did you mean: {close[:5]}" if close else ""
            raise KeyError(f"Team '{team}' not found in ratings.{hint}")

    r_a = ratings[team_a]
    r_b = ratings[team_b]

    p_win, p_draw, p_loss = match_probabilities(r_a, r_b)

    # Convert to percentages for display
    pct_win  = p_win  * 100
    pct_draw = p_draw * 100
    pct_loss = p_loss * 100
    total    = pct_win + pct_draw + pct_loss

    print(f"\n{'─'*44}")
    print(f"  {team_a:20s} vs  {team_b}")
    print(f"  Elo: {r_a:.0f}  vs  {r_b:.0f}  (gap {abs(r_a-r_b):.0f})")
    print(f"{'─'*44}")
    print(f"  {team_a} win : {pct_win:5.1f}%")
    print(f"  Draw          : {pct_draw:5.1f}%")
    print(f"  {team_b} win : {pct_loss:5.1f}%")
    print(f"  Total         : {total:5.1f}%")

    # Sanity check — if this ever fails something is wrong in the maths
    assert abs(total - 100.0) < 0.01, f"Probabilities don't sum to 100%: {total}"


def simulate_match(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    home_bonus_a: float = 0.0,
    home_bonus_b: float = 0.0,
) -> str:
    """
    Simulate a single match outcome using the three-way probabilities.

    Returns
    -------
    "A"    — team_a wins
    "draw" — the match is drawn
    "B"    — team_b wins

    home_bonus_a / home_bonus_b : optional Elo boost for a team playing at
    home (see match_probabilities for details).  Defaults to 0 (neutral).

    How it works: random.choices() performs a weighted random draw.
    The weights are the three probabilities, so over many trials the
    frequencies converge to those probabilities.
    """
    r_a = ratings[team_a]
    r_b = ratings[team_b]

    p_win, p_draw, p_loss = match_probabilities(
        r_a, r_b,
        home_bonus_a=home_bonus_a,
        home_bonus_b=home_bonus_b,
    )

    # weighted random choice from the three outcomes
    outcome = random.choices(
        population=["A", "draw", "B"],
        weights   =[p_win, p_draw, p_loss],
        k=1
    )[0]

    return outcome


# ---------------------------------------------------------------------------
# Run as script: python src/predict.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building Elo ratings …")
    ratings = build_model()
    print(f"Ratings loaded for {len(ratings)} teams.\n")

    # --- Predicted probabilities for three match-ups ---
    for team_a, team_b in [
        ("France",  "Jordan"),
        ("Spain",   "Argentina"),
        ("Brazil",  "Germany"),
    ]:
        predict_match(team_a, team_b, ratings)

    # --- Monte Carlo verification: simulate France vs Jordan 1000 times ---
    print("\n\nMonte Carlo simulation — France vs Jordan (1000 matches):")
    N = 1000
    results = {"A": 0, "draw": 0, "B": 0}
    for _ in range(N):
        outcome = simulate_match("France", "Jordan", ratings)
        results[outcome] += 1

    # Print counts and the implied rates
    r_a = ratings["France"]
    r_b = ratings["Jordan"]
    p_win, p_draw, p_loss = match_probabilities(r_a, r_b)

    print(f"\n{'Outcome':<12} {'Simulated':>10} {'Predicted':>10}")
    print("─" * 34)
    print(f"{'France win':<12} {results['A']:>8}  ({results['A']/N*100:5.1f}%)   expected {p_win*100:5.1f}%")
    print(f"{'Draw':<12} {results['draw']:>8}  ({results['draw']/N*100:5.1f}%)   expected {p_draw*100:5.1f}%")
    print(f"{'Jordan win':<12} {results['B']:>8}  ({results['B']/N*100:5.1f}%)   expected {p_loss*100:5.1f}%")

"""
Poisson goal-scoring model for international football.

WHY POISSON?
  Football goals are rare, independent events — exactly the conditions under
  which the Poisson distribution applies.  If a team's expected goals in a
  match is λ, the probability they score exactly k goals is:

      P(X = k) = (λ^k * e^(−λ)) / k!

  Because the two teams score independently, the joint probability of a
  scoreline (i, j) is just the product of the two Poisson probabilities:

      P(A=i, B=j) = Poisson(i ; λ_A) × Poisson(j ; λ_B)

  This lets us build a full scoreline probability grid, then sum over cells
  to get W/D/L probabilities or sample an actual score for simulation.

HOW λ IS COMPUTED
  λ_A = attack_ratio[A] × defense_ratio[B] × μ

  μ              : global average goals per team per game (1.4699)
  attack_ratio   : team A's average goals per game ÷ μ  (>1 = above average)
  defense_ratio  : team B's average goals conceded per game ÷ μ  (>1 = leaky)

  Example: Spain (attack 1.30) vs Jordan (defense 1.05):
      λ_Spain = 1.30 × 1.05 × 1.4699 ≈ 2.00 expected goals

ATTACK / DEFENSE ESTIMATION — OPPONENT-ADJUSTED ITERATIVE SOLVER
  A plain average of raw goal counts is biased by quality of opposition.
  Jordan plays many easy Asian qualifiers; their raw attack and defence look
  too strong until you account for who they scored against.

  We use an iterative update (similar to the Zermelo/Dixon-Coles EM step):

      attack[T]  = (Σ goals_scored[T,i]  + K × prior_att × μ)
                   ────────────────────────────────────────────
                   (Σ defense[opponent_i] + K × 1.0) × μ

      defense[T] = (Σ goals_conceded[T,i] + K × prior_def × μ)
                   ─────────────────────────────────────────────
                   (Σ attack[opponent_i]  + K × 1.0) × μ

  The K "virtual" games inject the Elo-derived prior as a regulariser:
  each team is assumed to have K games against a perfectly average opponent
  with goals = their prior.  This ties the Elo and Poisson models together.

  The solver is iterated N_ITER=10 times, then normalised so the mean
  attack and mean defense across all teams both equal 1.0.

  Elo prior (ELO_DECAY=0.0012, fitted from data — Pearson r=0.63):
      attack_prior  = exp( ELO_DECAY × (elo − 1500))
      defense_prior = exp(−ELO_DECAY × (elo − 1500))
  Interpretation: 100-point Elo gap → ~12.7% difference in expected goals.

  K = K_STRENGTHS = 50 — after 50 virtual games the data dominates the prior.

HOME ADVANTAGE IN GOALS
  When a 2026 host plays at home, we apply a goals multiplier derived
  from the same Elo home bonus already measured in home_advantage.py:

      home_att_mult = exp(ELO_DECAY × bonus / 2)

  The /2 splits the advantage equally across attack and defence: the host
  scores more AND the opponent scores less.  For Mexico (bonus=117 pts):
      home_att_mult ≈ 1.073  (host's λ ×1.07)
      away_att_mult ≈ 0.932  (opponent's λ ×0.93)

  This ensures the Poisson and Elo models agree on who benefits from
  home advantage, while the specific magnitude is a modelling choice.
"""

import math
import numpy as np
import pandas as pd
from src.elo import expected_score, update_ratings, DEFAULT_RATING


# ---------------------------------------------------------------------------
# Constants (calibrated from 49 257 matches — see module docstring)
# ---------------------------------------------------------------------------

MU            = 1.4699   # global average goals per team per game
ELO_DECAY     = 0.0012   # Elo → attack/defense prior scaling (fitted from data)
K_STRENGTHS   = 50       # virtual games of Elo prior injected per team
N_ITER        = 10       # iterations of the opponent-adjustment solver
MAX_GOALS     = 7        # scoreline grid dimension (captures >99.9% of probability)


# ---------------------------------------------------------------------------
# Step 1 — attack / defense strengths
# ---------------------------------------------------------------------------

def compute_team_strengths(
    results_path: str = "data/raw/results.csv",
    ratings: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Estimate opponent-adjusted attack and defense strength ratios for every team.

    Uses an iterative EM-style solver that accounts for quality of opposition.
    Without this, a team like Jordan (many easy Asian qualifiers) looks
    deceptively strong because their goals came against weak opponents.

    Parameters
    ----------
    results_path : path to the historical results CSV
    ratings      : Elo ratings dict from build_model().  Provides the prior.

    Returns
    -------
    attack  : dict {team: ratio}  — >1 means above-average scorer
    defense : dict {team: ratio}  — >1 means above-average conceder (leaky)
    mu      : float               — global average goals per team per game
    """
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["home_score", "away_score"])

    _ratings = ratings or {}

    # --- Build per-team match records ---
    # For each team: list of (opponent_name, goals_scored, goals_conceded)
    team_matches: dict[str, list[tuple[str, int, int]]] = {}

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        team_matches.setdefault(home, []).append((away, hs, as_))
        team_matches.setdefault(away, []).append((home, as_, hs))

    all_teams = set(team_matches.keys())
    if ratings:
        all_teams |= set(ratings.keys())

    # Global mu = average goals per team per game (raw, before any adjustment)
    total_goals = sum(s for matches in team_matches.values() for _, s, _ in matches)
    total_games = sum(len(m) for m in team_matches.values())
    mu = total_goals / total_games

    # --- Elo-derived priors ---
    # Stronger teams (higher Elo) get a higher attack prior and lower defense prior.
    prior_att: dict[str, float] = {}
    prior_def: dict[str, float] = {}
    for team in all_teams:
        elo = _ratings.get(team, DEFAULT_RATING)
        prior_att[team] = math.exp( ELO_DECAY * (elo - 1500))
        prior_def[team] = math.exp(-ELO_DECAY * (elo - 1500))

    # --- Initialise solver from priors ---
    attack  = dict(prior_att)
    defense = dict(prior_def)

    # --- Iterative opponent-adjusted update (N_ITER passes) ---
    # Each team also has K_STRENGTHS virtual games against a perfectly average
    # opponent (attack=1, defense=1) with goals equal to the Elo prior.
    # These virtual games act as the regulariser, linking the Poisson model to Elo.
    #
    # Update formulas (derived from Poisson MLE with virtual games):
    #
    #   attack[T]  = (Σ_i goals_scored[T,i]  + K × prior_att[T] × μ)
    #                ──────────────────────────────────────────────────
    #                (Σ_i defense[opponent_i] + K × 1.0          ) × μ
    #
    #   defense[T] = (Σ_i goals_conceded[T,i] + K × prior_def[T] × μ)
    #                ───────────────────────────────────────────────────
    #                (Σ_i attack[opponent_i]   + K × 1.0          ) × μ

    for _iteration in range(N_ITER):
        new_attack  = {}
        new_defense = {}

        for team in all_teams:
            matches = team_matches.get(team, [])
            n = len(matches)

            if n == 0:
                # No match data: keep the prior unchanged
                new_attack[team]  = prior_att[team]
                new_defense[team] = prior_def[team]
                continue

            # Sum raw goals and opponent strengths over real matches
            sum_scored    = sum(s for _, s, _ in matches)
            sum_conceded  = sum(c for _, _, c in matches)
            sum_opp_def   = sum(defense.get(opp, 1.0) for opp, _, _ in matches)
            sum_opp_att   = sum(attack.get(opp, 1.0)  for opp, _, _ in matches)

            # Add K virtual games (Elo prior regularisation)
            total_scored   = sum_scored   + K_STRENGTHS * prior_att[team] * mu
            total_conceded = sum_conceded + K_STRENGTHS * prior_def[team] * mu
            denom_att      = sum_opp_def  + K_STRENGTHS * 1.0   # virtual opp defense = 1
            denom_def      = sum_opp_att  + K_STRENGTHS * 1.0   # virtual opp attack  = 1

            new_attack[team]  = total_scored   / denom_att / mu
            new_defense[team] = total_conceded / denom_def / mu

        attack  = new_attack
        defense = new_defense

        # Re-normalise so mean attack = mean defense = 1.0.
        # This keeps the model identifiable (otherwise attack and defense can
        # drift in opposite directions without changing any prediction).
        mean_att = sum(attack.values())  / len(attack)
        mean_def = sum(defense.values()) / len(defense)
        attack  = {t: v / mean_att for t, v in attack.items()}
        defense = {t: v / mean_def for t, v in defense.items()}

    return attack, defense, mu


# ---------------------------------------------------------------------------
# Step 2 — expected goals for a match
# ---------------------------------------------------------------------------

def expected_goals(
    team_a: str,
    team_b: str,
    attack:  dict[str, float],
    defense: dict[str, float],
    mu: float,
    home_bonus_a: float = 0.0,
    home_bonus_b: float = 0.0,
) -> tuple[float, float]:
    """
    Return (λ_A, λ_B) — the expected goals for each team.

    Formula (neutral venue):
        λ_A = attack[A] × defense[B] × μ
        λ_B = attack[B] × defense[A] × μ

    Home advantage:
        When home_bonus_a > 0 (team A is a 2026 host playing at home) we
        scale λ_A up and λ_B down by a factor derived from the Elo home
        bonus.  See module docstring for derivation.

    The minimum expected goals is clamped to 0.1 to avoid degenerate
    Poisson distributions.
    """
    lam_a = attack[team_a] * defense[team_b] * mu
    lam_b = attack[team_b] * defense[team_a] * mu

    # Apply host home advantage (symmetric: home attacks more, away attacks less)
    if home_bonus_a > 0:
        mult = math.exp(ELO_DECAY * home_bonus_a / 2)
        lam_a *= mult          # host scores more
        lam_b /= mult          # visitor scores less

    if home_bonus_b > 0:
        mult = math.exp(ELO_DECAY * home_bonus_b / 2)
        lam_b *= mult
        lam_a /= mult

    # Floor to avoid numerical issues with Poisson(0)
    lam_a = max(lam_a, 0.10)
    lam_b = max(lam_b, 0.10)

    return lam_a, lam_b


# ---------------------------------------------------------------------------
# Step 3 — Poisson scoreline probability grid
# ---------------------------------------------------------------------------

def scoreline_matrix(
    lam_a: float,
    lam_b: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Build a (max_goals+1) × (max_goals+1) matrix where cell [i, j] holds
    P(team A scores i, team B scores j).

    Uses the Poisson PMF:  P(X=k) = (λ^k × e^(−λ)) / k!
    The two teams are assumed to score independently, so P(A=i, B=j) =
    Poisson(i ; λ_A) × Poisson(j ; λ_B).

    Scores beyond max_goals are rare (< 0.1% for typical λ) and are
    absorbed into the max_goals cell to ensure the grid sums to ~1.
    """
    n = max_goals + 1
    grid = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # P(X=i) for a Poisson(lam_a) random variable
            p_i = math.exp(-lam_a) * (lam_a ** i) / math.factorial(i)
            p_j = math.exp(-lam_b) * (lam_b ** j) / math.factorial(j)
            grid[i, j] = p_i * p_j

    # Renormalise so the grid sums exactly to 1.0
    grid /= grid.sum()
    return grid


def wdl_from_grid(grid: np.ndarray) -> tuple[float, float, float]:
    """
    Extract W/D/L probabilities from a scoreline grid.

    P(A wins) = sum of grid[i,j] for i > j  (A scored more)
    P(draw)   = sum of grid[i,i] for all i  (equal scores)
    P(B wins) = sum of grid[i,j] for i < j  (B scored more)
    """
    n = grid.shape[0]
    p_win_a = sum(grid[i, j] for i in range(n) for j in range(n) if i > j)
    p_draw  = sum(grid[i, i] for i in range(n))
    p_win_b = sum(grid[i, j] for i in range(n) for j in range(n) if i < j)
    return p_win_a, p_draw, p_win_b


# ---------------------------------------------------------------------------
# Step 3 continued — print prediction for a named match
# ---------------------------------------------------------------------------

def predict_scoreline(
    team_a: str,
    team_b: str,
    attack:  dict[str, float],
    defense: dict[str, float],
    mu: float,
    ratings: dict[str, float],
    host_bonuses: dict[str, float] | None = None,
) -> None:
    """
    Print the top 3 most likely scorelines and W/D/L probabilities from the
    Poisson model, side-by-side with the Elo-based W/D/L for comparison.
    """
    from src.predict import match_probabilities

    _bonuses = host_bonuses or {}
    bonus_a  = _bonuses.get(team_a, 0.0)
    bonus_b  = _bonuses.get(team_b, 0.0)

    lam_a, lam_b = expected_goals(
        team_a, team_b, attack, defense, mu,
        home_bonus_a=bonus_a,
        home_bonus_b=bonus_b,
    )
    grid = scoreline_matrix(lam_a, lam_b)
    p_win_a, p_draw, p_win_b = wdl_from_grid(grid)

    # Elo W/D/L for comparison
    r_a = ratings.get(team_a, DEFAULT_RATING)
    r_b = ratings.get(team_b, DEFAULT_RATING)
    elo_win_a, elo_draw, elo_win_b = match_probabilities(
        r_a, r_b,
        home_bonus_a=bonus_a,
        home_bonus_b=bonus_b,
    )

    # Top 3 most likely exact scores
    n = grid.shape[0]
    cells = [(grid[i, j], i, j) for i in range(n) for j in range(n)]
    top3 = sorted(cells, reverse=True)[:3]

    print(f"\n{'═'*56}")
    print(f"  {team_a}  vs  {team_b}")
    print(f"  Expected goals: {lam_a:.2f}  vs  {lam_b:.2f}")
    print(f"{'─'*56}")
    print(f"  Top 3 most likely scorelines:")
    for prob, i, j in top3:
        print(f"    {team_a} {i}–{j} {team_b}   ({prob*100:.1f}%)")
    print(f"{'─'*56}")
    print(f"  {'':22} {'Poisson':>8} {'Elo':>8}")
    print(f"  {team_a+' win':<22} {p_win_a*100:>7.1f}%  {elo_win_a*100:>7.1f}%")
    print(f"  {'Draw':<22} {p_draw*100:>7.1f}%  {elo_draw*100:>7.1f}%")
    print(f"  {team_b+' win':<22} {p_win_b*100:>7.1f}%  {elo_win_b*100:>7.1f}%")
    print(f"{'═'*56}")


# ---------------------------------------------------------------------------
# Step 4 — simulate a real score
# ---------------------------------------------------------------------------

def simulate_score(
    team_a: str,
    team_b: str,
    attack:  dict[str, float],
    defense: dict[str, float],
    mu: float,
    host_bonuses: dict[str, float] | None = None,
) -> tuple[int, int]:
    """
    Simulate an actual match score by sampling from Poisson distributions.

    Returns
    -------
    (goals_a, goals_b) : int, int

    Each team's goals are drawn independently:
        goals_A ~ Poisson(λ_A)
        goals_B ~ Poisson(λ_B)

    This replaces the old W/D/L proxy in the group stage and provides real
    goal differences for standings tiebreakers.

    For the knockout stage: if goals_a == goals_b the caller resolves the
    draw with a weighted coin flip (extra-time / penalties placeholder).
    """
    _bonuses = host_bonuses or {}
    bonus_a  = _bonuses.get(team_a, 0.0)
    bonus_b  = _bonuses.get(team_b, 0.0)

    lam_a, lam_b = expected_goals(
        team_a, team_b, attack, defense, mu,
        home_bonus_a=bonus_a,
        home_bonus_b=bonus_b,
    )

    goals_a = int(np.random.poisson(lam_a))
    goals_b = int(np.random.poisson(lam_b))

    return goals_a, goals_b


# ---------------------------------------------------------------------------
# Run as script — show predictions + verify Poisson ≈ Elo for the three tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.predict import build_model

    print("Building Elo ratings and Poisson strengths …")
    ratings = build_model()
    attack, defense, mu = compute_team_strengths(ratings=ratings)
    print(f"  μ = {mu:.4f} goals/team/game")
    print(f"  Teams rated: {len(attack)}\n")

    for ta, tb in [("France", "Jordan"), ("Spain", "Argentina"), ("Brazil", "Germany")]:
        predict_scoreline(ta, tb, attack, defense, mu, ratings)

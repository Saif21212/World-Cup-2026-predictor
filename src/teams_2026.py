"""
2026 FIFA World Cup group assignments.

GROUPS maps group letter → list of team names exactly as they appear in
the Elo ratings dataset (martj42/international_results).  Names were
verified against the dataset via check_team_names() before use.
"""

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}


def check_team_names(ratings: dict[str, float]) -> None:
    """
    Check every team in GROUPS against the Elo ratings dict.

    Prints:
      - teams that matched fine
      - teams with NO match, plus the closest names found in the dataset
        (any dataset name that contains the team string or vice-versa)
    """
    matched = []
    unmatched = []

    all_dataset_teams = list(ratings.keys())

    for group, teams in GROUPS.items():
        for team in teams:
            if team in ratings:
                matched.append(f"  [Group {group}] {team}")
            else:
                # Find close candidates: dataset names that share substrings
                close = [
                    t for t in all_dataset_teams
                    if team.lower() in t.lower() or t.lower() in team.lower()
                ]
                unmatched.append((group, team, close[:5]))

    print(f"=== Team name check: {len(matched)} matched, {len(unmatched)} unmatched ===\n")

    print("MATCHED:")
    for m in matched:
        print(m)

    print(f"\nUNMATCHED ({len(unmatched)}):")
    if unmatched:
        for group, team, suggestions in unmatched:
            hint = f"  → suggestions: {suggestions}" if suggestions else "  → no close match found"
            print(f"  [Group {group}] '{team}'{hint}")
    else:
        print("  (none — all teams found)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.predict import build_model
    ratings = build_model()
    check_team_names(ratings)

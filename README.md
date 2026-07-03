# FIFA World Cup 2026 Predictor

A full match-prediction pipeline for the 2026 FIFA World Cup, built from scratch in Python — from raw historical results to calibrated win/draw/loss probabilities and simulated tournament outcomes.

## What it does

- **Data**: Ingests international football results dating back to 1872.
- **Ratings**: Computes Elo ratings for every national team, extended with home-advantage and tournament-importance weighting.
- **Modeling**: Layers a Dixon-Coles adjustment on top of Elo to model realistic scorelines (not just win/draw/loss), rather than relying on vanilla Elo alone.
- **Simulation**: Runs Monte Carlo simulations of the full tournament bracket to produce probabilities for group outcomes, knockout advancement, and the eventual champion.
- **Validation**: Backtested against past World Cups using a leakage-free walk-forward setup — no future results leak into training. Achieved a Brier skill score improvement of **+0.076** over a baseline model.
- **Serving**: Exposes predictions through a FastAPI backend, with a React frontend for browsing matchups and probabilities.

## Why

Most public World Cup predictors either (a) hand-wave probabilities from vibes, or (b) use black-box models with no visible validation. This project is built to be transparent end-to-end: every modeling choice (Elo parameters, Dixon-Coles fit, simulation count) is backtestable and the code shows the work.

All predictions were pre-registered before the tournament kicked off on June 11, 2026, so results can be checked against what the model actually said in advance rather than fitted after the fact.

## Tech stack

- **Modeling**: Python, NumPy, pandas, SciPy
- **Backend**: FastAPI
- **Frontend**: React
- **Validation**: Custom walk-forward backtesting framework, Brier score scoring

## Project structure

```
├── data/               # Historical match results (1872–present)
├── models/             # Elo, Dixon-Coles, Monte Carlo simulation code
├── backtesting/        # Walk-forward validation framework
├── api/                # FastAPI backend
├── frontend/           # React app
└── notebooks/          # Exploration and analysis
```

## Getting started

```bash
git clone https://github.com/Saif21212/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm start
```

## Results

- Brier skill score: **+0.076** improvement over baseline
- Backtested across [N] historical tournaments with no data leakage
- Pre-registered predictions submitted before June 11, 2026 kickoff

## Notes

Personal underdog storyline I'm tracking: Jordan's tournament run, as a nod to where I'm from.

## Status

Actively maintained — next up: [add your current focus, e.g. live score updates, post-tournament model evaluation against actual results].

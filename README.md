# omaDOS 🐑

**omaDOS** is a reinforcement-learning agent for the traditional Bavarian card game **Schafkopf** — built on a from-scratch, fully tested rules engine.

---

## 🃏 What's in a Name?

The name is a triple-layered pun:

* **The "Oma" (Grandma):** In 3-player Schafkopf, the "Oma" refers to the "ghost" fourth hand. Traditionally, this hand follows a brainless heuristic (simply playing the highest card).
* **DOS (Spanish for 2):** the evolution of the "Oma" into its second, smarter iteration.
* **The Technical Backronym:** **D**ifferentially **O**ptimized **S**chafkopf.

## What's implemented

* **Game engine** (`src/omados/engine`): cards, the eight Spielarten, trump/legal-move generation, two-phase bidding with viable-game enforcement, trick resolution, and scoring (win / Schneider / Schwarz, zero-sum rewards).
* **Agents** (`src/omados/agents`): `RandomAgent`, `GreedyAgent`, and a learnable `RLAgent`.
* **Environment** (`src/omados/env`): `SchafkopfEnv` runs full games end-to-end (deal → bidding → 8 tricks → outcome).
* **RL** (`src/omados/models`, `src/omados/training`): a configurable `PolicyNet` (MLP) and a REINFORCE training loop with a running-mean baseline and periodic self-play opponent snapshots.
* **Tests** (`tests/`): 150+ unit tests covering the engine (legal moves, bidding, scoring, teams, tricks).

## Approach

The `RLAgent` encodes the game state into a ~1300-dim observation tensor (hand, legal moves, trump, Rufsau, Spielart, player/caller IDs, current trick, and the 7-trick history) and applies a **masked softmax** over the 32 cards so only legal moves get probability mass. Training uses **REINFORCE** with discounted returns and a baseline for variance reduction; opponents are frozen snapshots of the policy, refreshed periodically for self-play.

## Status

Engine and bidding are complete and tested. The RL agent currently trains against random opponents and self-play snapshots — work in progress on reward shaping and stronger evaluation. This is an active learning project, not a finished product.

## 🛠 Setup

Uses [uv](https://docs.astral.sh/uv/) for reproducible dependency management.

```bash
# install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

## Quickstart

```bash
# play one demo game between four random agents
uv run python main.py

# train the RL agent (hyperparameters at the top of the script)
uv run python scripts/train_reinforce.py

# run the test suite
uv run pytest
```

### Managing dependencies

Use uv rather than `pip install` so the lockfile stays authoritative:

```bash
uv add <package-name>          # runtime dependency
uv add --dev <package-name>    # dev-only tool (tests, linting, notebooks)
```

Always commit `uv.lock` so environments are reproducible.

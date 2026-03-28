# omaDOS 🐑

**omaDOS** is a Reinforcement Learning (RL) based Artificial Intelligence designed to master the traditional Bavarian card game, **Schafkopf**.

---

## 🃏 What’s in a Name?

The name is a triple-layered pun:

* **The "Oma" (Grandma):** In 3-player Schafkopf, the "Oma" refers to the "ghost" fourth hand. Traditionally, this hand follows a brainless heuristic (simply playing the highest card).
* **DOS (Spanish for 2):** Representing the evolution of the "Oma" into its second, smarter iteration.
* **The Technical Backronym:** > **D**ifferentially **O**ptimized **S**chafkopf.

## 🛠 Development Setup

We use [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management. It handles Python versions and virtual environments automatically.

### 1. Install uv
If you don't have it yet, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Initialize the Project
Clone the repository and run the following command from the project root:
```bash
uv sync
```

### 3. Managing Packages

To keep our environments identical, do not use pip install. Instead, use uv to update the project configuration:

* **Add a Core Package**: (e.g., a new RL library or data tool)
    ```bash
    uv add <package-name>
    ```

* **Add a Dev Tool**: Use the --dev flag for tools only needed for coding or testing (e.g., plotting, linting). This keeps the production AI lean:
    ```bash
    uv add --dev <package-name>
    ```

* **Syncing Changes**: If you see that I added a package in a Git commit, simply run uv sync again to update your local environment.


> **IMPORTANT**
Always commit the uv.lock file. This file ensures we both use the exact same versions of every library, preventing "it works on my machine" bugs.
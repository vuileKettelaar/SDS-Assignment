# Getting Started

This project uses `uv` to manage Python dependencies.

## 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Initialize uv

```bash
uv init
```

## 3. Install dependencies Or Sync the dependancies to match what everyone is using

```bash
uv sync
```

## 4. Run the project

```bash
uv run script.py
```

## 5. Add a new library (Example)

```bash
uv add matlplotlib 
```

## 6. Activate the virtual environment

```bash
.venv/Scripts/activate 
```

## 7. USe the environment in Jupyter

Just select the Python version that is inside of your .venv folder and Jupyter will run with all of the correct dependancies


That's it! `uv` handles the virtual environment for you automatically.
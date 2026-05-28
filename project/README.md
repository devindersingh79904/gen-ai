## LangGraph sandbox

### Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

### Run (no LLM required)

```bash
. .venv/bin/activate
python hello_langgraph.py
```

### Run (LLM)

1) Create `.env` (you can start from `.env.example`)

```bash
cp .env.example .env
```

2) Put your key in `.env`:

```bash
OPENAI_API_KEY=...
```

3) Run:

```bash
. .venv/bin/activate
python llm_langgraph.py
```

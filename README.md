---
title: PipelineEnv
emoji: 🔧
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - ci-cd
  - devops
  - rl-environment
---

# PipelineEnv 🔧

> An OpenEnv-compliant Reinforcement Learning environment where an AI agent diagnoses and repairs broken CI/CD pipelines — a real-world DevOps self-healing scenario.

**Live Demo:** [https://endraode-pipeline-env.hf.space/ui](https://endraode-pipeline-env.hf.space/ui)

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Environment Specification](#environment-specification)
- [Tasks](#tasks)
- [Reward Function](#reward-function)
- [Grading System](#grading-system)
- [API Endpoints](#api-endpoints)
- [Local Setup](#local-setup)
- [Docker Build & Deploy](#docker-build--deploy)
- [Baseline Inference](#baseline-inference)
- [Validation](#validation)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Every engineering team faces broken CI/CD pipelines — a bad merge breaks tests, an invalid Dockerfile kills the build, a missing environment variable crashes deployment. **PipelineEnv** simulates these exact scenarios in a structured RL environment where an agentic system must diagnose failures and apply the correct repair actions in the correct order.

### Key Features

- **Real-world domain** — models actual DevOps failure modes engineers encounter daily
- **3 difficulty tiers** — easy (single fix), medium (multi-component), hard (ordered sequence)
- **Deterministic grading** — stage-weighted health scores with action-order enforcement
- **Interactive dashboard** — Gradio UI with live terminal, health bar, and stage visualization
- **REST API** — fully OpenEnv-compliant `step() / reset() / state()` endpoints
- **Docker-native** — containerized deployment tested with `docker build &amp;&amp; docker run`

---

## Quick Start

```bash
# Local development
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Open dashboard
open http://localhost:7860/ui
```

The environment starts immediately. No dataset downloads, no database setup.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  FastAPI Server (server/app.py)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────  │
│  │ /reset      │  │ /step        │  │ /state     │  │
│  │ POST        │  │ POST         │  │ GET        │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬─────┘  │
│         │                │                 │        │
│  ┌──────▼────────────────▼─────────────────▼──────┐ │
│  │  PipelineEnvironment (RL loop)                  │ │
│  │  - scenario selection                           │ │
│  │  - action execution                             │ │
│  │  - health computation                           │ │
│  │  - reward shaping                               │ │
│  │  - action history tracking                      │ │
│  └──────┬─────────────────────────────────────────┘ │
│         │                                            │
│  ┌──────▼─────────────────────────────────────────┐ │
│  │  Graders (server/graders.py)                    │ │
│  │  - compute_health_score() (weighted stages)    │ │
│  │  - grade_task() (deterministic 0.0-1.0)        │ │
│  │  - ACTION_ORDER enforcement (hard task)        │ │
│  └────────────────────────────────────────────────┘ │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │  Gradio UI (/ui) — deterministic agent demo    │ │
│  │  Pipeline stages | Health bar | Terminal       │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## Environment Specification

### Observation Space

The agent observes the full pipeline state after each action:

| Field | Type | Description |
|-------|------|-------------|
| `pipeline_name` | `str` | Name of the pipeline scenario |
| `stages` | `List[dict]` | All stages with `name`, `status`, `error`, `runtime` |
| `failing_count` | `int` | Number of failing stages |
| `health_score` | `float` | Overall health in `[0.0, 1.0]` |
| `error_messages` | `List[str]` | Human-readable error strings from failing stages |
| `available_actions` | `List[str]` | All repair actions the agent can take |
| `task_description` | `str` | Natural-language description of the failure |
| `step_number` | `int` | Current step counter |
| `max_steps` | `int` | Maximum allowed steps before forced `done=True` |

### Action Space

| Action | Description | Affected Stages |
|--------|-------------|-----------------|
| `fix_test` | Fix a failing unit/integration test | `test` (builds deploy) |
| `set_env_var` | Set a missing environment variable | `deploy` |
| `fix_docker_config` | Fix Dockerfile misconfiguration | `build` (unskips test) |
| `fix_yaml_config` | Fix disabled pipeline YAML config | `deploy` |
| `retry_stage` | Retry a flaky stage (partial recovery) | specified stage only |
| `rollback_commit` | Rollback a breaking commit (partial) | `build` (exposes dependency) |
| `add_dependency` | Install a missing package/dependency | `build`, `test` |
| `no_op` | Pass — penalized -0.1 per step | none |

---

## Tasks

| Task | Pipeline | Scenario | Max Steps | Start Health | Required Actions |
|------|----------|----------|-----------|-------------|------------------|
| `easy` | simple-app-pipeline | A unit test is failing; fix the test | 5 | 0.20 `fix_test` |
| `medium` | dockerized-api-pipeline | Docker build config broken + missing env var | 8 | 0.00 | `fix_docker_config` → `set_env_var` |
| `hard` | multi-service-pipeline | Cascading 3-stage failure: bad commit, missing dependency, disabled YAML, disabled in config | 12 | 0.00 | `rollback_commit` → `add_dependency` → `fix_yaml_config` |

### Task Breakdown

#### Easy — `simple-app-pipeline`
- **Build:** passing (green)
- **Test:** failing — `AssertionError: test_add failed — expected 4 got 5`
- **Deploy:** skipped (blocked by failing test)
- **Fix:** Apply `fix_test` → all stages transition to passing

#### Medium — `dockerized-api-pipeline`
- **Build:** failing — `Docker build failed: invalid FROM instruction`
- **Test:** skipped (blocked by build failure)
- **Deploy:** failing — `Missing env var: DATABASE_URL`
- **Fix:** `fix_docker_config` fixes build and unskips test, `set_env_var` fixes deploy

#### Hard — `multi-service-pipeline`
- **Build:** failing — `ModuleNotFoundError: No module named 'requests'`
- **Test:** failing — `ImportError: cannot import requests`
- **Deploy:** failing — `Deploy stage disabled in pipeline YAML`
- **Fix:** Must be done **in order** — rollback exposes the missing dependency, add_dependency resolves imports, fix_yaml re-enables deploy
- **Wrong order penalized** — grader enforces correct action sequence

---

## Reward Function

The reward provides **dense, varying signals** throughout the episode — never a sparse binary signal:

| Signal | Reward |
|--------|--------|
| Health improvement | `+delta + 0.05` bonus |
| Health regression | `+delta - 0.05` penalty |
| No change in health | `-0.05` |
| `no_op` action | `-0.1` |
| Episode done (`health >= 0.99`) | End of episode |

This means the agent receives **immediate feedback** after every action, allowing it to learn from partial progress and course-correct on wrong decisions.

---

## Grading System

### Health Score (`compute_health_score`)
Deterministic weighted sum over stage statuses:

| Stage | Weight |
|-------|--------|
| `build` | 0.2 |
| `test` | 0.3 |
| `deploy` | 0.5 |

Same pipeline state always produces the same score. Scores are in `[0.0, 1.0]`.

### Task Grader (`grade_task`)
- Returns `1.0` if `health >= 0.99` AND (for hard task) actions are in correct order
- Returns `0.7` for hard task if actions are out of order (even with full health)
- Returns `health_score` for partial progress on easy/medium tasks
- **100% deterministic** — same action sequence always produces same score

---

## API Endpoints

All endpoints are OpenEnv-compliant and tested via `openenv validate`, `docker build`, and `HF Space` deployment.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check, version info |
| `GET` | `/health` | Server health status |
| `POST` | `/reset` | Start new episode `{"task_id": "easy"}` |
| `POST` | `/step` | Take a repair action `{"action": "fix_test"}` |
| `GET` | `/state` | Current episode metadata |

### Response Format
```json
POST /reset → {"task_id": "hard"}
{
  "pipeline_name": "multi-service-pipeline",
  "stages": [
    {"name": "build", "status": "failing", "error": "ModuleNotFoundError: No module named 'requests'", "runtime": 1.5},
    {"name": "test", "status": "failing", "error": "ImportError: cannot import requests", "runtime": 1.0},
    {"name": "deploy", "status": "failing", "error": "Deploy stage disabled in pipeline YAML", "runtime": 0.5}
  ],
  "failing_count": 3,
  "health_score": 0.0,
  "error_messages": ["ModuleNotFoundError…", "ImportError…", "Deploy stage disabled…"],
  "available_actions": ["fix_test", "set_env_var", "fix_docker_config", "fix_yaml_config", "retry_stage", "rollback_commit", "add_dependency", "no_op"],
  "task_description": "A bad commit removed a critical dependency…",
  "step_number": 0,
  "max_steps": 12
}
```

---

## Docker Build & Deploy

### Build
```bash
docker build -t pipeline-env .
```

### Run
```bash
docker run -p 7860:7860 pipeline-env
```

### Environment Variables (optional)
| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Model for inference |
| `HF_TOKEN` | `none` | HuggingFace API key (for LLM calls) |

---

## Baseline Inference

The `inference.py` script runs a headless benchmark over all 3 tasks:

```bash
export HF_TOKEN=hf_xxx          # Your HuggingFace token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export BASE_URL=http://localhost:7860
python inference.py
```

### Output Format (strict std format)
```
[START] task=easy env=pipeline-env model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=fix_test reward=0.85 done=true error=null
[END] success=true steps=1 score=1.00 rewards=0.85
```

---

## Validation

Run the full test suite (163 assertions):
```bash
python test_suite.py
```

Run the OpenEnv validator:
```bash
openenv validate
# [OK] pipeline: Ready for multi-mode deployment
```

Run the pre-submission checker:
```bash
./validate-submission.sh https://endraode-pipeline-env.hf.space .
```

---

## Project Structure

```
pipeline-env/
├── server/
│   ├── __init__.py
│   ├── app.py                  # FastAPI REST server + Gradio mount
│   ├── pipeline_environment.py # Core RL environment (reset/step/state)
│   ├── pipeline_scenarios.py   # Pre-broken pipeline definitions
│   ├── graders.py              # Deterministic health & task graders
│   └── requirements.txt        # Server dependencies
├── models.py                   # Pydantic models (Action, Observation, State)
├── inference.py                # Baseline headless benchmark script
├── ui.py                       # Gradio dashboard (interactive demo)
├── test_suite.py               # Comprehensive test suite (163 tests)
├── openenv.yaml                # OpenEnv metadata & task definitions
├── pyproject.toml              # Project config + setuptools scripts entry
├── Dockerfile                  # Containerized build
├── README.md                   # This file
└── uv.lock                     # Deterministic dependency lock file
```

---

## License

MIT

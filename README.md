---
title: Pipeline Env
emoji: 🔧
colorFrom: purple
colorTo: cyan
sdk: docker
app_port: 7860
tags:
  - openenv
  - ci-cd
  - devops
  - rl-environment
---

# PipelineEnv 🔧

An OpenEnv-compliant RL environment where an AI agent diagnoses and repairs broken CI/CD pipelines.

## Environment Description

The agent observes a broken pipeline with failing stages and must apply repair actions to restore it to full health. The environment simulates real DevOps scenarios engineers face daily.

## Action Space

| Action | Description |
|--------|-------------|
| `fix_test` | Fix a failing unit/integration test |
| `set_env_var` | Set a missing environment variable |
| `fix_docker_config` | Fix Dockerfile misconfiguration |
| `fix_yaml_config` | Fix pipeline YAML configuration |
| `retry_stage` | Retry a flaky stage |
| `rollback_commit` | Rollback a breaking commit |
| `add_dependency` | Add a missing package/dependency |
| `no_op` | Do nothing (penalized) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `pipeline_name` | str | Name of the pipeline |
| `stages` | list | All pipeline stages with status |
| `failing_count` | int | Number of failing stages |
| `health_score` | float | 0.0 → 1.0 pipeline health |
| `error_messages` | list | Human-readable errors |
| `available_actions` | list | Valid actions |
| `task_description` | str | Natural language goal |
| `step_number` | int | Current step |
| `max_steps` | int | Maximum steps allowed |

## Tasks

| Task | Description | Max Steps | Start Health |
|------|-------------|-----------|--------------|
| `easy` | Fix a single failing unit test | 5 | 0.2 |
| `medium` | Fix broken Docker config + missing env var | 8 | 0.0 |
| `hard` | Fix cascading 3-stage failure in correct order | 12 | 0.05 |

## Reward Function

- Positive reward = health improvement + 0.05 bonus
- Negative reward = health regression - 0.05 penalty
- `no_op` = -0.1 penalty
- Full score (1.0) when health >= 0.99

## Setup
```bash
pip install openenv-core fastapi uvicorn pydantic openai requests
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## API Endpoints

- `POST /reset` — Start new episode `{"task_id": "easy"}`
- `POST /step` — Take action `{"action": "fix_test"}`
- `GET /state` — Get episode metadata

## Baseline Scores

| Task | Score |
|------|-------|
| easy | 1.0 (1 correct action) |
| medium | 1.0 (2 correct actions) |
| hard | 1.0 (3 correct actions in order) |

## License
MIT
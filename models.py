# models.py — Typed models for PipelineEnv
from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ─── ACTION ──────────────────────────────────────────
class RepairAction(str, Enum):
    fix_test          = "fix_test"
    set_env_var       = "set_env_var"
    fix_docker_config = "fix_docker_config"
    fix_yaml_config   = "fix_yaml_config"
    retry_stage       = "retry_stage"
    rollback_commit   = "rollback_commit"
    add_dependency    = "add_dependency"
    no_op             = "no_op"


class PipelineAction(Action):
    action: RepairAction
    target: Optional[str] = Field(None, description="Stage or component to act on")
    value:  Optional[str] = Field(None, description="Value for set_env_var or add_dependency")


# ─── OBSERVATION ─────────────────────────────────────
class PipelineStage(BaseModel):
    name:    str
    status:  str
    error:   Optional[str] = None
    runtime: float = 0.0


class PipelineObservation(Observation):
    pipeline_name:     str
    stages:            List[dict]
    failing_count:     int
    health_score:      float
    error_messages:    List[str]
    available_actions: List[str]
    task_description:  str
    step_number:       int
    max_steps:         int


# ─── STATE ───────────────────────────────────────────
class PipelineState(State):
    task_id:       str
    episode_id:    str
    step_count:    int
    health_score:  float
    scenario_name: str
    is_done:       bool
# server/pipeline_environment.py — Core RL environment logic
import copy
from uuid import uuid4
from openenv.core.env_server import Environment
from models import PipelineAction, PipelineObservation, PipelineState, RepairAction
from server.pipeline_scenarios import SCENARIOS
from server.graders import compute_health_score, grade_task


class PipelineEnvironment(Environment):

    def __init__(self):
        self.current_task_id  = None
        self.episode_id       = None
        self.step_count       = 0
        self.is_done          = False
        self.health_score     = 0.0
        self.max_steps        = 5
        self.task_description = ""
        self.scenario_name    = "none"
        self.stages           = []

    # ─── RESET ───────────────────────────────────────────
    def reset(self, task_id: str = "easy") -> PipelineObservation:
        scenario = SCENARIOS[task_id]

        self.current_task_id  = task_id
        self.episode_id       = str(uuid4())
        self.step_count       = 0
        self.is_done          = False
        self.max_steps        = scenario["max_steps"]
        self.task_description = scenario["task_description"]
        self.scenario_name    = scenario["name"]
        self.stages           = copy.deepcopy(scenario["stages"])
        self.required_actions = list(scenario["required_actions"])
        self.health_score     = compute_health_score(self.stages)

        return self._make_observation()

    # ─── STEP ────────────────────────────────────────────
    def step(self, action: PipelineAction):
        if self.is_done:
            return self._make_step_result(reward=0.0)

        self.step_count += 1
        prev_health = self.health_score

        # Apply action
        self._apply_action(action)

        # Recompute health
        self.health_score = compute_health_score(self.stages)

        # Compute reward
        reward = self._compute_reward(prev_health, self.health_score, action)

        # Check done
        self.is_done = (
            self.health_score >= 0.99
            or self.step_count >= self.max_steps
        )

        return self._make_step_result(reward=reward)

    # ─── STATE ───────────────────────────────────────────
    @property
    def state(self) -> PipelineState:
        return PipelineState(
            task_id       = self.current_task_id or "none",
            episode_id    = self.episode_id or "not-started",
            step_count    = self.step_count,
            health_score  = self.health_score,
            scenario_name = self.scenario_name,
            is_done       = self.is_done,
        )

    # ─── INTERNAL HELPERS ────────────────────────────────
    def _apply_action(self, action: PipelineAction):
        a = action.action

        if a == RepairAction.fix_test:
            self._heal_stage("test")
            self._heal_stage("deploy")

        elif a == RepairAction.fix_docker_config:
            self._heal_stage("build")

        elif a == RepairAction.set_env_var:
            self._heal_stage("deploy")

        elif a == RepairAction.fix_yaml_config:
            self._heal_stage("deploy")

        elif a == RepairAction.rollback_commit:
            # Partial — clears build error but doesn't fully fix
            for s in self.stages:
                if s["name"] == "build":
                    s["error"] = "Missing dependency: requests"

        elif a == RepairAction.add_dependency:
            self._heal_stage("build")
            self._heal_stage("test")

        elif a == RepairAction.retry_stage:
            # Small bump — retrying a flaky stage
            target = action.target or ""
            for s in self.stages:
                if s["name"] == target and s["status"] == "failing":
                    s["status"] = "passing"
                    s["error"]  = None

        elif a == RepairAction.no_op:
            pass  # penalized in reward

    def _heal_stage(self, stage_name: str):
        for s in self.stages:
            if s["name"] == stage_name:
                s["status"] = "passing"
                s["error"]  = None

    def _compute_reward(self, prev: float, new: float, action: PipelineAction) -> float:
        if action.action == RepairAction.no_op:
            return -0.1
        delta = new - prev
        if delta > 0:
            return round(delta + 0.05, 3)   # bonus for progress
        elif delta < 0:
            return round(delta - 0.05, 3)   # penalty for making it worse
        else:
            return -0.05                     # no change — small penalty

    def _make_observation(self) -> PipelineObservation:
        failing = [s for s in self.stages if s["status"] == "failing"]
        errors  = [s["error"] for s in self.stages if s["error"]]
        return PipelineObservation(
            pipeline_name     = self.scenario_name,
            stages            = self.stages,
            failing_count     = len(failing),
            health_score      = self.health_score,
            error_messages    = errors,
            available_actions = [a.value for a in RepairAction],
            task_description  = self.task_description,
            step_number       = self.step_count,
            max_steps         = self.max_steps,
        )

    def _make_step_result(self, reward: float):
        obs = self._make_observation()
        return {
            "observation": obs.model_dump(),
            "reward":      reward,
            "done":        self.is_done,
            "info": {
                "health_score": self.health_score,
                "grader_score": grade_task(self.current_task_id, self.health_score),
                "step_count":   self.step_count,
            }
        }
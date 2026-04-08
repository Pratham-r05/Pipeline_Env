# server/graders.py — Deterministic graders for PipelineEnv

STAGE_WEIGHTS = {
    "build":  0.2,
    "test":   0.3,
    "deploy": 0.5,
}


ACTION_ORDER = {
    "hard": ["rollback_commit", "add_dependency", "fix_yaml_config"],
}


def compute_health_score(stages: list) -> float:
    """
    Deterministic grader — same pipeline state always → same score.
    Weighted by stage importance: deploy > test > build.
    Returns float strictly in (0.0, 1.0)
    """
    if not stages:
        return 0.001

    total_weight  = 0.0
    passing_weight = 0.0

    for stage in stages:
        w = STAGE_WEIGHTS.get(stage["name"], 0.1)
        total_weight += w
        if stage["status"] == "passing":
            passing_weight += w

    if total_weight == 0:
        return 0.001

    score = round(passing_weight / total_weight, 3)
    # Clamp to strictly (0, 1) range
    return max(0.001, min(0.999, score))


def _check_action_order(required: list, taken: list) -> bool:
    """
    Check that required actions were taken in the correct order.
    Ignores unrelated actions (no_op, retry, etc.) in the history.
    """
    relevant = [a for a in taken if a in required]
    return relevant == required


def grade_task(task_id: str, final_health: float, action_history: list = None) -> float:
    """
    Final grader for each task.
    Returns score strictly in (0.0, 1.0).
    Full score only if pipeline fully healed.
    For "hard" task, actions must be in correct order.
    """
    action_history = action_history or []

    if final_health >= 0.99:
        # For hard task, also verify action ordering
        order = ACTION_ORDER.get(task_id)
        if order and not _check_action_order(order, action_history):
            score = round(final_health * 0.7, 3)
            return max(0.001, min(0.999, score))  # penalty for wrong order
        return 0.999  # Full success, but strictly less than 1.0

    # Clamp to strictly (0, 1) range
    return max(0.001, min(0.999, round(final_health, 3)))
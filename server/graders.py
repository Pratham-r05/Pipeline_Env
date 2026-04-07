# server/graders.py — Deterministic graders for PipelineEnv

STAGE_WEIGHTS = {
    "build":  0.2,
    "test":   0.3,
    "deploy": 0.5,
}


def compute_health_score(stages: list) -> float:
    """
    Deterministic grader — same pipeline state always → same score.
    Weighted by stage importance: deploy > test > build.
    Returns float in [0.0, 1.0]
    """
    if not stages:
        return 0.0

    total_weight  = 0.0
    passing_weight = 0.0

    for stage in stages:
        w = STAGE_WEIGHTS.get(stage["name"], 0.1)
        total_weight += w
        if stage["status"] == "passing":
            passing_weight += w

    if total_weight == 0:
        return 0.0

    return round(passing_weight / total_weight, 3)


def grade_task(task_id: str, final_health: float) -> float:
    """
    Final grader for each task.
    Returns score in [0.0, 1.0].
    Full score only if pipeline fully healed.
    """
    if final_health >= 0.99:
        return 1.0
    return round(final_health, 3)
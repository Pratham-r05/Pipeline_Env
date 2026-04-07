# server/pipeline_scenarios.py — Pre-broken pipeline scenarios

SCENARIOS = {
    "easy": {
        "name": "simple-app-pipeline",
        "task_description": "A unit test is failing in the test stage. Fix it to restore the pipeline.",
        "max_steps": 5,
        "stages": [
            {"name": "build",  "status": "passing", "error": None,                          "runtime": 12.0},
            {"name": "test",   "status": "failing", "error": "AssertionError: test_add failed — expected 4 got 5", "runtime": 3.0},
            {"name": "deploy", "status": "skipped", "error": "Skipped due to failed test stage", "runtime": 0.0},
        ],
        "breakages": ["test"],
        "required_actions": ["fix_test"],
    },

    "medium": {
        "name": "dockerized-api-pipeline",
        "task_description": "Docker config is broken and a required environment variable is missing. Fix both to restore the pipeline.",
        "max_steps": 8,
        "stages": [
            {"name": "build",  "status": "failing", "error": "Docker build failed: invalid FROM instruction", "runtime": 2.0},
            {"name": "test",   "status": "skipped", "error": "Skipped due to failed build",                  "runtime": 0.0},
            {"name": "deploy", "status": "failing", "error": "Missing env var: DATABASE_URL",                "runtime": 1.0},
        ],
        "breakages": ["build", "deploy"],
        "required_actions": ["fix_docker_config", "set_env_var"],
    },

    "hard": {
        "name": "multi-service-pipeline",
        "task_description": (
            "A bad commit removed a critical dependency, the pipeline YAML disabled the deploy stage, "
            "and the build is broken. Fix all three in the correct order: rollback the commit, "
            "add the missing dependency, then fix the YAML config."
        ),
        "max_steps": 12,
        "stages": [
            {"name": "build",      "status": "failing", "error": "ModuleNotFoundError: No module named 'requests'", "runtime": 1.5},
            {"name": "test",       "status": "failing", "error": "ImportError: cannot import requests",             "runtime": 1.0},
            {"name": "deploy",     "status": "failing", "error": "Deploy stage disabled in pipeline YAML",          "runtime": 0.5},
        ],
        "breakages": ["build", "test", "deploy"],
        "required_actions": ["rollback_commit", "add_dependency", "fix_yaml_config"],
    },
}
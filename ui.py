# ui.py — Visual agent dashboard for PipelineEnv
# Mounts a Gradio app at /ui on top of the FastAPI server.
# Uses a built-in deterministic agent — no external LLM required.

import json
import os
import time
import gradio as gr

from server.pipeline_environment import PipelineEnvironment
from models import PipelineAction, RepairAction

# ── Config ─────────────────────────────────────────────
USE_LLM = os.getenv("USE_LLM", "").lower() == "true"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"

env = PipelineEnvironment()

# Deterministic action plans for each task
ACTION_PLANS = {
    "easy":   ["fix_test"],
    "medium": ["fix_docker_config", "set_env_var"],
    "hard":   ["rollback_commit", "add_dependency", "fix_yaml_config"],
}

ACTION_ICONS = {
    "fix_test": "\U0001f9ea",
    "set_env_var": "\U0001f511",
    "fix_docker_config": "\U0001f40b",
    "fix_yaml_config": "\U0001f4c4",
    "retry_stage": "\U0001f504",
    "rollback_commit": "\u23ea",
    "add_dependency": "\U0001f4e6",
    "no_op": "\U0001f4a4",
}


# ── Helpers ────────────────────────────────────────────

def _render_env(obs, step, max_s):
    lines = []
    name = obs.get("pipeline_name", "pipeline")
    lines.append(
        f'<div style="color:#58a6ff;font-weight:700;margin-bottom:6px">'
        f'\U0001f4e6 Pipeline: {name} &nbsp;|&nbsp; Step {step}/{max_s}</div>'
    )
    for s in obs.get("stages", []):
        icon = {"passing": "\u2705", "failing": "\u274c", "skipped": "\u23ed\ufe0f"}.get(s["status"], "?")
        err = f' <span style="color:#f85149;font-size:11px">({s["error"]})</span>' if s.get("error") else ""
        lines.append(f'<span style="color:#cdd9e5">{icon} {s["name"]:<10} {s["status"]:<10}</span>{err}')
    return "<br>".join(lines)


def _render_health(score):
    pct = int(score * 100)
    color = "#3fb950" if score >= 0.8 else "#d29922" if score >= 0.4 else "#f85149"
    return (
        f'<div style="height:12px;background:#21262d;border-radius:4px;margin:4px 0">'
        f'<div style="height:12px;width:{pct}%;background:{color};border-radius:4px;transition:width .3s"></div></div>'
        f'<div style="font-size:12px;color:#8b949e;text-align:right">Health: {score:.2f} / 1.0 ({pct}%)</div>'
    )


def _deterministic_action(task_id, step_num):
    """Built-in agent that knows the correct action sequence."""
    plan = ACTION_PLANS.get(task_id, [])
    if step_num - 1 < len(plan):
        return {"action": plan[step_num - 1], "target": None, "value": None}
    return {"action": "no_op", "target": None, "value": None}


# ── Yielding generator ──────────────────────────────────

def run_task(task_id):
    obs = env.reset(task_id).model_dump()
    max_s = obs["max_steps"]

    term_lines = []
    action_log = []
    rewards = []
    success = False
    current_score = 0.0

    # reset banner
    term_lines.append("\U0001f680 RESET  \u2014 starting episode")
    term_lines.append(f"    Task  : {task_id}")
    term_lines.append(f"    Desc  : {obs['task_description']}")
    term_lines.append(f"    Steps : {max_s}")
    term_lines.append(f"    Health: {obs['health_score']:.2f}")

    def _snapshot():
        te = "<br>".join(term_lines)
        term_box = f'<div style="font-family:JetBrains Mono,Fira Code,monospace;font-size:13px;line-height:1.6;color:#c9d1d9;background:#0d1117;padding:12px;border-radius:6px;white-space:pre-wrap;max-height:340px;overflow-y:auto">{te}</div>'
        al = "<br>".join(f'<span style="color:#58a6ff;font-family:monospace;font-size:12px">{a}</span>' for a in action_log) if action_log else '<span style="color:#8b949e;font-size:12px">Waiting\u2026</span>'
        sm = f'<span style="color:#8b949e;font-size:12px">Running\u2026 step {len(action_log)}/{max_s} | health {obs["health_score"]:.2f} | score {current_score:.3f}</span>'
        return _render_env(obs, len(action_log), max_s), _render_health(obs["health_score"]), term_box, al, sm

    yield _snapshot()
    time.sleep(0.4)

    for step_num in range(1, max_s + 1):
        # Use deterministic agent
        ad = _deterministic_action(task_id, step_num)
        aname = ad.get("action", "no_op")
        icon = ACTION_ICONS.get(aname, "\u2699")
        action_log.append(f"{icon} Step {step_num}: {aname}")
        term_lines.append(f'    \u279c Action \u279c {aname}')

        try:
            ra = getattr(RepairAction, aname)
        except Exception:
            ra = RepairAction.no_op
        result = env.step(PipelineAction(action=ra, target=ad.get("target"), value=ad.get("value")))
        rwd = result.get("reward", 0) or 0.0
        done = result.get("done", False)
        obs = result.get("observation", obs)
        info = result.get("info", {})
        current_score = info.get("grader_score", current_score)
        rewards.append(rwd)

        if rwd > 0:
            term_lines.append(f'<span style="color:#3fb950">      reward {rwd:+.3f}   health \u2192 {obs["health_score"]:.2f}</span>')
        elif rwd < 0:
            term_lines.append(f'<span style="color:#f85149">      reward {rwd:+.3f}   health \u2192 {obs["health_score"]:.2f}</span>')
        else:
            term_lines.append(f'      reward {rwd:+.3f}   health \u2192 {obs["health_score"]:.2f}')

        yield _snapshot()
        time.sleep(0.3)

        if done:
            success = current_score >= 0.99
            break

    # --- final summary ---
    total_r = sum(rewards)
    avg_r = total_r / len(rewards) if rewards else 0
    sc = "#3fb950" if success else "#f85149"
    st = "\u2705 PIPELINE HEALED" if success else "\u274c FAILED TO HEAL"
    term_lines.append(f'<span style="color:{sc};font-weight:700;font-size:14px">{st}</span>')
    term_lines.append(f'    Steps: {len(action_log)} | Total reward: {total_r:+.3f} | Score: {current_score:.3f}')
    term_lines.append(f'    Agent: Deterministic (built-in)')

    yield _snapshot()


# ── Gradio layout ──────────────────────────────────────

with gr.Blocks(title="PipelineEnv \u2014 Agent Dashboard") as demo:
    gr.HTML('<style>body{font-family:JetBrains Mono,Fira Code,monospace !important;}</style>')

    gr.HTML(
        '<div style="text-align:center;padding:20px 0 8px">'
        '<div style="font-size:28px;font-weight:700">\U0001f527 PipelineEnv</div>'
        '<div style="color:#8b949e;font-size:14px;margin-top:4px">RL Agent \u2022 CI/CD Self-Healing Dashboard</div></div>'
    )

    with gr.Row():
        task_dd = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task")
        run_btn = gr.Button("\u25b6  Run Agent", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div style="color:#8b949e;font-size:12px;margin-bottom:4px">PIPELINE STAGES</div>')
            pipeline_disp = gr.HTML('<div style="color:#8b949e">Select a task and click Run Agent.</div>')
            gr.HTML('<div style="color:#8b949e;font-size:12px;margin:12px 0 4px">HEALTH</div>')
            health_disp = gr.HTML("")

        with gr.Column(scale=1):
            gr.HTML('<div style="color:#8b949e;font-size:12px;margin-bottom:4px">ACTION LOG</div>')
            action_disp = gr.HTML('<div style="color:#8b949e;font-size:12px">Waiting\u2026</div>')
            gr.HTML('<div style="color:#8b949e;font-size:12px;margin:12px 0 4px">SUMMARY</div>')
            summary_disp = gr.HTML("")

    gr.HTML('<div style="color:#8b949e;font-size:12px;margin-bottom:4px">TERMINAL</div>')
    terminal_disp = gr.HTML("")

    run_btn.click(
        fn=run_task,
        inputs=[task_dd],
        outputs=[pipeline_disp, health_disp, terminal_disp, action_disp, summary_disp],
    )


if __name__ == "__main__":
    demo.launch(server_port=7861)

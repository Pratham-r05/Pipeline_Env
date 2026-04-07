# ui.py — Visual agent dashboard for PipelineEnv
# Add this file to your HuggingFace Space root.
# It mounts a Gradio app at /ui on top of your existing FastAPI server.

import os
import json
import time
import requests
import gradio as gr
from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "https://endraode-pipeline-env.hf.space")
API_BASE    = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME  = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN    = os.getenv("HF_TOKEN", "dummy")

STAGE_ICONS = {"passing": "✅", "failing": "❌", "skipped": "⏭️"}
ACTION_ICONS = {
    "fix_test": "🧪", "set_env_var": "🔑", "fix_docker_config": "🐳",
    "fix_yaml_config": "📄", "retry_stage": "🔄", "rollback_commit": "⏪",
    "add_dependency": "📦", "no_op": "💤",
}

TASK_DESCRIPTIONS = {
    "easy":   "🟢 Easy — Fix a single failing unit test",
    "medium": "🟡 Medium — Fix broken Docker config + missing env var",
    "hard":   "🔴 Hard — Fix cascading 3-stage failure in correct order",
}

CSS = """
body { font-family: 'JetBrains Mono', 'Fira Code', monospace !important; }
.pipeline-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    font-family: monospace;
}
.health-bar-wrap { background: #21262d; border-radius: 4px; height: 20px; margin: 8px 0; }
.health-bar { height: 20px; border-radius: 4px; transition: width 0.5s ease; }
.stage-row { display: flex; gap: 12px; align-items: center; padding: 6px 0; border-bottom: 1px solid #21262d; }
.step-log { font-size: 12px; color: #8b949e; }
.action-taken { font-size: 14px; font-weight: bold; color: #58a6ff; }
.reward-positive { color: #3fb950; }
.reward-negative { color: #f85149; }
"""

def health_color(score: float) -> str:
    if score >= 0.8: return "#3fb950"
    if score >= 0.4: return "#d29922"
    return "#f85149"

def render_stages(stages: list) -> str:
    rows = ""
    for s in stages:
        icon = STAGE_ICONS.get(s["status"], "❓")
        err  = f'<span style="color:#f85149;font-size:11px"> — {s["error"]}</span>' if s.get("error") else ""
        rows += f'<div class="stage-row"><b style="color:#cdd9e5;width:70px">{s["name"]}</b> {icon} <span style="color:#8b949e">{s["status"]}</span>{err}</div>'
    return rows

def render_health_bar(score: float) -> str:
    pct   = int(score * 100)
    color = health_color(score)
    return f"""
    <div class="health-bar-wrap">
        <div class="health-bar" style="width:{pct}%;background:{color}"></div>
    </div>
    <div style="text-align:right;font-size:12px;color:#8b949e">Health: {score:.2f} / 1.0 ({pct}%)</div>
    """

def get_agent_action(client, obs: dict) -> dict:
    errors    = obs.get("error_messages", [])
    actions   = obs.get("available_actions", [])
    health    = obs.get("health_score", 0)
    task_desc = obs.get("task_description", "")

    prompt = f"""You are a DevOps engineer fixing a broken CI/CD pipeline.

Task: {task_desc}
Current pipeline health: {health:.2f}/1.0
Errors: {errors}
Available repair actions: {actions}

Respond with ONLY a JSON object like:
{{"action": "fix_test", "target": null, "value": null}}

Choose the single best action to fix the pipeline."""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        return {"action": "no_op", "target": None, "value": None}


def run_agent(task_id: str):
    """
    Generator — yields (pipeline_html, health_html, log_html, summary_html) at each step.
    """
    client = OpenAI(base_url=API_BASE, api_key=HF_TOKEN)
    log_lines = []

    def add_log(line, color="#cdd9e5"):
        log_lines.append(f'<div style="color:{color};font-size:12px;padding:2px 0">{line}</div>')

    # ── RESET ──────────────────────────────────────────────────
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=15)
        r.raise_for_status()
        obs = r.json()
    except Exception as e:
        yield (
            f'<div style="color:#f85149">❌ Failed to connect to environment: {e}</div>',
            "", "", ""
        )
        return

    max_steps = obs.get("max_steps", 12)
    add_log(f"🚀 Episode started | task={task_id} | max_steps={max_steps}", "#58a6ff")

    pipeline_html = f"""
    <div class="pipeline-card">
        <div style="color:#58a6ff;font-size:13px;margin-bottom:8px">📋 {obs.get('pipeline_name','pipeline')} — {TASK_DESCRIPTIONS.get(task_id,'')}</div>
        {render_stages(obs.get('stages', []))}
    </div>"""

    health_html = render_health_bar(obs.get("health_score", 0))
    log_html    = "".join(log_lines)
    summary_html = '<div style="color:#8b949e">Agent running...</div>'

    yield pipeline_html, health_html, log_html, summary_html

    rewards     = []
    steps_taken = 0
    success     = False

    # ── STEP LOOP ──────────────────────────────────────────────
    for step in range(1, max_steps + 1):
        action_dict = get_agent_action(client, obs)
        action_name = action_dict.get("action", "no_op")
        action_icon = ACTION_ICONS.get(action_name, "⚙️")

        add_log(f"── Step {step} ──────────────────────", "#30363d")
        add_log(f"{action_icon} Agent chose: <b style='color:#58a6ff'>{action_name}</b>")

        try:
            sr     = requests.post(f"{BASE_URL}/step", json=action_dict, timeout=15)
            result = sr.json()
        except Exception as e:
            add_log(f"❌ Step failed: {e}", "#f85149")
            break

        reward  = result.get("reward") or 0.0
        done    = result.get("done", False)
        obs     = result.get("observation", obs)
        err     = result.get("info", {}).get("error")
        health  = obs.get("health_score", 0)

        rewards.append(reward)
        steps_taken = step

        r_color = "#3fb950" if reward > 0 else "#f85149" if reward < 0 else "#8b949e"
        add_log(f"   Reward: <span style='color:{r_color}'>{reward:+.3f}</span>  |  Health: {health:.2f}")
        if err:
            add_log(f"   ⚠️ {err}", "#d29922")

        pipeline_html = f"""
        <div class="pipeline-card">
            <div style="color:#58a6ff;font-size:13px;margin-bottom:8px">📋 {obs.get('pipeline_name','pipeline')} — Step {step}/{max_steps}</div>
            {render_stages(obs.get('stages', []))}
        </div>"""

        health_html  = render_health_bar(health)
        log_html     = "".join(log_lines)
        summary_html = '<div style="color:#8b949e">Agent running...</div>'

        yield pipeline_html, health_html, log_html, summary_html
        time.sleep(0.3)

        if done:
            success = health >= 0.99
            break

    # ── FINAL SUMMARY ─────────────────────────────────────────
    total_reward = sum(rewards)
    avg_reward   = total_reward / len(rewards) if rewards else 0
    status_color = "#3fb950" if success else "#f85149"
    status_text  = "✅ PIPELINE HEALED" if success else "❌ FAILED TO HEAL"

    summary_html = f"""
    <div class="pipeline-card" style="border-color:{status_color}">
        <div style="font-size:18px;color:{status_color};margin-bottom:12px">{status_text}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
            <div>
                <div style="color:#8b949e;font-size:11px">TASK</div>
                <div style="color:#cdd9e5;font-size:16px">{task_id.upper()}</div>
            </div>
            <div>
                <div style="color:#8b949e;font-size:11px">STEPS TAKEN</div>
                <div style="color:#cdd9e5;font-size:16px">{steps_taken}</div>
            </div>
            <div>
                <div style="color:#8b949e;font-size:11px">FINAL HEALTH</div>
                <div style="color:{health_color(obs.get('health_score',0))};font-size:16px">{obs.get('health_score',0):.2f}</div>
            </div>
            <div>
                <div style="color:#8b949e;font-size:11px">TOTAL REWARD</div>
                <div style="color:{'#3fb950' if total_reward>0 else '#f85149'};font-size:16px">{total_reward:+.3f}</div>
            </div>
            <div>
                <div style="color:#8b949e;font-size:11px">AVG REWARD/STEP</div>
                <div style="color:#cdd9e5;font-size:16px">{avg_reward:+.3f}</div>
            </div>
            <div>
                <div style="color:#8b949e;font-size:11px">MODEL</div>
                <div style="color:#cdd9e5;font-size:12px">{MODEL_NAME.split('/')[-1]}</div>
            </div>
        </div>
    </div>"""

    add_log(f"{'✅ SUCCESS' if success else '❌ FAILED'} | steps={steps_taken} | total_reward={total_reward:+.3f}", status_color)
    log_html = "".join(log_lines)

    yield pipeline_html, render_health_bar(obs.get("health_score", 0)), log_html, summary_html


# ── BUILD GRADIO UI ────────────────────────────────────────────────────────────
with gr.Blocks(title="PipelineEnv — Agent Dashboard") as demo:
    gr.HTML(f"<style>{CSS}</style>")

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px">
        <div style="font-size:28px;font-weight:700;letter-spacing:-1px">🔧 PipelineEnv</div>
        <div style="color:#8b949e;font-size:14px;margin-top:4px">RL Agent • CI/CD Self-Healing Dashboard</div>
    </div>
    """)

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Select Task",
            scale=2,
        )
        run_btn = gr.Button("▶  Run Agent", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div style='color:#8b949e;font-size:12px;margin-bottom:4px'>PIPELINE STAGES</div>")
            pipeline_display = gr.HTML('<div style="color:#8b949e">Select a task and click Run Agent.</div>')

            gr.HTML("<div style='color:#8b949e;font-size:12px;margin:12px 0 4px'>HEALTH SCORE</div>")
            health_display = gr.HTML("")

        with gr.Column(scale=1):
            gr.HTML("<div style='color:#8b949e;font-size:12px;margin-bottom:4px'>STEP LOG</div>")
            log_display = gr.HTML(
                '<div style="color:#8b949e;font-size:12px">Waiting for agent...</div>',
                elem_id="log-box",
            )

    gr.HTML("<div style='color:#8b949e;font-size:12px;margin:12px 0 4px'>EPISODE SUMMARY</div>")
    summary_display = gr.HTML("")

    run_btn.click(
        fn=run_agent,
        inputs=[task_dropdown],
        outputs=[pipeline_display, health_display, log_display, summary_display],
    )

# ── MOUNT ON FASTAPI ───────────────────────────────────────────────────────────
# In your server/app.py, add these lines at the bottom:
#
#   from ui import demo
#   import gradio as gr
#   app = gr.mount_gradio_app(app, demo, path="/ui")
#
# Then visit:  https://endraode-pipeline-env.hf.space/ui

if __name__ == "__main__":
    demo.launch(server_port=7861)
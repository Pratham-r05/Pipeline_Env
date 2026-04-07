# inference.py — MANDATORY baseline script
import os
import json
import requests
from openai import OpenAI

# ── Mandatory env vars ────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
BASE_URL     = os.getenv("BASE_URL",     "http://localhost:7860")


# ── MANDATORY stdout format ───────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={r_str}", flush=True)


# ── LLM Agent ─────────────────────────────────────────
def get_agent_action(client, observation: dict) -> dict:
    errors    = observation.get("error_messages", [])
    actions   = observation.get("available_actions", [])
    health    = observation.get("health_score", 0)
    task_desc = observation.get("task_description", "")

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
            model       = MODEL_NAME,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 100,
            temperature = 0,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"action": "no_op", "target": None, "value": None}


# ── Main benchmark loop ───────────────────────────────
def run_benchmark():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    tasks  = ["easy", "medium", "hard"]

    for task_id in tasks:
        log_start(task=task_id, env="pipeline-env", model=MODEL_NAME)
        rewards, steps_taken, success = [], 0, False

        try:
            # Reset
            r   = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=15)
            r.raise_for_status()
            obs = r.json()

            max_steps = obs.get("max_steps", 12)

            for step in range(1, max_steps + 1):
                # Fallback action if LLM fails
                try:
                    action_dict = get_agent_action(client, obs)
                except Exception:
                    action_dict = {"action": "no_op", "target": None, "value": None}

                sr     = requests.post(f"{BASE_URL}/step", json=action_dict, timeout=15)
                result = sr.json()

                reward = result.get("reward") or 0.0
                done   = result.get("done",  False)
                obs    = result.get("observation", obs)
                err    = result.get("info", {}).get("error")

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_dict["action"],
                         reward=reward, done=done, error=err)

                if done:
                    success = obs.get("health_score", 0) >= 0.99
                    break

        except Exception as e:
            print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    run_benchmark()
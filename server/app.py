# server/app.py — FastAPI server with all required endpoints
import os
import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from server.pipeline_environment import PipelineEnvironment
from models import PipelineAction, RepairAction

# ── Only build the Gradio UI if running on HF Spaces ───
if os.getenv("SPACE_ID"):
    import gradio as gr
    from ui import demo as gradio_demo

app = FastAPI(title="PipelineEnv", version="1.0.0")

# Single global environment instance
env = PipelineEnvironment()


# ─── REQUEST MODELS ──────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    action: str
    target: Optional[str] = None
    value:  Optional[str] = None


# ─── ENDPOINTS ───────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "env": "pipeline-env", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(req: ResetRequest):
    obs = env.reset(task_id=req.task_id)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    try:
        action = PipelineAction(
            action=RepairAction(req.action),
            target=req.target,
            value=req.value,
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    result = env.step(action)
    return result


@app.get("/state")
def state():
    return env.state.model_dump()


# ─── GRADIO MOUNT (only on HF Spaces) ──────────────────
if os.getenv("SPACE_ID"):
    from ui import demo as _demo
    app = gr.mount_gradio_app(app, _demo, path="/ui")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

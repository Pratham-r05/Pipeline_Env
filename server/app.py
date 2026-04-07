# server/app.py — FastAPI server with all required endpoints
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from server.pipeline_environment import PipelineEnvironment
from models import PipelineAction, RepairAction

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
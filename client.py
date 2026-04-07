# client.py — EnvClient for PipelineEnv
from openenv.core import EnvClient, SyncEnvClient
from models import PipelineAction, PipelineObservation, PipelineState


class PipelineEnv(EnvClient[PipelineAction, PipelineObservation, PipelineState]):

    def _step_payload(self, action: PipelineAction) -> dict:
        return {
            "action": action.action.value,
            "target": action.target,
            "value":  action.value,
        }

    def _parse_observation(self, payload: dict) -> PipelineObservation:
        obs_data = payload.get("observation", payload)
        return PipelineObservation(**obs_data)

    def _parse_state(self, payload: dict) -> PipelineState:
        return PipelineState(**payload)
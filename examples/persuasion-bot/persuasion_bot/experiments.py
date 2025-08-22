from pydantic import BaseModel

import art


class JustTheFactsConfig(BaseModel):
    learning_rate: float = 1e-6
    num_epochs: int = 20

    eval_steps: int = 1
    groups_per_step: int = 3
    trajectories_per_group: int = 4
    scale_rewards: bool = False


models: dict[str, art.TrainableModel[JustTheFactsConfig]] = {
    "persuasion-14b-001": art.TrainableModel(
        name="persuasion-14b-001",
        project="persuasion-bot",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=JustTheFactsConfig(
            num_epochs=20,
        ),
    )
}

models["persuasion-14b-002"] = models["persuasion-14b-001"].model_copy(deep=True)
models["persuasion-14b-002"].name = "persuasion-14b-002"
models["persuasion-14b-002"].base_model = "Qwen/Qwen2.5-14B-Instruct"

models["persuasion-14b-003"] = models["persuasion-14b-001"].model_copy(deep=True)
models["persuasion-14b-003"].name = "persuasion-14b-003"
models["persuasion-14b-003"].base_model = "Qwen/Qwen2.5-14B-Instruct"
models["persuasion-14b-003"].config.scale_rewards = True
models["persuasion-14b-003"].config.trajectories_per_group = 12


models["persuasion-7b-001"] = models["persuasion-14b-001"].model_copy(deep=True)
models["persuasion-7b-001"].name = "persuasion-7b-001"
models["persuasion-7b-001"].base_model = "Qwen/Qwen2.5-7B-Instruct"

models["persuasion-qwen3-4b-001"] = models["persuasion-14b-001"].model_copy(deep=True)
models["persuasion-qwen3-4b-001"].name = "persuasion-qwen3-4b-001"
models["persuasion-qwen3-4b-001"].base_model = "Qwen/Qwen3-4B"

models["persuasion-qwen3-4b-002"] = models["persuasion-14b-001"].model_copy(deep=True)
models["persuasion-qwen3-4b-002"].name = "persuasion-qwen3-4b-002"
models["persuasion-qwen3-4b-002"].base_model = "Qwen/Qwen3-4B"

models["persuasion-qwen3-4b-003"] = models["persuasion-14b-001"].model_copy(deep=True)
models["persuasion-qwen3-4b-003"].name = "persuasion-qwen3-4b-003"
models["persuasion-qwen3-4b-003"].base_model = "Qwen/Qwen3-4B"
models["persuasion-qwen3-4b-003"].config.groups_per_step = 8

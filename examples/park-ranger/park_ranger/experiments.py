from pydantic import BaseModel

import art


class ParkRangerConfig(BaseModel):
    learning_rate: float = 1e-6
    num_epochs: int = 20

    eval_steps: int = 1
    groups_per_step: int = 3
    trajectories_per_group: int = 4


models: dict[str, art.TrainableModel[ParkRangerConfig]] = {
    "ranger-14b-001": art.TrainableModel(
        name="ranger-14b-001",
        project="park-ranger",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=ParkRangerConfig(
            num_epochs=20,
        ),
    )
}


models["ranger-7b-001"] = models["ranger-14b-001"].model_copy(deep=True)
models["ranger-7b-001"].name = "ranger-7b-001"
models["ranger-7b-001"].base_model = "Qwen/Qwen2.5-7B-Instruct"

models["ranger-qwen3-4b-001"] = models["ranger-14b-001"].model_copy(deep=True)
models["ranger-qwen3-4b-001"].name = "ranger-qwen3-4b-001"
models["ranger-qwen3-4b-001"].base_model = "Qwen/Qwen3-4B"

models["ranger-qwen3-4b-002"] = models["ranger-14b-001"].model_copy(deep=True)
models["ranger-qwen3-4b-002"].name = "ranger-qwen3-4b-002"
models["ranger-qwen3-4b-002"].base_model = "Qwen/Qwen3-4B"

models["ranger-qwen3-4b-003"] = models["ranger-14b-001"].model_copy(deep=True)
models["ranger-qwen3-4b-003"].name = "ranger-qwen3-4b-003"
models["ranger-qwen3-4b-003"].base_model = "Qwen/Qwen3-4B"
models["ranger-qwen3-4b-003"].config.groups_per_step = 8

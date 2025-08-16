"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import asyncio

import art
from art.skypilot.backend import SkyPilotBackend
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class McpPolicyConfig(BaseModel):
    max_turns: int = 5
    max_tokens: int = 2048

    base_model: str = "Qwen/Qwen2.5-14B-Instruct"

    # MCP server configuration
    smithery_mcp_url: str = "asdf"

    # Training configuration fields
    trajectories_per_group: int = 7
    groups_per_step: int = 4
    learning_rate: float = 1e-6
    eval_steps: int = 1
    val_set_size: int = 8
    training_dataset_size: int = 16
    num_epochs: int = 80
    # Model name to use for RULER rescoring (LLM-as-a-judge)
    ruler_judge_model: str = "openrouter/openai/o4-mini"
    minimum_reward_std_dev: float = 0.0
    # Random seed to control which subset of the training data is sampled
    training_dataset_seed: int | None = None

    # Fork configuration
    fork_from_model: str | None = None
    fork_from_project: str | None = None
    fork_not_after_step: int | None = None

    # Training configuration
    scale_rewards: bool = True


async def register_model():
    backend = await SkyPilotBackend().initialize_cluster(
        cluster_name="test_launch",
        gpu="H100-SXM",
        env_path=".env",
        # force_restart=True,
    )

    model = art.TrainableModel(
        name="mcp-nws-14b-001",
        project="mcp-smithery",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        # _internal_config=art.dev.InternalModelConfig(
        #     init_args=art.dev.InitArgs(
        #         max_seq_length=16384,
        #     ),
        # ),
        config=McpPolicyConfig(
            num_epochs=160,
            smithery_mcp_url="asdf",
            # trajectories_per_group=2,
            # groups_per_step=1,
        ),
    )

    await backend.register(model)

    print("model registered")


if __name__ == "__main__":
    asyncio.run(register_model())

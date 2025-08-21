import asyncio
import copy
import os
import random

import weave
from dotenv import load_dotenv

import art
from art.rewards.ruler import ruler_score_group
from art.skypilot import SkyPilotBackend
from park_ranger.rollout import rollout
from park_ranger.scenarios import ParkRangerScenario, val_scenarios

load_dotenv()

random.seed(42)

# Initialize the server
backend = None


# comparison models
gpt_4o_mini = art.Model(
    name="gpt-4o-mini",
    project="park-ranger",
    inference_model_name="openai/gpt-4o-mini",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)

gpt_4o = copy.deepcopy(gpt_4o_mini)
gpt_4o.name = "gpt-4o"
gpt_4o.inference_model_name = "openai/gpt-4o"

gpt_4_1 = copy.deepcopy(gpt_4o_mini)
gpt_4_1.name = "gpt-4.1"
gpt_4_1.inference_model_name = "openai/gpt-4.1"

grok_3 = copy.deepcopy(gpt_4o_mini)
grok_3.name = "grok-3"
grok_3.inference_model_name = "x-ai/grok-3"

grok_4 = copy.deepcopy(gpt_4o_mini)
grok_4.name = "grok-4"
grok_4.inference_model_name = "x-ai/grok-4"

sonnet_4 = copy.deepcopy(gpt_4o_mini)
sonnet_4.name = "sonnet-4"
sonnet_4.inference_model_name = "anthropic/claude-sonnet-4"

gemini_2_5_pro = copy.deepcopy(gpt_4o_mini)
gemini_2_5_pro.name = "gemini-2.5-pro"
gemini_2_5_pro.inference_model_name = "google/gemini-2.5-pro"

r1_0528 = copy.deepcopy(gpt_4o_mini)
r1_0528.name = "r1-0528"
r1_0528.inference_model_name = "deepseek/deepseek-r1-0528:free"


async def generate_val_groups(
    model: art.Model, val_scenarios: list[ParkRangerScenario]
) -> list[art.TrajectoryGroup]:
    groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(rollout(model, val_scenarios[i]) for _ in range(4))
            for i in range(len(val_scenarios))
        ),
        pbar_desc=f"gather {model.name}",
        max_exceptions=1,
    )

    return groups


async def calculate_beat_comp(
    groups: list[art.TrajectoryGroup],
    control_groups: list[art.TrajectoryGroup],
    control_first: bool = True,
):
    promises = []

    if control_groups is not None:
        for i in range(len(groups)):
            for j in range(len(groups[i].trajectories)):
                trajectories = [
                    control_groups[i].trajectories[j],
                    groups[i].trajectories[j],
                ]
                group = art.TrajectoryGroup(
                    trajectories if control_first else reversed(trajectories)
                )

                async def score_group(group_idx: int, trajectory_idx: int):
                    scored_group = await ruler_score_group(
                        group,
                        judge_model="openai/o4-mini",
                        debug=True,
                    )

                    if control_first:
                        control_score = scored_group.trajectories[0].reward
                        benchmark_score = scored_group.trajectories[1].reward
                    else:
                        benchmark_score = scored_group.trajectories[0].reward
                        control_score = scored_group.trajectories[1].reward

                    reward_diff = benchmark_score - control_score

                    metric_name = (
                        "beat_comp" if control_first else "beat_comp_control_last"
                    )

                    if reward_diff > 0.1:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 1
                    elif reward_diff < -0.1:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 0
                    else:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 0.5

                promises.append(score_group(i, j))

    await asyncio.gather(*promises)


async def log_comparison_model(
    comparison_model: art.Model,
    val_scenarios: list[ParkRangerScenario],
    control_groups: list[art.TrajectoryGroup] | None = None,
) -> list[art.TrajectoryGroup]:
    groups = await generate_val_groups(comparison_model, val_scenarios)

    if control_groups is not None:
        await calculate_beat_comp(groups, control_groups, control_first=True)
        await calculate_beat_comp(groups, control_groups, control_first=False)

    await comparison_model.log(
        groups,
        split="val",
    )
    await backend._experimental_push_to_s3(
        comparison_model,
    )

    return groups


async def run_benchmarks():
    global backend
    backend = await SkyPilotBackend.initialize_cluster(
        cluster_name="park-ranger",
        gpu="H100-SXM",
        tail_logs=False,
        env_path="../../.env",
        art_version="../../",
        force_restart=True,
    )
    await gpt_4o_mini.register(backend)
    await gpt_4o.register(backend)
    await gpt_4_1.register(backend)
    await grok_3.register(backend)
    await grok_4.register(backend)
    await sonnet_4.register(backend)
    await gemini_2_5_pro.register(backend)
    await r1_0528.register(backend)

    weave.init("park-ranger")

    control_groups = await generate_val_groups(gpt_4_1, val_scenarios)

    for comparison_model in [
        gpt_4o_mini,
        gpt_4o,
        gpt_4_1,
        grok_3,
        grok_4,
        sonnet_4,
        gemini_2_5_pro,
        r1_0528,
    ]:
        await log_comparison_model(comparison_model, val_scenarios, control_groups)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())

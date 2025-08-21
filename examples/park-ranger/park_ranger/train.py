import asyncio
import os

import weave
from dotenv import load_dotenv

import art
from art.rewards.ruler import ruler_score_group
from art.utils import iterate_dataset
from park_ranger.experiments import ParkRangerConfig, models
from park_ranger.generate_benchmarks import calculate_beat_comp, generate_val_groups
from park_ranger.rollout import rollout
from park_ranger.scenarios import train_scenarios, val_scenarios

load_dotenv()

os.environ["WEAVE_LOG_LEVEL"] = "CRITICAL"

weave.init(project_name="park-ranger")

gpt_4_1 = art.Model(
    name="openai/gpt-4.1",
    project="park-ranger",
    inference_model_name="openai/gpt-4.1",
    inference_api_key=os.environ["OPENROUTER_API_KEY"],
    inference_base_url="https://openrouter.ai/api/v1",
)


async def train(
    model: art.TrainableModel[ParkRangerConfig], use_skypilot: bool = False
):
    if use_skypilot:
        from art.skypilot.backend import SkyPilotBackend

        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name="park-ranger",
            gpu="H100-SXM",
            tail_logs=False,
            env_path="../../.env",
            force_restart=True,
            art_version="../../",
        )
    else:
        from art.local import LocalBackend

        backend = LocalBackend()

    print(f"Pulling latest checkpoint from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
    await backend._experimental_pull_from_s3(
        model,
        s3_bucket=os.environ["BACKUP_BUCKET"],
        verbose=True,
        only_step="latest",  # Only pull the latest checkpoint
    )

    await model.register(backend)

    print(f"Training data size: {len(train_scenarios)}")
    print(f"Validation data size: {len(val_scenarios)}")

    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=model.config.groups_per_step,
        num_epochs=model.config.num_epochs,
        initial_step=await model.get_step(),
    )

    control_groups = await generate_val_groups(gpt_4_1, val_scenarios)

    # Main training loop using iterate_dataset
    for batch in train_iterator:
        print("Gathering trajectory groups with RULER scoring...")

        # Use gather_trajectory_groups with ruler_score_group
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario, False)
                    for _ in range(model.config.trajectories_per_group)
                )
                for scenario in batch.items
            ),
            pbar_desc=f"train gather step {batch.step}",
            after_each=lambda group: ruler_score_group(
                group,
                judge_model="openai/o4-mini",
                debug=True,  # Show judge reasoning
                swallow_exceptions=True,
            ),
        )

        print("train groups finished")

        if batch.step % model.config.eval_steps == 0:
            print("starting comparison val gather")
            val_groups = await generate_val_groups(model, val_scenarios)
            await calculate_beat_comp(val_groups, control_groups, control_first=True)
            await calculate_beat_comp(val_groups, control_groups, control_first=False)

            await model.log(val_groups, split="val")

        print("starting train")
        await model.train(
            groups, config=art.TrainConfig(learning_rate=model.config.learning_rate)
        )

        await backend._experimental_push_to_s3(
            model,
        )

    print("Training finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name of the model to train")
    parser.add_argument("--use-skypilot", action="store_true", help="Use Skypilot")
    args = parser.parse_args()

    model = models[args.model]

    asyncio.run(train(model=model, use_skypilot=args.use_skypilot))

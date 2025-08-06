import argparse
import json
import textwrap
import concurrent.futures
import traceback
import sky
from dotenv import dotenv_values
from sky import ClusterStatus
import art

trainable_models = {
    "001": art.TrainableModel(
        name="overfitting-001",
        project="overfitting_experiments",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        _internal_config=art.dev.InternalModelConfig(
            engine_args=art.dev.EngineArgs(gpu_memory_utilization=0.7),
        ),
    ),
}

parser = argparse.ArgumentParser(
    description="Train overfitting experiment models on RunPod."
)
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Comma-separated list of model keys to train (e.g. 001,002,003).",
)
parser.add_argument(
    "--use-cluster-name",
    type=str,
    required=False,
    help="Use a specific cluster name for the task.",
)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Whether to use fast launch (skip setup).",
)
args = parser.parse_args()

requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
unknown = [m for m in requested_models if m not in trainable_models]
if unknown:
    raise ValueError(
        f"Unknown model keys requested: {', '.join(unknown)}. Valid keys: {', '.join(trainable_models.keys())}"
    )


def launch_model(model_key: str):
    trainable_model = trainable_models[model_key]
    print(f"Launching {model_key} on SkyPilot…")

    model_json = json.dumps(trainable_model.model_dump())

    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt update && apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env

            # Install project in editable mode
            uv remove openpipe-art
            uv add --editable ~/ART --extra backend --extra plotting

            # Sync dependencies
            uv sync
        """
    )

    run_script = textwrap.dedent(
        f"""
        # Run the overfitting experiment
        uv remove openpipe-art
        uv add --editable ~/ART --extra backend --extra plotting

        echo '{model_json}' > model_config.json
        uv run yes_no_maybe.py
    """
    )

    task = sky.Task(
        name=f"overfitting-expt-{model_key}",
        setup=setup_script,
        run=run_script,
        workdir=".",
        envs=dict(dotenv_values()),
    )

    num_gpus = 1
    if trainable_model._internal_config is not None:
        num_gpus = trainable_model._internal_config.get("engine_args", {}).get(
            "tensor_parallel_size", 1
        )

    task.set_resources(
        sky.Resources(
            accelerators=f"H200-SXM:{num_gpus}",
            cloud=sky.clouds.RunPod(),
            region="US",
        )
    )
    task.set_file_mounts({"~/ART": "../.."})

    cluster_name = args.use_cluster_name or f"overfitting-expt-{model_key}"
    print(f"Launching task on cluster: {cluster_name}")

    print("Checking for existing cluster and jobs…")
    cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
    if len(cluster_status) > 0 and cluster_status[0]["status"] == ClusterStatus.UP:
        print(f"Cluster {cluster_name} is UP. Canceling any active jobs…")
        sky.stream_and_get(sky.cancel(cluster_name, all=True))

    job_id, _ = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=60,
            down=True,
            fast=args.fast,
        )
    )

    print(f"Job submitted for {model_key} (ID: {job_id}). Streaming logs…")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} for {model_key} finished with exit code {exit_code}.")


with concurrent.futures.ThreadPoolExecutor(
    max_workers=len(requested_models)
) as executor:
    futures = [executor.submit(launch_model, key) for key in requested_models]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            print(f"Traceback: {traceback.format_exc()}")

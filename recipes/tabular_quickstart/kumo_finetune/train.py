"""Fine-tune NVIDIA Kumo Tabular on a sklearn dataset with Cortex Training.

Each step samples context and query rows from the training split, preprocesses them here with sdm, and trains one
network call: classification with ``weighted_cross_entropy``, regression with a pinball loss computed here and sent
through ``output_gradient``. Validation runs the job's no-gradient forward on an sdm ensemble with the whole training
split as context. At the end the weights-only checkpoint is exported, loaded into sdm, and checked against the job.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import chz
import torch
from recipes.tabular_quickstart.kumo_finetune import task as tables
from recipes.utils import make_client
from recipes.utils import optimizer_step
from recipes.utils import running_job
from recipes.utils import save_checkpoint

from cortex_training.client import DEBUG_OPTIONS_ENV
from cortex_training.client import tabular

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None

    model_name: str = "nvidia/Kumo-Tabular"
    task: str = "classification"
    size: str = "small"
    # A directory on the GPU node in the nvidia/Kumo-Tabular layout to load instead of the model directory.
    weights_dir: str | None = None
    hardware: str = "h200"
    debug_image_tag: str | None = None

    max_epochs: int = 10
    steps_per_epoch: int = 30
    context_size: int = 256
    query_size: int = 128
    lr: float = 1e-5
    num_estimators: int = 8
    seed: int = 0
    verify_export: bool = True


def job_body(config: Config) -> dict[str, Any]:
    options: dict[str, Any] = {"adapter": "kumo_tabular", "task": config.task, "size": config.size}
    if config.weights_dir:
        options["weights_dir"] = config.weights_dir
    body: dict[str, Any] = {
        "sub_job_configs": [
            {
                "job_type": "training",
                "model_name": config.model_name,
                "seed": config.seed,
                "training_config": {
                    "model_provider": "tabular",
                    "tabular": options,
                    "n_gpus": 1,
                    "max_seq_len": 1,
                    "train_batch_size": 1,
                    "gradient_clipping": 0.0,
                    "optimizer": {"name": "AdamW", "lr": config.lr, "weight_decay": 0.0},
                },
            }
        ],
    }
    if config.hardware.upper() != "H200":
        body["hardware"] = config.hardware.upper()
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
        # The tag is this run's explicit opt-in to the create-job debug options.
        os.environ[DEBUG_OPTIONS_ENV] = "1"
    return body


def pinball(levels: int) -> Any:
    quantile_levels = torch.linspace(0.001, 0.999, levels)

    def loss(outputs: list[torch.Tensor], y_query: torch.Tensor) -> torch.Tensor:
        diff = y_query.reshape(-1, 1, 1) - outputs[0]
        return torch.maximum(quantile_levels * diff, (quantile_levels - 1) * diff).mean()

    return loss


def forward(client: Any, job_id: str, calls: list[tabular.Call]) -> list[torch.Tensor]:
    request_id = client.forward(job_id, tabular.build_forward_payload(calls), sub_job_type="training")["request_id"]
    return tabular.outputs_from_forward_result(client.poll_request(job_id, request_id))


def validate(client: Any, job_id: str, splits: tables.Splits, config: Config, query: Any):
    encoded = tables.encode(
        tables.eval_recipe(),
        splits.train,
        query,
        splits.target,
        num_estimators=config.num_estimators,
        generator=torch.Generator().manual_seed(config.seed),
    )
    outputs = torch.stack([out[:, 0] for out in forward(client, job_id, encoded.calls())])  # [E, R_query, K]
    return tables.metric(splits, tables.decode(encoded, outputs), query), encoded, outputs


def train_step(client: Any, job_id: str, splits: tables.Splits, encoded: tables.EncodedTables, query: Any) -> float:
    calls = encoded.calls()
    y_query = tables.query_targets(encoded, query, splits.target)
    if splits.task == "classification":
        payload = tabular.build_forward_backward_payload(calls, [y_query.unsqueeze(1)], num_classes=splits.num_classes)
        return float(client.poll_request(job_id, client.forward_backward(job_id, payload))["avg_loss"])
    outputs = forward(client, job_id, calls)
    loss_fn = pinball(outputs[0].shape[-1])
    loss, grads = tabular.output_gradients(outputs, lambda leaves: loss_fn(leaves, y_query))
    client.poll_request(job_id, client.forward_backward(job_id, tabular.build_output_gradient_payload(calls, grads)))
    return loss


def verify_export(client: Any, job_id: str, checkpoint_id: str, config: Config, encoded, outputs) -> float:
    """Largest gap between the job's outputs and sdm's on the exported weights, for the same tables."""
    import sdm
    from safetensors.torch import load_file

    files = {f["filename"].rsplit("/", 1)[-1]: f["url"] for f in client.export_checkpoint(job_id, checkpoint_id)["files"]}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.safetensors"
        with urllib.request.urlopen(files["model.safetensors"], timeout=300) as response:
            path.write_bytes(response.read())
        network = sdm.models.KumoTabular(task=config.task, size=config.size, pretrained=False, device="cpu")
        network = network.models[config.task]
        network.load_state_dict(load_file(path), strict=True)
    network.eval()
    gaps = []
    with torch.no_grad():
        for x, y, mask, out in zip(encoded.x, encoded.y, encoded.categorical_mask, outputs, strict=True):
            local = network(x=x, y=y if config.task == "classification" else y.float(), categorical_mask=mask)
            gaps.append(float((local.float() - out).abs().max()))
    return max(gaps)


def main(config: Config) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    splits = tables.load_splits(config.task, seed=config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    metric_name = "acc" if config.task == "classification" else "rmse"

    client = make_client(config.config, config.connection)
    history = []
    with running_job(client, job_body(config), job_id=config.job_id) as job_id:
        val_metric, _, _ = validate(client, job_id, splits, config, splits.val)
        logger.info("epoch=0/%d val_%s=%.4f", config.max_epochs, metric_name, val_metric)
        history.append({"epoch": 0, f"val_{metric_name}": val_metric})
        for epoch in range(1, config.max_epochs + 1):
            start, total_loss = time.time(), 0.0
            for _ in range(config.steps_per_epoch):
                index = torch.randperm(splits.train.size(0), generator=generator)
                index = index[: config.context_size + config.query_size]
                context = splits.train[index[: config.context_size]]
                query = splits.train[index[config.context_size :]]
                encoded = tables.encode(tables.train_recipe(), context, query, splits.target, generator=generator)
                total_loss += train_step(client, job_id, splits, encoded, query)
                optimizer_step(client, job_id, learning_rate=config.lr)
            val_metric, _, _ = validate(client, job_id, splits, config, splits.val)
            train_loss = total_loss / config.steps_per_epoch
            logger.info(
                "epoch=%d/%d val_%s=%.4f train_loss=%.4f time=%.1fs",
                epoch, config.max_epochs, metric_name, val_metric, train_loss, time.time() - start,
            )
            history.append({"epoch": epoch, f"val_{metric_name}": val_metric, "train_loss": train_loss})

        test_metric, encoded, outputs = validate(client, job_id, splits, config, splits.test)
        logger.info("test_%s=%.4f", metric_name, test_metric)
        saved = save_checkpoint(client, job_id, checkpoint_type="weights-only")
        export_gap = None
        if config.verify_export:
            export_gap = verify_export(client, job_id, saved["checkpoint_id"], config, encoded, outputs)
            logger.info("exported checkpoint %s: max |sdm - job| output gap %.3g", saved["checkpoint_id"], export_gap)

    return {
        "job_id": job_id,
        "checkpoint_id": saved["checkpoint_id"],
        f"test_{metric_name}": test_metric,
        "export_max_abs_gap": export_gap,
        "history": history,
    }


if __name__ == "__main__":
    chz.nested_entrypoint(main)

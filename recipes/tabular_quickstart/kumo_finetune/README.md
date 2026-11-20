# Fine-tune Kumo Tabular

[NVIDIA Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular) is a tabular
foundation model: given labeled context rows and unlabeled query rows of one
table, it predicts the query targets in a single forward pass. This recipe
fully fine-tunes the small model (about 28M parameters) on a sklearn dataset,
following the
[sdm fine-tuning example](https://github.com/NVIDIA/structured-data-models/blob/main/examples/tabular/finetune.py),
through a `tabular` job with adapter `kumo_tabular`.

## Install

```bash
uv pip install -e . chz structured-data-models scikit-learn pandas
```

## Run

```bash
python -m recipes.tabular_quickstart.kumo_finetune.train connection=training task=classification
python -m recipes.tabular_quickstart.kumo_finetune.train connection=training task=regression
```

`task=classification` trains on digits with `weighted_cross_entropy`.
`task=regression` trains on California housing: each step runs a forward pass,
computes the pinball loss over the 999 quantiles here, and sends its gradient
with `output_gradient`. sdm preprocesses every table here. After every epoch,
validation runs the job's forward pass on an `num_estimators`-member sdm
ensemble (one network call per member) with the whole training split as
context. At the end the recipe saves a weights-only checkpoint, downloads its
`model.safetensors`, loads it into sdm, and logs the largest gap between sdm's
outputs and the job's.

| Option | Default | Meaning |
|---|---|---|
| `task` | `classification` | `classification` (digits) or `regression` (California housing) |
| `size` | `small` | `small`, `medium` or `large` |
| `model_name` | `nvidia/Kumo-Tabular` | The job's model |
| `weights_dir` | none | A directory on the GPU node in the `nvidia/Kumo-Tabular` layout to load instead |
| `max_epochs` | `10` | Epochs of `steps_per_epoch` steps |
| `steps_per_epoch` | `30` | Training steps per epoch |
| `context_size`, `query_size` | `256`, `128` | Rows sampled per step |
| `lr` | `1e-5` | AdamW learning rate |
| `num_estimators` | `8` | sdm ensemble members for validation |
| `verify_export` | `true` | Download the checkpoint and compare sdm's outputs with the job's |

The regression loss is computed on the standardized target, so its value is the
sdm example's divided by the context target's standard deviation.

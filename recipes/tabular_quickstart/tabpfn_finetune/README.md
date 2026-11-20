# Fine-tune TabPFN v2

Fine-tunes the [TabPFN v2](https://github.com/PriorLabs/TabPFN) classifier on
sklearn digits through a `tabular` job (adapter `tabpfn_v2`). Episodes are
sampled and preprocessed the way TabPFN's own `FinetunedTabPFNClassifier`
does; each ensemble estimator is one network call. Held-out log loss and
accuracy are scored with the job's `forward` before and after training.

## Install

```bash
uv pip install -e . "tabpfn==10.0.0" scikit-learn
```

## Run

```bash
python -m recipes.tabular_quickstart.tabpfn_finetune.train_remote connection=training steps=30
python -m recipes.tabular_quickstart.tabpfn_finetune.train_remote connection=training steps=15 mode=output_gradient
```

`mode=weighted_cross_entropy` (the default) has the server train mean
cross-entropy over the query rows. `mode=output_gradient` runs a forward pass,
computes the same loss here, and sends its gradient: two round trips per step,
but any loss works.

| Option | Default | Meaning |
|---|---|---|
| `steps` | `30` | Optimizer steps |
| `lr` | `3e-5` | AdamW learning rate |
| `n_estimators` | `2` | Ensemble estimators (network calls) per episode |
| `episode_rows` | `400` | Rows per episode |
| `model_name` | `Prior-Labs/TabPFN-v2-clf` | The job's model |

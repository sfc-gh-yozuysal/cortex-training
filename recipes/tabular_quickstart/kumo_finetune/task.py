"""Kumo Tabular tables on Cortex Training: datasets, client-side preprocessing, and the tensors the job takes.

The server runs only the Kumo Tabular network. ``structured-data-models`` (``sdm``) preprocesses each table here, as
its ``KumoTabular`` wrapper would before calling the network, and turns the network's outputs back into predictions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import sdm
import sdm.processing as sp
import torch
from sdm.processing.execution import RecipeExecution

from cortex_training.client import tabular

TASKS = ("classification", "regression")


@dataclass(frozen=True)
class Splits:
    """A dataset's 60/20/20 train/validation/test tables and its target column."""

    task: str
    target: str
    train: Any
    val: Any
    test: Any

    @property
    def num_classes(self) -> int | None:
        if self.task != "classification":
            return None
        return int(self.train[self.target].categorical.categories[0].numel())


def load_splits(task: str, seed: int = 0) -> Splits:
    """sklearn's digits (classification) or California housing (regression), split as the sdm fine-tuning example."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    if task == "classification":
        df = load_digits(as_frame=True).frame
        target = "target"
        stypes = sdm.infer_stypes(df, overrides={target: "categorical"})
    elif task == "regression":
        df = fetch_california_housing(as_frame=True).frame
        target = "MedHouseVal"
        stypes = sdm.infer_stypes(df)
    else:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    table = sdm.TableTensor.from_pandas(df, stypes=stypes)
    stratify = df[target] if task == "classification" else None
    train_index, rest_index = train_test_split(range(len(df)), test_size=0.4, random_state=seed, stratify=stratify)
    val_index, test_index = train_test_split(
        rest_index,
        test_size=0.5,
        random_state=seed,
        stratify=df[target].iloc[rest_index] if task == "classification" else None,
    )
    pick = lambda index: table[torch.as_tensor(index)]  # noqa: E731
    return Splits(task=task, target=target, train=pick(train_index), val=pick(val_index), test=pick(test_index))


def train_recipe() -> sp.Recipe:
    """The default recipe without target flipping or output post-processing, so the loss sees raw network outputs."""
    recipe = sdm.models.KumoTabular.default_recipe()
    recipe.target = sp.StypeDispatch(
        categorical=[sp.AlignCategories(), sp.ShuffleCategories(method="shift")],
        numerical=sp.Standardize(),
    )
    recipe.output = sp.Identity()
    return recipe


def eval_recipe() -> sp.Recipe:
    return sdm.models.KumoTabular.default_recipe()


@dataclass
class EncodedTables:
    """One preprocessed table per ensemble member, and what is needed to read the network's outputs back."""

    x: list[torch.Tensor]  # [R, C] per member, context rows first
    y: list[torch.Tensor]  # [R_context] per member
    categorical_mask: list[torch.Tensor]  # [C] per member
    classes: list[torch.Tensor | None]  # class labels in each member's code order, None for regression
    execution: RecipeExecution

    def calls(self) -> list[tabular.Call]:
        """One network call per member, in the shared layout: ``x`` ``[R, 1, C]``, ``y`` ``[R_context, 1]``.

        Members can keep different columns, so each is its own call.
        """
        return [
            tabular.Call(
                x=x.unsqueeze(1),
                y=y.float().unsqueeze(1),
                categorical_inds=[mask.nonzero().flatten().tolist()],
            )
            for x, y, mask in zip(self.x, self.y, self.categorical_mask, strict=True)
        ]


def encode(
    recipe: sp.Recipe,
    context: Any,
    query: Any,
    target: str,
    *,
    num_estimators: int | None = None,
    generator: torch.Generator | None = None,
) -> EncodedTables:
    """Fit ``recipe`` on the context rows and transform both, as ``KumoTabular`` does before calling its network."""
    from sdm import Stype

    execution = RecipeExecution(copy.deepcopy(recipe))
    with torch.no_grad():
        members = execution.fit_transform(
            x=context.drop_columns(target),
            y=context[target],
            related_tables=None,
            num_members=num_estimators,
            generator=generator,
        )
        queries = execution.transform(x=query.drop_columns(target), related_tables=None)
    encoded = EncodedTables(x=[], y=[], categorical_mask=[], classes=[], execution=execution)
    for member, member_query in zip(members, queries, strict=True):
        encoded.x.append(torch.cat([member.x.numerical, member_query.x.numerical], dim=-2).float())
        encoded.categorical_mask.append(
            torch.tensor(
                [member.input_stypes.get(column) == Stype.categorical for column in member.x.columns[Stype.numerical]],
                dtype=torch.bool,
            )
        )
        if member.y.categorical.size(-1) > 0:
            encoded.y.append(member.y.categorical.code.squeeze(-1).long())
            encoded.classes.append(member.y.categorical.categories[0])
        else:
            encoded.y.append(member.y.numerical.squeeze(-1).float())
            encoded.classes.append(None)
    return encoded


def query_targets(encoded: EncodedTables, query: Any, target: str) -> torch.Tensor:
    """The query rows' targets in the network's space: class indices in each member's order, or transformed floats."""
    from sdm.processing.execution import _to_ensemble_table

    if encoded.classes[0] is not None:
        labels = query[target].categorical
        values = labels.categories[0][labels.code.squeeze(-1)]
        rows = []
        for classes in encoded.classes:
            index = {value: i for i, value in enumerate(classes.tolist())}
            rows.append(torch.tensor([index[value] for value in values.tolist()], dtype=torch.long))
        return rows[0] if len(rows) == 1 else torch.stack(rows)
    with torch.no_grad():
        transformed = encoded.execution.recipe.target.transform_ensemble(
            _to_ensemble_table(query[target], len(encoded.x), expand=True)
        )
    rows = [transformed[i].numerical.squeeze(-1).float() for i in range(len(encoded.x))]
    return rows[0] if len(rows) == 1 else torch.stack(rows)


def decode(encoded: EncodedTables, outputs: torch.Tensor) -> Any:
    """Predictions from the network's outputs (``[E, R_query, K]`` or ``[R_query, K]``), as ``KumoTabular`` returns."""
    from sdm import Stype
    from sdm import TableTensor

    if outputs.dim() == 2:
        outputs = outputs.unsqueeze(0)
    members = []
    for member_outputs, classes in zip(outputs, encoded.classes, strict=True):
        if classes is None:
            columns = [f"q{i:03d}" for i in range(1, member_outputs.shape[-1] + 1)]
            members.append(TableTensor(columns={Stype.numerical: columns}, numerical=member_outputs))
        else:
            columns = [str(value) for value in classes.tolist()]
            members.append(
                TableTensor(columns={Stype.numerical: columns}, numerical=member_outputs[..., : len(columns)])
            )
    with torch.no_grad():
        if encoded.classes[0] is None:
            members = list(encoded.execution.inverse_transform_target(members))
        return encoded.execution.transform_output(members)


def metric(splits: Splits, predictions: Any, query: Any) -> float:
    """Accuracy (classification) or RMSE (regression) of ``predictions`` on ``query``, as the sdm example scores."""
    if splits.task == "classification":
        prob, y = sdm.evaluation.to_class_indices(predictions, query[splits.target], missing_score=0.0)
        return float((prob.argmax(dim=-1) == y).float().mean())
    pred = predictions.numerical.mean(dim=-1)
    y = query[:, splits.target].numerical.squeeze(-1)
    return float((pred - y).pow(2).mean().sqrt())

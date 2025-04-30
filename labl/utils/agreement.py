import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
from krippendorff import alpha
from krippendorff.krippendorff import LevelOfMeasurement, ValueScalarType
from scipy.stats import kendalltau, pearsonr, spearmanr

logger = logging.getLogger(__name__)

AgreementType = Literal[
    "krippendorff_nominal",
    "krippendorff_ordinal",
    "krippendorff_interval",
    "krippendorff_ratio",
    "krippendorff_custom",
]
CorrelationType = Literal[
    "pearsonr",
    "spearmanr",
    "kendalltau",
]
MetricType = AgreementType | CorrelationType


class MetricFunction(Protocol):
    def __call__(self, data: npt.NDArray[ValueScalarType], **kwargs) -> float: ...


@dataclass
class MetricOutput:
    """Data class for storing the output of metric computations.

    Attributes:
        score (float | None): The full agreement for all annotation sets. If a global agreement is not defined for the
            metric, this is set to the average of non-diagonal elements of the pairwise score matrix.
        scores_pairs (list[list[float]]): Pairwise scores between label sets.
        type (str): The type of metric employed.
    """

    score: float | None
    scores_pairs: list[list[float]]
    type: MetricType

    def __str__(self) -> str:
        pairs_str = "    | " + " | ".join(f"A{i:<3}" for i in range(len(self.scores_pairs))) + " |\n"
        for idx_row, row in enumerate(self.scores_pairs):
            pairs_str += (
                f"A{idx_row:<2} | "
                + " | ".join(
                    f"{round(x, 2):<4}" if idx_col != idx_row else f"{' ':<4}" for idx_col, x in enumerate(row)
                )
                + " |\n"
            )
        pairs_str = pairs_str.replace("\n", "\n" + " " * 16)
        return dedent(f"""\
        MetricOutput(
            type: {self.type},
            score: {round(self.score, 4) if self.score is not None else None},
            scores_pairs:
                {pairs_str}
        )
        """)


def _get_unique_labels(labels_array: npt.NDArray[ValueScalarType]) -> npt.NDArray[ValueScalarType]:
    if labels_array.dtype.kind in {"i", "u", "f"}:
        unique_vals = np.unique(labels_array[~np.isnan(labels_array)])
    elif labels_array.dtype.kind in {"U", "S"}:  # Unicode or byte string.
        # `np.asarray` will coerce `np.nan` values to "nan".
        unique_vals = np.unique(labels_array[labels_array != "nan"])
    else:
        raise ValueError(
            f"Unsupported label type: {labels_array.dtype}. Please specify the level of measurement explicitly."
        )
    return unique_vals


def _has_multiple_labels(labels_array: npt.NDArray[ValueScalarType]) -> bool:
    return len(_get_unique_labels(labels_array)) > 1


def _compute_pairwise_scores(
    score_fn: MetricFunction,
    labels_array: npt.NDArray[ValueScalarType],
    label_type: type,
    is_symmetric: bool = True,
    score_fn_kwargs: dict[str, Any] = {},
) -> list[list[float]]:
    """Compute pairwise scores for a given metric function.
    Args:
        score_fn (MetricFunction): The metric function to compute the scores.
        labels_array (npt.NDArray[ValueScalarType]): The array of labels.
        is_symmetric (bool): Whether the metric is symmetric. Default: True.
        score_fn_kwargs (dict): Additional arguments for the metric function.
    Returns:
        The pairwise scores.
    """
    num_annotators = labels_array.shape[0]
    pair_scores = np.identity(num_annotators)
    for i in range(num_annotators):
        for j in range(i if is_symmetric else 0, num_annotators):
            if i == j or np.array_equal(
                labels_array[i, :], labels_array[j, :], equal_nan=True if label_type is float else False
            ):
                pair_score = 1.0
            else:
                pair_score = score_fn(
                    labels_array[[i, j], :],
                    **score_fn_kwargs,
                )
            pair_scores[i, j] = pair_score
            if is_symmetric:
                pair_scores[j, i] = pair_score
    pair_scores = cast(list[list[float]], pair_scores.tolist())
    return pair_scores


def _compute_nondiag_mean(pair_scores: list[list[float]]) -> float:
    return np.array(pair_scores)[~np.eye(len(pair_scores), dtype=bool)].mean()


def compute_agreement(
    label_type: type,
    labels_array: npt.NDArray[ValueScalarType],
    level_of_measurement: LevelOfMeasurement | None = None,
) -> MetricOutput:
    """Compute the inter-annotator agreement using
    [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) for an (M, N) array of labels,
    where M is the number of annotators and N is the number of units.

    Args:
        level_of_measurement (Literal['nominal', 'ordinal', 'interval', 'ratio']): The level of measurement for the
            labels when using Krippendorff's alpha. Can be "nominal", "ordinal", "interval", or "ratio", depending
            on the type of labels. Default: "nominal" for string labels, "ordinal" for int labels, and "interval"
            for float labels.

    Returns:
        Inter-annotator agreement (for categorical) between the two entries
    """
    if level_of_measurement is None:
        if label_type is str:
            level_of_measurement = "nominal"
        elif label_type is int:
            level_of_measurement = "ordinal"
        elif label_type is float:
            level_of_measurement = "interval"
        else:
            raise ValueError(
                f"Unsupported label type: {label_type}. Please specify the level of measurement explicitly."
            )
    if not _has_multiple_labels(labels_array):
        raise RuntimeError(
            "A single non-empty label is present, hence agreement cannot be computed."
            "Relabel your data setting `None` values to some label to compute agreement between presence/absence of label."
        )
    full_score = alpha(reliability_data=labels_array, level_of_measurement=level_of_measurement)
    pair_scores = _compute_pairwise_scores(
        alpha,  # type: ignore
        labels_array,
        label_type,
        is_symmetric=True,
        score_fn_kwargs={"level_of_measurement": level_of_measurement},
    )
    agreement_type = "krippendorff_" + (level_of_measurement if isinstance(level_of_measurement, str) else "custom")
    return MetricOutput(score=full_score, scores_pairs=pair_scores, type=agreement_type)


def compute_correlation(
    label_type: type,
    labels_array: npt.NDArray[ValueScalarType],
    correlation_type: CorrelationType | None = None,
    correlation_kwargs: dict[str, Any] = {},
) -> MetricOutput:
    """Compute the correlation between two or more sets of labels.

    Args:
        label_type (type): The type of the labels. Can be str (only if binary), int, or float.
        labels_array (npt.NDArray[ValueScalarType]): The array of labels.
        correlation_type (str): The type of correlation to compute. Can be "pearsonr", "spearmanr", or "kendalltau".
            Default: "spearmanr".

    Returns:
        The correlation between the two sets of labels.
    """
    if correlation_type is None:
        correlation_type = "spearmanr"
    if label_type is str:
        unique_vals = _get_unique_labels(labels_array)
        if len(unique_vals) > 1:
            raise ValueError("Correlation is not defined for string labels.")
        logger.warning(
            "Converting string labels to binary values for correlation computation. "
            "This may not be meaningful for your data."
        )
        labels_array = (labels_array == unique_vals[0]).astype(int)

    def corr_fn(data: npt.NDArray[ValueScalarType], **kwargs) -> float:
        if correlation_type == "pearsonr":
            metric_fn = pearsonr
        elif correlation_type == "spearmanr":
            metric_fn = spearmanr
        elif correlation_type == "kendalltau":
            metric_fn = kendalltau
        else:
            raise ValueError(f"Unsupported correlation type: {correlation_type}.")
        return metric_fn(data[0], data[1], **kwargs).correlation  # type: ignore

    pair_scores = _compute_pairwise_scores(
        corr_fn,  # type: ignore
        labels_array,
        label_type,
        is_symmetric=True,
        score_fn_kwargs=correlation_kwargs,
    )
    return MetricOutput(score=_compute_nondiag_mean(pair_scores), scores_pairs=pair_scores, type=correlation_type)

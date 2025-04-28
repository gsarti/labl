import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from krippendorff import alpha
from krippendorff.krippendorff import LevelOfMeasurement, ValueScalarType
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

MeasurementType = Literal[
    "krippendorff_nominal",
    "krippendorff_ordinal",
    "krippendorff_interval",
    "krippendorff_ratio",
    "krippendorff_custom",
    "spearmanr_binary",
]


@dataclass
class AgreementOutput:
    """Data class for storing the output of the inter-annotator agreement computation.

    Attributes:
        full (float | None): The full agreement for all annotation sets.
        pair (list[list[float]]): Pairwise agreement between all annotators.
        type (str): The type of agreement measure employed.
    """

    full: float | None
    pair: list[list[float]]
    type: MeasurementType

    def __str__(self) -> str:
        pairs_str = "    | " + " | ".join(f"A{i:<3}" for i in range(len(self.pair))) + " |\n"
        for idx_row, row in enumerate(self.pair):
            pairs_str += (
                f"A{idx_row:<2} | "
                + " | ".join(
                    f"{round(x, 2):<4}" if idx_col != idx_row else f"{' ':<4}" for idx_col, x in enumerate(row)
                )
                + " |\n"
            )
        pairs_str = pairs_str.replace("\n", "\n" + " " * 16)
        return dedent(f"""\
        AgreementOutput(
            type: {self.type},
            full: {round(self.full, 4) if self.full is not None else None},
            pair:
                {pairs_str}
        )
        """)


def get_labels_agreement(
    label_type: type,
    labels_array: npt.NDArray[ValueScalarType],
    level_of_measurement: LevelOfMeasurement | None = None,
) -> AgreementOutput:
    """Compute the inter-annotator agreement using
    [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) for an (M, N) array of labels,
    where M is the number of annotators and N is the number of units.

    Args:
        level_of_measurement (Literal['nominal', 'ordinal', 'interval', 'ratio']): The level of measurement for the
            labels when using Krippendorff's alpha. Can be "nominal", "ordinal", "interval", or "ratio", depending
            on the type of labels. Default: "nominal" for string labels, "ordinal" for int labels, and "interval"
            for float labels.

    Returns:
        Labels correlation (for numeric) or inter-annotator agreement (for categorical) between the two entries
    """
    num_annotators = labels_array.shape[0]
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
    if labels_array.dtype.kind in {"i", "u", "f"}:
        unique_vals = np.unique(labels_array[~np.isnan(labels_array)])
    elif labels_array.dtype.kind in {"U", "S"}:  # Unicode or byte string.
        # `np.asarray` will coerce `np.nan` values to "nan".
        unique_vals = np.unique(labels_array[labels_array != "nan"])
    else:
        raise ValueError(
            f"Unsupported label type: {labels_array.dtype}. Please specify the level of measurement explicitly."
        )
    measurement_type = None
    if len(unique_vals) > 1:
        full_score = alpha(reliability_data=labels_array, level_of_measurement=level_of_measurement)
        pair_scores = np.identity(num_annotators)
        for i in range(num_annotators):
            for j in range(i + 1, num_annotators):
                if np.array_equal(
                    labels_array[i, :], labels_array[j, :], equal_nan=True if label_type is float else False
                ):
                    pair_score = 1.0
                else:
                    pair_score = alpha(
                        reliability_data=labels_array[[i, j], :],
                        level_of_measurement=level_of_measurement,
                    )
                pair_scores[i, j] = pair_score
                pair_scores[j, i] = pair_score
        measurement_type = "krippendorff_" + (
            level_of_measurement if isinstance(level_of_measurement, str) else "custom"
        )
    else:
        full_score = None
        binary_labels = (labels_array == unique_vals[0]).astype(int)
        pair_scores = spearmanr(binary_labels, axis=1).statistic  # type: ignore
        measurement_type = "spearmanr_binary"
    pair_scores = cast(list[list[float]], pair_scores.tolist())
    return AgreementOutput(
        full=full_score,
        pair=pair_scores,
        type=measurement_type,
    )

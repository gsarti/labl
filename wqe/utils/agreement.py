import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import cast

import numpy as np
import numpy.typing as npt
from krippendorff import alpha
from krippendorff.krippendorff import LevelOfMeasurement, ValueScalarType

logger = logging.getLogger(__name__)


@dataclass
class AgreementOutput:
    """Data class for storing the output of the inter-annotator agreement computation.

    Attributes:
        full (float): The full agreement for all annotation sets.
        pair (list[list[float]]): Pairwise agreement between all annotators.
        level (str): The level of measurement used in the computation.
        method (str): The correlation method used in the computation.
    """

    full: float
    pair: list[list[float]]
    level: LevelOfMeasurement

    def __str__(self) -> str:
        pairs_str = " " * 4 + " | ".join(f"A{i:<3}" for i in range(len(self.pair))) + "|\n"
        for idx_row, row in enumerate(self.pair):
            pairs_str += (
                f"A{idx_row:<3}"
                + " | ".join(
                    f"{round(x, 2):<4}" if idx_col != idx_row else f"{' ':<4}" for idx_col, x in enumerate(row)
                )
                + "|\n"
            )
        pairs_str = pairs_str.replace("\n", "\n" + " " * 16)
        return dedent(f"""\
        AgreementOutput(
            level: {self.level},
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
    pair_scores = cast(list[list[float]], pair_scores.tolist())
    return AgreementOutput(
        full=full_score,
        pair=pair_scores,
        level=level_of_measurement,
    )

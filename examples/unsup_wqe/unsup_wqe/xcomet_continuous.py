# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
XCOMET Metric
==============
    eXplainable Metric is a multitask metric that performs error span detection along with
    sentence-level regression. It can also be used for QE (reference-free evaluation).
"""

import torch
from comet.models import XCOMETMetric
from comet.models.utils import Prediction
from torch import nn


class XCOMETContinuousMetric(XCOMETMetric):
    """eXplainable COMET is same has Unified Metric but overwrites predict function.
    This way we can control better for the models inference.

    To cast back XCOMET models into UnifiedMetric (and vice-versa) we can simply run
    model.__class__ = UnifiedMetric

    """

    def predict_step(self, batch: dict[str, torch.Tensor], **kwargs) -> Prediction:
        """PyTorch Lightning predict_step

        Args:
            batch (Dict[str, torch.Tensor]): The output of your prepare_sample function
            batch_idx (Optional[int], optional): Integer displaying which batch this is
                Defaults to None.
            dataloader_idx (Optional[int], optional): Integer displaying which
                dataloader this is. Defaults to None.

        Returns:
            Prediction: Model Prediction
        """
        mt_mask = batch[0]["label_ids"] != -1
        mt_length = mt_mask.sum(dim=1)
        seq_len = mt_length.max()
        # XCOMET is suposed to be used with a reference thus 3 different inputs.
        if len(batch) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Weighted average of the softmax probs along the different inputs.
            subword_probs = [
                nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :] * (w if len(batch) == 3 else 1)
                for w, o in zip(self.input_weights_spans, predictions, strict=False)
            ]
            subword_probs = torch.sum(torch.stack(subword_probs), dim=0)
        # XCOMET if reference is not available we fall back to QE model.
        else:
            model_output = self.forward(**batch[0])
            mt_mask = batch[0]["label_ids"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            subword_probs = nn.functional.softmax(model_output.logits, dim=2)[:, :seq_len, :]
        input_ids = batch[0]["input_ids"]
        mt_offsets = batch[0]["mt_offsets"]
        all_tokens = []
        all_no_error_probs = []
        all_minor_error_probs = []
        all_major_error_probs = []
        all_critical_error_probs = []
        for i in range(len(mt_offsets)):
            seq_len = len(mt_offsets[i])
            tokens = self.encoder.tokenizer.convert_ids_to_tokens(input_ids[i, :seq_len])
            all_probs = subword_probs[i, :seq_len]
            no_error_probs = all_probs[:, 0].squeeze().tolist()
            minor_error_probs = all_probs[:, 1].squeeze().tolist()
            major_error_probs = all_probs[:, 2].squeeze().tolist()
            critical_error_probs = all_probs[:, 3].squeeze().tolist()
            all_tokens.append(tokens)
            all_no_error_probs.append(no_error_probs)
            all_minor_error_probs.append(minor_error_probs)
            all_major_error_probs.append(major_error_probs)
            all_critical_error_probs.append(critical_error_probs)
        batch_prediction = Prediction(
            scores=torch.tensor([0.0 for i in range(len(mt_offsets))]),  # Sentence-level scores are ignored
            metadata=Prediction(
                tokens=all_tokens,
                no_error_probs=all_no_error_probs,
                minor_error_probs=all_minor_error_probs,
                major_error_probs=all_major_error_probs,
                critical_error_probs=all_critical_error_probs,
            ),
        )
        return batch_prediction

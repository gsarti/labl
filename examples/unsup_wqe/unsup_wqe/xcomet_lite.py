import os

import comet.encoders
import torch
from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder
from comet.models import XCOMETMetric
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.deberta_v2 import modeling_deberta_v2


class DeBERTaEncoder(BERTEncoder):
    """DeBERTa encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
        local_files_only (bool): Whether or not to only look at local files.
    """

    def __init__(
        self,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super(Encoder, self).__init__()
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, local_files_only=local_files_only)
        if load_pretrained_weights:
            self.model = AutoModel.from_pretrained(pretrained_model)
        else:
            self.model = AutoModel.from_config(
                AutoConfig.from_pretrained(pretrained_model, local_files_only=local_files_only),
            )
        self.model.encoder.output_hidden_states = True

        self.model.encoder.layer = nn.ModuleList(
            [
                modeling_deberta_v2.DebertaV2Layer(AutoConfig.from_pretrained(pretrained_model))
                for _ in range(self.model.config.num_hidden_layers)
            ]
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face
            local_files_only (bool): Whether or not to only look at local files.

        Returns:
            DeBERTaEncoder: DeBERTaEncoder object.
        """
        return DeBERTaEncoder(pretrained_model, load_pretrained_weights, local_files_only=local_files_only)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        return {
            "sentemb": model_output.last_hidden_state[:, 0, :],
            "wordemb": model_output.last_hidden_state,
            "all_layers": model_output.hidden_states,
            "attention_mask": attention_mask,
        }


class XCOMETLiteMetric(XCOMETMetric, PyTorchModelHubMixin):
    """xCOMET-Lite model."""

    def __init__(
        self,
        encoder_model="DeBERTa",
        pretrained_model="microsoft/mdeberta-v3-base",
        word_layer=8,
        validation_data=[],
        word_level_training=True,
        hidden_sizes=(3072, 1024),
        load_pretrained_weights=False,
        *args,
        **kwargs,
    ):
        comet.encoders.str2encoder["DeBERTa"] = DeBERTaEncoder
        super().__init__(
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            word_layer=word_layer,
            layer_transformation="softmax",
            validation_data=validation_data,
            word_level_training=word_level_training,
            hidden_sizes=hidden_sizes,
            load_pretrained_weights=load_pretrained_weights,
        )

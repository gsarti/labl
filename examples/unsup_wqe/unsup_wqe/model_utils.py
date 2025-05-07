from typing import cast

import torch
from inseq import AttributionModel
from inseq.attr import StepFunctionArgs
from inseq.attr.step_functions import probability_fn
from inseq.data.attribution import FeatureAttributionSequenceOutput
from inseq.models import HuggingfaceModel
from torch import FloatTensor, Tensor
from torch.autograd import grad
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.cohere import CohereForCausalLM
from transformers.models.llama import LlamaForCausalLM
from transformers.models.m2m_100 import M2M100ForConditionalGeneration
from transformers.models.marian import MarianMTModel
from transformers.models.mbart import MBartForConditionalGeneration
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


def get_rank(logits, target_indices):
    vals = logits[:, target_indices].squeeze()
    return (logits > vals).long().sum(dim=-1)


def logit_lens(model, token_layer_act):
    if isinstance(model, M2M100ForConditionalGeneration):
        return model.lm_head(model.model.decoder.layer_norm(token_layer_act))
    elif isinstance(model, MarianMTModel):
        return model.lm_head(token_layer_act)
    elif isinstance(model, MBartForConditionalGeneration):
        return model.lm_head(model.model.decoder.layer_norm(token_layer_act))
    elif isinstance(model, CohereForCausalLM):
        return model.lm_head(token_layer_act) * model.logit_scale
    elif isinstance(model, Qwen2ForCausalLM):
        return model.lm_head(token_layer_act)
    elif isinstance(model, LlamaForCausalLM):
        return model.lm_head(token_layer_act)
    else:
        raise NotImplementedError(f"Logit lens not implemented for model type {model.__class__.__name__}")


def get_mcd_probs(args: StepFunctionArgs, n_mcd_steps: int = 10, logprob: bool = False) -> torch.Tensor:
    # Original probability from the model without noise
    orig_prob = probability_fn(args, logprob=logprob)

    # Compute noisy predictions using the noisy model
    # Important: must be in train mode to ensure noise for MCD
    args.attribution_model.train()
    noisy_probs = []
    for _ in range(n_mcd_steps - 1):
        aux_batch = args.attribution_model.formatter.convert_args_to_batch(args)  # type: ignore
        aux_output = args.attribution_model.get_forward_output(
            aux_batch,  # type: ignore
            use_embeddings=args.attribution_model.is_encoder_decoder,
        )
        args.forward_output = aux_output
        noisy_prob = probability_fn(args, logprob=logprob).to(orig_prob.device)
        noisy_probs.append(noisy_prob)
    out = torch.stack([orig_prob] + noisy_probs)
    return out


def get_blood_score(curr_layer_states, next_layer_states, n_estimators: int | None = 10) -> torch.Tensor:
    if n_estimators is not None:
        ests = []
        for _ in range(n_estimators):
            v = torch.randn((next_layer_states.shape[0], next_layer_states.shape[2])).to(next_layer_states.device)
            est = grad((next_layer_states[:, 0, :] * v).sum(), curr_layer_states, retain_graph=True)[0][:, 0, :]
            w = torch.randn((next_layer_states.shape[0], next_layer_states.shape[2])).to(next_layer_states.device)
            ests.append(((est * w).sum(dim=1) ** 2).cpu())  # (batch_size)
        norm_ests = torch.stack(ests, dim=1)  # (batch_size, n_estimators)
        return norm_ests.mean(dim=1)  # (batch_size)

    else:
        grads = [
            grad(next_layer_states[:, 0, j].sum(), curr_layer_states, retain_graph=True)[0][:, 0, :].cpu()
            for j in range(next_layer_states.shape[2])
        ]
        norm_ests = torch.cat(grads, dim=1)  # (batch_size, embeding_size*embeding_size)
        return (norm_ests**2).sum(dim=1)  # (batch_size)


def get_decoder_states(output: Seq2SeqLMOutput | CausalLMOutput) -> tuple[FloatTensor, ...]:
    if isinstance(output, Seq2SeqLMOutput):
        return cast(tuple[FloatTensor, ...], output.decoder_hidden_states)
    elif isinstance(output, CausalLMOutput | CausalLMOutputWithPast):
        return cast(tuple[FloatTensor, ...], output.hidden_states)
    else:
        raise ValueError(f"Unsupported output type: {output.__class__.__name__}")


def get_num_layers(model: AttributionModel) -> int:
    hf_model = model.model
    if isinstance(hf_model, M2M100ForConditionalGeneration):
        return hf_model.config.num_hidden_layers
    if isinstance(hf_model, MBartForConditionalGeneration):
        return hf_model.config.num_hidden_layers
    elif isinstance(hf_model, MarianMTModel):
        return hf_model.config.num_hidden_layers
    elif isinstance(hf_model, CohereForCausalLM):
        return hf_model.config.num_hidden_layers
    elif isinstance(hf_model, Qwen2ForCausalLM):
        return hf_model.config.num_hidden_layers
    elif isinstance(hf_model, LlamaForCausalLM):
        return hf_model.config.num_hidden_layers
    else:
        raise NotImplementedError(f"Number of layers not implemented for model type {model.__class__.__name__}")


def get_mt_tokens_and_metrics(
    out: FeatureAttributionSequenceOutput, model: HuggingfaceModel
) -> tuple[list[str], Tensor]:
    if model.is_encoder_decoder:
        mt_tokens = [t.token for t in out.target[1:]]
    else:
        mt_tokens = [t.token for t in out.target[len(out.source) :]]
    out_metrics = out.step_scores["unsupervised_qe_metrics_fn"]  # type: ignore
    return mt_tokens, out_metrics


def get_attributions(out: FeatureAttributionSequenceOutput) -> Tensor:
    if out.source_attributions is not None and out.target_attributions is not None:
        attributions = torch.cat([out.source_attributions, out.target_attributions], dim=0).permute(1, 0, 2, 3) / 2
    elif out.target_attributions is not None:
        attributions = out.target_attributions.permute(1, 0, 2, 3)
    else:
        raise ValueError("No attribution scores found in the output.")
    return attributions

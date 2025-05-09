import torch
import torch.nn.functional as F
from inseq import AttributionModel
from inseq.attr import StepFunctionArgs
from inseq.utils.typing import SingleScorePerStepTensor
from torch.distributions import Categorical

from .model_utils import (
    get_blood_score,
    get_decoder_states,
    get_mcd_probs,
    get_num_layers,
    get_rank,
    logit_lens,
)


def unsupervised_qe_metrics_fn(
    args: StepFunctionArgs,
    mcd_n_steps: int = 10,
    mcd_logprob: bool = True,
    blood_n_estimators: int | None = 10,
) -> SingleScorePerStepTensor:
    forward_output = args.attribution_model.get_forward_output(
        args.attribution_model.formatter.convert_args_to_batch(args),  # type: ignore
        output_hidden_states=True,
        output_attentions=True,
    )
    print("Step 1")
    all_layer_states = get_decoder_states(forward_output)
    out_logits = args.attribution_model.output2logits(args.forward_output)
    output_logprobs = F.log_softmax(out_logits, dim=-1)
    logprob = output_logprobs[0, args.target_ids].item()

    ll_logit_per_layer = []
    ll_entropy_per_layer = []
    ll_logprob_per_layer = []
    ll_kl_div_per_layer = []
    blood_per_layer = []
    logit_lens_prediction_ranks = []
    num_layers = len(all_layer_states)
    for layer_idx, layer_states in enumerate(all_layer_states):
        print("Step 3")
        # logit_lens_logprob
        ll_logits = logit_lens(args.attribution_model.model, layer_states[:, -1, :])
        ll_entropy = Categorical(logits=ll_logits).entropy().item()
        ll_entropy_per_layer.append(ll_entropy)
        ll_logprobs = F.log_softmax(ll_logits, dim=-1)
        ll_target_logit = ll_logits[0, args.target_ids].item()
        ll_target_logprob = ll_logprobs[0, args.target_ids].item()
        ll_logit_per_layer.append(ll_target_logit)
        ll_logprob_per_layer.append(ll_target_logprob)

        rank = get_rank(ll_logprobs, args.target_ids).item()
        logit_lens_prediction_ranks.append(rank)

        # logit_lens_kl_div
        ll_kl_div = F.kl_div(ll_logprobs, output_logprobs, reduction="sum", log_target=True).item()
        ll_kl_div_per_layer.append(ll_kl_div)

        # blood
        if layer_idx < num_layers - 1:
            next_layer_states = all_layer_states[layer_idx + 1]
            blood_score = get_blood_score(layer_states, next_layer_states, n_estimators=blood_n_estimators)
            blood_per_layer.append(blood_score.item())

    # mcd_logprob_avg and mcd_logprob_var
    # mcd_logprobs = get_mcd_probs(args, mcd_n_steps, logprob=mcd_logprob)
    # mcd_logprob_avg = mcd_logprobs.mean().item()
    # mcd_logprob_var = mcd_logprobs.var().item()

    print("Step 4")
    # logit_lens_rank
    logit_lens_rank = logit_lens_prediction_ranks.index(0) if 0 in logit_lens_prediction_ranks else num_layers

    # logprobs_entropy
    print("Step 5")
    logprobs_entropy = Categorical(logits=output_logprobs.detach().clone().requires_grad_(False)).entropy().item()

    # logit_lens_logprob_variation
    logit_lens_logprob_variation = Categorical(logits=torch.tensor(ll_logit_per_layer)).entropy().item()

    # logit_lens_kl_div_variation
    logit_lens_kl_div_variation = Categorical(logits=torch.tensor(ll_kl_div_per_layer)).entropy().item()
    print("Step 6")
    return torch.tensor(
        [
            ll_logprob_per_layer
            + ll_entropy_per_layer
            + ll_kl_div_per_layer
            + blood_per_layer
            + [
                logprob,
                0,
                0,
                logit_lens_rank,
                logprobs_entropy,
                logit_lens_logprob_variation,
                logit_lens_kl_div_variation,
            ]
        ]
    )


def get_metric_names(model: AttributionModel) -> list[str]:
    n_layers = get_num_layers(model)
    metric_names = []
    metric_names += [f"logit_lens_logprob_layer_{i}" for i in range(n_layers + 1)]
    metric_names += [f"logit_lens_entropy_layer_{i}" for i in range(n_layers + 1)]
    metric_names += [f"logit_lens_kl_div_layer_{i}" for i in range(n_layers + 1)]
    metric_names += [f"blood_layer_{i}" for i in range(n_layers)]
    return metric_names + [
        "logprob",
        "mcd_logprob_mean",
        "mcd_logprob_var",
        "logit_lens_rank",
        "logprobs_entropy",
        "logit_lens_logprob_variation",
        "logit_lens_kl_div_variation",
    ]

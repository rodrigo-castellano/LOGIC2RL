"""PPO training-metrics aggregation + logit diagnostics (pillar: algorithm).

Module functions that take the :class:`~algorithms.ppo.ppo.PPO` instance and read
the buffers / policy off it. Kept out of ``ppo.py`` so the algorithm shell stays
the train/eval loop; these are pure post-hoc aggregation + diagnostics.
"""
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

from .loss import explained_variance


def compute_train_metrics(
    ppo, pg_losses: Tensor, vl_losses: Tensor, ent_losses: Tensor,
    clips: Tensor, kls: Tensor, batch_count: int, last_epoch_start: int,
) -> Dict[str, Any]:
    """Aggregate per-batch losses into summary metrics. Single GPU→CPU sync."""
    with torch.no_grad():
        values = ppo.rollout_buffer.values.flatten()
        returns = ppo.rollout_buffer.returns.flatten()
        advantages = ppo.rollout_buffer.advantages.flatten()

        final_metrics = torch.stack([
            pg_losses[:batch_count].mean(), vl_losses[:batch_count].mean(),
            -ent_losses[:batch_count].mean(), clips[:batch_count].mean(),
            kls[last_epoch_start:batch_count].mean(), explained_variance(values, returns),
            values.mean(), values.std(), advantages.abs().mean(), advantages.std(),
        ])

        # Per-epoch loss lines are diagnosis-only (10+ lines per iteration; the
        # aggregate losses land in last_metrics / metrics.json regardless).
        verbose_gpu = None
        if ppo.verbose and getattr(ppo.config, "log_diagnostics", False) and ppo._epoch_end_indices:
            epoch_losses = torch.stack([info[2] for info in ppo._epoch_end_indices])
            all_stats = torch.stack([
                torch.stack([vl_losses[:ei].mean(), pg_losses[:ei].mean(), ent_losses[:ei].mean(), kls[:ei].mean(), clips[:ei].mean()])
                for _, ei, _ in ppo._epoch_end_indices
            ])
            verbose_gpu = torch.cat([epoch_losses.unsqueeze(1), all_stats], dim=1)

        final_metrics_cpu = final_metrics.cpu()
        verbose_cpu = verbose_gpu.cpu() if verbose_gpu is not None else None

    if verbose_cpu is not None:
        for i, (epoch, _, _) in enumerate(ppo._epoch_end_indices):
            print(f"Epoch {epoch+1}/{ppo.n_epochs}. ")
            print(f"Losses: total {verbose_cpu[i, 0]:.5f}, value {verbose_cpu[i, 1]:.5f}, "
                  f"policy {verbose_cpu[i, 2]:.5f}, entropy {verbose_cpu[i, 3]:.5f}, "
                  f"approx_kl {verbose_cpu[i, 4]:.5f} clip_fraction {verbose_cpu[i, 5]:.5f}. ")
    ppo._epoch_end_indices = []

    _KEYS = ["policy_loss", "value_loss", "entropy", "clip_fraction", "approx_kl",
             "explained_var", "value_mean", "value_std", "advantage_mean", "advantage_std"]
    result = dict(zip(_KEYS, final_metrics_cpu.tolist()))
    result.update(
        loss=result["policy_loss"] - ppo.ent_coef * result["entropy"] + ppo.vf_coef * result["value_loss"],
        learning_rate=ppo.learning_rate, entropy_loss=-result["entropy"],
        clip_range=ppo.clip_range, policy_gradient_loss=result["policy_loss"],
    )
    result.update(compute_logit_diagnostics(ppo))
    return result


@torch.no_grad()
def compute_logit_diagnostics(ppo) -> Dict[str, float]:
    """Logit/embedding diagnostics from one rollout buffer mini-batch.

    Reads the ActorCriticPolicy internals (feature extractor + actor bodies/heads);
    a policy without them (e.g. the scored QKGEPolicy) gets no diagnostics.
    """
    policy = ppo.policy
    if not (hasattr(policy, "features_extractor") and hasattr(policy, "mlp_extractor")):
        return {}
    rb = ppo.rollout_buffer
    bs = min(ppo.batch_size, rb.flat_obs['sub_index'].shape[0])
    sub_idx = rb.flat_obs['sub_index'][:bs].to(torch.int32)
    derived = rb.flat_obs['derived_sub_indices'][:bs].to(torch.int32)
    mask = rb.flat_obs['action_mask'][:bs]

    obs_emb = policy.features_extractor.embedder.get_embeddings_batch(sub_idx)
    act_emb = policy.features_extractor.embedder.get_embeddings_batch(derived)
    features = TensorDict({"obs_emb": obs_emb, "act_emb": act_emb, "action_mask": mask}, batch_size=obs_emb.shape[:1])
    logits = policy.mlp_extractor.forward_actor(features)

    net = policy.mlp_extractor
    p_obs = net.obs_head(net.obs_body(obs_emb))
    p_act = net.action_head(net.action_body(act_emb))

    valid = mask.bool()
    valid_logits = logits[valid]
    if valid_logits.numel() == 0:
        return {}
    probs = F.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(-1).mean()

    diag = {
        'logit_mean': valid_logits.mean().item(), 'logit_std': valid_logits.std().item(),
        'logit_min': valid_logits.min().item(), 'logit_max': valid_logits.max().item(),
        'logit_entropy': entropy.item(),
        'obs_norm': p_obs.norm(dim=-1).mean().item(), 'act_norm': p_act.norm(dim=-1).mean().item(),
    }
    if net.learnable_temperature:
        diag['logit_temperature'] = net.log_temp.exp().item()
    return diag

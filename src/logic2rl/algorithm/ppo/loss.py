"""Loss module and helpers for PPO training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def fused_clip_grad_norm_(parameters: list[Tensor], max_norm: float, norm_type: float = 2.0) -> Tensor:
    """Clip gradients without GPU-CPU sync. Concatenates all grads for a single fused norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total_norm = torch.cat([g.flatten() for g in grads]).norm(norm_type)
    clip_coef = (total_norm + 1e-6).reciprocal_().mul_(max_norm).clamp_(max=1.0)
    torch._foreach_mul_(grads, clip_coef)
    return total_norm


def explained_variance(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """1 - Var(residual) / Var(y_true). ~1.0 = good value function."""
    var_y = torch.var(y_true)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)


class PPOLossModule(nn.Module):
    """Fused policy forward + PPO loss computation."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs, actions, advantages, returns, old_log_probs, old_values,
                clip_range, clip_range_vf, ent_coef, vf_coef):
        """Compute PPO loss over the buffer's round-tripped observation dict. Returns
        ``(loss, policy_loss, value_loss, entropy_loss, approx_kl, clip_fraction)``."""
        values, log_probs, entropy = self.policy.evaluate_actions(obs, actions)
        values = values.flatten()

        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        pg1 = advantages * ratio
        pg2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.minimum(pg1, pg2).mean()
        clip_fraction = (torch.abs(ratio - 1) > clip_range).float().mean()

        # Python if/else avoids torch.tensor() inside compiled graph (prevents CUDA graph partitioning)
        value_diff = values - old_values
        if clip_range_vf > 0:
            values_pred = old_values + torch.clamp(value_diff, -clip_range_vf, clip_range_vf)
        else:
            values_pred = values
        value_loss = F.mse_loss(returns, values_pred)

        entropy_loss = -entropy.mean()
        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        approx_kl = ((ratio - 1.0) - log_ratio).mean()

        return loss, policy_loss, value_loss, entropy_loss, approx_kl.detach(), clip_fraction

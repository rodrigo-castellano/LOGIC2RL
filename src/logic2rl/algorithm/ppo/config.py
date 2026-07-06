"""PPO algorithm config (pillar: algorithm).

``PPOConfig`` declares only PPO / training-loop / policy-architecture DATA — no
env, dataset, or KGE fields, and no construction (that lives in the builders and
in ``PPO._build_policy``). A consumer composes it with an env config:

    class KGEConfig(BaseConfig, SearchConfig): ...   # SearchConfig(QKGEConfig(PPOConfig))

It is always composed with an env config, never instantiated bare, and imports
nothing from ``base`` or ``kge``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class PPOConfig:
    """PPO + training-loop + policy-architecture parameters."""

    # Algorithm selection
    algorithm_type: str = "ppo"  # 'ppo' | 'q_kge'

    # Policy architecture
    hidden_dim: int = 128
    num_layers: int = 8
    separate_value_network: bool = True  # Separate backbone for the value network
    use_l2_norm: bool = False
    sqrt_scale: bool = False
    temperature: Optional[float] = None  # None = raw dot product
    learnable_temperature: bool = False  # Learn temperature via nn.Parameter (requires use_l2_norm)
    obs_body_layers: Optional[int] = 8     # Residual blocks in obs body (None = Identity)
    obs_head_layers: Optional[int] = 2     # Projection layers after obs body (None = Identity)
    action_body_layers: Optional[int] = None  # Residual blocks in action body (None = Identity)
    action_head_layers: Optional[int] = None  # Projection layers after action body (None = Identity)
    shared_policy_body: bool = False       # Share body weights between obs and actions
    shared_policy_head: bool = False       # Share head weights (requires shared_policy_body=True)
    value_head_scale: float = 0.5  # Scale factor for value head hidden dim

    # PPO / training hyperparams
    n_steps: int = 256
    n_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clip range (None=disabled, matches SB3)
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # Early-stopping KL threshold (None=disabled, matches SB3)
    weight_decay: float = 0.0  # L2 weight decay for AdamW (0 = Adam-equivalent)
    normalize_advantage: bool = True  # Normalize advantages per batch (matches SB3)
    normalize_returns: bool = False  # Normalize returns for value targets
    total_timesteps: int = 100000000

    # Compilation / performance (the fast path: fullgraph torch.compile + CUDA graphs).
    # compile=False runs eager — for debugging graph breaks (slower). compile_mode is the
    # torch.compile mode for the rollout core / loss step / eval step.
    compile: bool = True
    compile_mode: str = 'reduce-overhead'
    use_amp: bool = True  # autocast (bf16 if supported) for the loss step, CUDA only

    # LR warmup
    lr_warmup: bool = False
    lr_warmup_steps: float = 0.1  # Warmup for first 10% of training

    # LR decay
    lr_decay: bool = False
    lr_init_value: float = 1e-4
    lr_final_value: float = 1e-6
    lr_start: float = 0.0
    lr_end: float = 1.0
    lr_transform: str = 'cos'

    # Entropy decay
    ent_coef_decay: bool = False
    ent_coef_init_value: float = 0.15
    ent_coef_final_value: float = 0.02
    ent_coef_start: float = 0.0
    ent_coef_end: float = 0.8
    ent_coef_transform: str = 'cos'

    # Model saving / eval cadence
    save_model: bool = True
    load_model: Union[bool, str] = False  # False or 'last_epoch' or path
    restore_best: bool = True  # restore_best_val_model
    load_best_metric: str = 'eval'
    eval_freq: int = 0

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience_steps: int = 3_000_000
    early_stopping_min_delta: float = 0.01
    # Metric the early-stop monitors. None → the checkpoint metric (eval_best_metric).
    # Decoupled so we can stop on the RL's own objective (rollout reward) while still
    # checkpointing best-MRR: on KGE-bound datasets MRR peaks at init (λ=0 = KGE ceiling)
    # and only decays, so MRR-early-stop fires before the policy has learned anything.
    # Aliases: 'reward'→'ep_rew_mean', 'mrr'→'mrr_mean'.
    early_stopping_metric: Optional[str] = None

    # Callback control
    use_callbacks: bool = True

    # Run flags
    profile: bool = False
    seed_run_i: int = 0  # specific run seed

    def __post_init__(self) -> None:
        _s = super()
        if hasattr(_s, "__post_init__"):
            _s.__post_init__()
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.shared_policy_head and not self.shared_policy_body:
            raise ValueError("shared_policy_head=True requires shared_policy_body=True")
        # Parity mode forces a symmetric shared architecture
        if getattr(self, "parity", False):
            self.shared_policy_body = True
            self.shared_policy_head = True
        # Decay sanity
        if self.lr_decay and self.lr_init_value <= self.lr_final_value:
            import warnings
            warnings.warn(
                f"lr_decay enabled but lr_init_value ({self.lr_init_value}) <= "
                f"lr_final_value ({self.lr_final_value})"
            )
        if self.ent_coef_decay and self.ent_coef_init_value <= self.ent_coef_final_value:
            import warnings
            warnings.warn(
                f"ent_coef_decay enabled but init ({self.ent_coef_init_value}) <= "
                f"final ({self.ent_coef_final_value})"
            )

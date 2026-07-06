# policy/ -- Actor-Critic Policy Networks

Neural networks that embed logical states and output action probabilities and value estimates.

## Files

| File | Purpose |
|------|---------|
| `actor_critic.py` | `ActorCriticPolicy` -- top-level policy |
| `network.py` | `SharedPolicyValueNetwork` -- unified architecture with asymmetric obs/action backbones and actor/critic heads |
| `layers.py` | `FusedLinearReluLayerNorm`, `FusedLinearRelu` -- CUDA-graph-safe fused layers |
| `extractor.py` | `CustomCombinedExtractor` -- observation dict to dense embeddings |

## Architecture

```
Input: obs dict
  sub_index           [B, 1, A, 3]   (current state indices)
  derived_sub_indices [B, G, A, 3]   (action/derived state indices)
  action_mask         [B, G]         (valid action mask)
  original_query      [B, 1, 3]      (goal, when goal_conditioned=True)
      |
      v
  CustomCombinedExtractor
      |  Embeds indices via shared atom embedder
      |  -> obs_emb [B,1,E], act_emb [B,G,E], goal_emb [B,1,E] (optional)
      v
  SharedPolicyValueNetwork
      |
      +--- Actor path:  goal fusion -> obs_body -> obs_head --|
      |                               action_body -> action_head --|--> dot product -> logits [B,G]
      +--- Critic path: goal fusion -> value_body -> value_head -> values [B]
```

## Key Classes

### ActorCriticPolicy

Top-level module wrapping extraction, architecture, and action distribution.

**Public methods:**
- `forward(obs, deterministic)` -- sampling-only: returns `(actions, values, log_probs)`
- `evaluate_actions(obs, actions)` -- returns `(values, log_probs, entropy)` from dict obs
- `get_values(obs, separate_heads=False)` -- value-only prediction; `separate_heads=True` returns `(v_pos, v_neg)`
- `get_logits(obs)` -- actor-only pass from obs dict

### CustomCombinedExtractor

Converts obs dict/TensorDict → dense embeddings as a `TensorDict`. Provides 3 forward paths to avoid unnecessary work:

| Method | Returns (TensorDict keys) | Skips |
|--------|--------------------------|-------|
| `forward(obs)` | `obs_emb`, `act_emb`, `action_mask`, + optional `goal_emb`, `query_kge_score`, `query_label` | nothing |
| `forward_actor(obs)` | `obs_emb`, `act_emb`, `action_mask`, + optional `goal_emb`, `query_kge_score` | `query_label` |
| `forward_critic(obs)` | `obs_emb`, + optional `goal_emb`, `query_kge_score`, `query_label` | action embedding (expensive), mask |

Optional keys are only present when the corresponding input is non-None. Network methods use `.get("key", None)` to handle absence.

### SharedPolicyValueNetwork

Unified architecture with configurable per-side layer counts.

**Body** (`obs_body_layers` / `action_body_layers`):
- `None`: `nn.Identity()` -- pass-through, output dim = E
- `N` (int): `SharedBody` -- input projection `[E]->[H]` + N residual blocks `[H]->[H]`, output dim = H

**Head** (`obs_head_layers` / `action_head_layers`):
- `None`: `nn.Identity()` -- pass-through
- `1`: `Linear(D_in, E)` -- single projection
- `N>=2`: `(N-1) x FusedLinearRelu(D_in, D_in) + Linear(D_in, E)`

Where `D_in` = H (if body has layers) or E (if body is Identity).

**Weight sharing:**
- `shared_policy_body=False` (default): separate body instances
- `shared_policy_body=True`: obs and action bodies are the same module
- `shared_policy_head=False` (default): separate head instances
- `shared_policy_head=True`: obs and action heads are the same module (requires `shared_policy_body=True`)

**Forward methods (3-path pattern matching extractor):**

| Method | Input | Returns | Use case |
|--------|-------|---------|----------|
| `forward(features)` | TensorDict (all keys) | `(logits [B,G], values [B])` | Training: `evaluate_actions`, `forward` |
| `forward_actor(features)` | TensorDict (no `query_label`) | `logits [B,G]` | `get_logits` |
| `forward_critic(features, separate_heads)` | TensorDict (no `act_emb`/`action_mask`) | `values [B]` or `(v_pos, v_neg)` | `get_values` |

All methods accept `TensorDict` features from the matching extractor path and access fields by key.

### SharedBody

Residual MLP backbone: input projection `[E]->[H]` followed by N residual blocks with skip connections. Uses `FusedLinearReluLayerNorm` layers (the `init` arg selects `"relu"` He init or `"linear"` SB3-parity init).

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | Hidden layer width (H) for body with layers |
| `obs_body_layers` | 8 | Residual blocks in obs body (None = Identity) |
| `obs_head_layers` | 2 | Projection layers after obs body (None = Identity) |
| `action_body_layers` | 8 | Residual blocks in action body (None = Identity) |
| `action_head_layers` | 2 | Projection layers after action body (None = Identity) |
| `shared_policy_body` | False | Share body weights between obs and actions |
| `shared_policy_head` | False | Share head weights (requires shared_policy_body=True) |
| `separate_value_network` | True | Independent backbone for critic |
| `value_head_scale` | 0.5 | Scale factor for value head hidden dim |
| `sqrt_scale` | False | Scale logits by 1/sqrt(E) |
| `temperature` | None | Fixed temperature divisor for logits |
| `learnable_temperature` | False | CLIP-style learned log-temperature |
| `goal_conditioned` | False | Condition on original query |

## Asymmetric Processing

The obs and action sides can use different layer counts:

| obs_body | obs_head | act_body | act_head | shared_body | shared_head | Description |
|----------|----------|----------|----------|-------------|-------------|-------------|
| 8 | 2 | 8 | 2 | False | False | Default: independent deep paths |
| 8 | 2 | 8 | 2 | True | True | Old symmetric (parity mode) |
| 8 | 2 | None | None | False | False | Deep obs, raw actions |
| 8 | 2 | None | 2 | False | False | Deep obs, lightweight action head only |
| 8 | None | 8 | None | False | False | Same body, no heads (dot product in H-space) |
| None | None | None | None | False | False | Pure embeddings baseline |

The value path always uses `obs_body_layers` architecture (shared with `obs_body`, or a separate copy if `separate_value_network=True`).

## Parity Mode

When `parity=True`, config forces `shared_policy_body=True` and `shared_policy_head=True` to match the legacy SB3 reference, and the network is built with `init="linear"` (nn.Linear-compatible kaiming_uniform + uniform bias) for bit-exact SB3 weight parity. Production (`parity=False`) uses `init="relu"` (He init, zero bias).

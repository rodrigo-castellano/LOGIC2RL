# `base/env/` — theorem proving as a reinforcement-learning environment

Suppose you have a logic program:

```prolog
parent(bob, alice).   parent(carol, bob).   parent(dave, carol).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

and you want an agent to learn to prove queries like `grandparent(carol, alice)`. Proving is
search: start from the query, repeatedly apply one backward-chaining (SLD) resolution step, and
try to reach a state where nothing is left to prove. This folder packages that search as a
standard RL environment:

- an **episode** is one proof attempt of one query;
- a **state** is the current *resolvent* — the list of goal atoms still to be proved — plus the
  candidate successor resolvents the logic engine derived from it;
- an **action** picks one of those candidates (= chooses which resolution to apply); a boolean
  `action_mask` marks the real ones;
- the episode **terminates** when the resolvent is all-`TRUE` (success), contains a `FALSE`
  (failure), or the depth limit is hit (truncation, kept separate gym-style);
- the base **reward** is +1 on a successful terminal step, else 0.

For the query above, a successful episode looks like: `grandparent(carol, alice)` →
*(apply the rule)* → `parent(carol, Y), parent(Y, alice)` → *(unify with `parent(carol, bob)`)*
→ `parent(bob, alice)` → *(a fact)* → `TRUE`. Three actions, reward 1.

Everything is **batched**: `B` proofs advance simultaneously in pure tensor ops, and the whole
step is written to run under `torch.compile(fullgraph=True)` and CUDA graphs. This env is what
the repo's PPO / evaluators train against (the KGE task), and it is task-agnostic — the MNIST
addition example runs a completely different program through the identical code.

## The three ideas that shape the code

**1. The core is stateless.** `FuncEnv` (`core.py`) never stores episode state on `self`. You
hold the `EnvState` and thread it: `state, out = core.step_autoreset(state, actions)`. A step is
therefore a pure tensors-in → tensors-out function — exactly what compilers and CUDA graphs
need, and what makes search algorithms (beam, MCTS) able to fork/replay states freely. For code
that wants classic gym ergonomics, `GymVecEnvWrapper` (`env.py`) is an *optional* stateful
facade holding one `EnvState` cell.

**2. Features are plug-in components.** The core owns only the invariant proof mechanics.
Everything optional — visited-state pruning, auto-advancing through forced moves, stop actions,
per-rule ids, negative sampling — is an `EnvComponent` (`component.py`): a small object that
declares extra recurrent state fields and hooks into fixed points of the step. Components are
chosen per task in the config (`make_components()`), and each **owns its own knobs as
constructor arguments** — the env constructor has no feature flags. Adding a feature never means
editing `base/env`.

**3. The term language is flat.** Atoms are fixed-width tuples `(predicate, arg1, …,
arg_max_arity)` over constant ids and runtime-variable ids — Datalog-style, no function symbols,
no nested terms. `True`/`False` marker predicates always exist. Queries are single atoms; to
prove a conjunction, add a rule `goal :- a, b` and query `goal`.

**Shape glossary** (used in every signature): `B` = batch (`n_envs`) · `A` = `padding_atoms`
(max atoms per resolvent) · `G` = `padding_states` (max candidates per state = the action-space
width) · `W` = `atom_width` = `max_arity + 1`. A state is `[A, W]`; the candidate set is
`[B, G, A, W]`. Binary KGE has `W = 3`; MNIST's `add/3` has `W = 4`.

```
   your DataLoader (the program)      your EnvComponents (the features)
              │                                  │
              ▼                                  ▼
   ┌─ FuncEnv — stateless compiled core (build_env returns this) ──────────────┐
   │    reset_pool / reset_core → step_autoreset / step_core → observation     │
   │    recurrent EnvState (you thread it)  +  per-transition StepOutput       │
   └───────────────────────────────────────────────────────────────────────────┘
              ▲ optional: GymVecEnvWrapper(core) — stateful gym facade
```

## What lives where

| File | Owns |
|---|---|
| `core.py` | `FuncEnv`: the step/reset/observation algebra, state composition, auto-reset splice, the base `_reward`. Plus the `EnvState` / `StepOutput` NamedTuples. |
| `component.py` | The extension API: `EnvComponent` (every hook, all no-ops by default), `FieldSpec` (a declared state field), `TransitionCtx` (what `step_trace` gets to see). |
| `unification_logic.py` | `UnificationLogic`: candidate generation — engine derive → proof/dead-end marking → keep-masks → one fused compaction. Pure mechanism; `_derive_step` is the single pipeline primitive. |
| `memory.py` | `VisitMemoryComponent`: hashes visited states, prunes already-seen candidates. |
| `proof_components.py` | `UnaryAdvanceComponent`: auto-steps through forced single-successor states (owns its skip flag / iteration bound / unification-safe mode). |
| `env.py` | `GymVecEnvWrapper`: the `gymnasium.vector.VectorEnv` facade. |

Task-specific material lives with the task, not here: the KGE app's components sit in
`kge/env/` (stop actions, rule ids, terminal-outcome taxonomy, negative sampling) and its
4-way reward is an override of `FuncEnv._reward` in `kge/env/env.py:KGEFuncEnv`. The MNIST
example (`examples/mnist/`) mirrors the same anatomy. `base/env` imports neither.

## Using it

`build_env(config)` (in `base/builder.py`) converts a config + data loader into a ready
`FuncEnv`. From there, two mutually exclusive interfaces — pick one per object and don't mix:

**Stateless (what the algorithms use).** You own the state:

```python
core.set_queries(pool)                    # [N, W] query pool (optional; defaults to train split)
state = core.reset_pool()                 # initial EnvState for all B envs
obs   = core.observation(state)           # TensorDict: sub_index, derived_sub_indices, action_mask
for _ in range(n_steps):
    actions = policy(obs)                                  # [B] long
    state, out = core.step_autoreset(state, actions)      # step + same-step reset of done envs
    obs = core.observation(state)
    # out.step_rewards / out.step_dones / out.step_truncated / component trace fields
```

For evaluation and scoring use `core.step_core(state, actions)` instead: identical step, but
finished envs freeze rather than reset, so a batch of proofs can be run to completion exactly
once. The `torch.compile` + CUDA-graph wiring of these calls lives in
`algorithms/ppo/compilation.py`.

**Stateful gym (for external consumers).**

```python
env = GymVecEnvWrapper(core)
obs, info = env.reset()                                   # or options={"queries": q}
obs, rew, term, trunc, info = env.step(actions)
```

The wrapper declares gymnasium's `SAME_STEP` auto-reset mode (done lanes are reset within the
step; their terminal observation is surfaced as `info["final_obs"]` with `info["final_obs_mask"]`
saying which lanes finished). `term` and `trunc` are disjoint. Every `StepOutput` field beyond
the canonical three appears in `info` under its own name. Returns are torch tensors on the env's
device (a numpy-based consumer needs a thin adapter); the wrapper is `VectorEnv`-API compatible
but intentionally not byte-exact `SyncVectorEnv`.

The smallest complete, runnable example of all of the above is
`examples/mnist/tests/test_tiny_walkthrough.py`, which drives the grandparent program from the
top of this page both ways (fixture: `examples/mnist/tests/tiny_kb.py`).

## Inside one step

`step_core(state, actions)` does, in order:

1. **Apply** — each active env replaces its resolvent with the chosen candidate; depth += 1;
   finished envs are frozen throughout.
2. **Classify** — the invariant proof outcome (all-`TRUE` / any-`FALSE` / empty), then each
   component's `step_terminal_outcome` layers its own terminal semantics (e.g. an agent-chosen
   early stop), then depth truncation.
3. **Reward** — `_reward(state, step_done, is_success, active)`.
4. **Derive** — `UnificationLogic` produces the next candidate set: components refresh their
   working fields (`step_update_fields` — memory appends the new state's hash) → one engine
   `derive` call → validity mask ∧ every component `candidate_keep_mask` (memory drops visited)
   → one fused compaction, with a `FALSE` fallback where nothing survived → `candidate_refine`
   in component order (unary auto-advance, stop-action appends) → `step_transform_candidates`
   (e.g. force `{TRUE, FALSE}` as the only choices at the depth limit).
5. **Emit** — the next `EnvState` (components commit their fields) and a `StepOutput`
   (components attach trace fields via `step_trace`, which receives a read-only `TransitionCtx`:
   the actions plus the done/truncation/success edges of this transition).

`step_autoreset` runs `step_core`, draws fresh queries for the finished envs (round-robin over
the pool; components may transform the draw — KGE corrupts a share into negatives), builds their
reset state, and **splices** it in lane-wise. The splice iterates the composed state fields
generically, so component state participates without the core knowing it exists. `StepOutput`
comes from before the splice — the finished episode's data always survives.

## The three data structures

**`EnvState`** — recurrent, fed back each step. Core fields: `current_states [B,A,W]`,
`derived_states [B,G,A,W]`, `derived_counts [B]`, `original_queries [B,A,W]`,
`next_var_indices [B]`, `depths [B]`, `done [B]`, `success [B]`, `per_env_ptrs [B]`. At
construction the core builds `env.State` = these + every component's declared `FieldSpec`s, as
one flat NamedTuple (fixed type → one compiled graph). Membership rule: a tensor belongs in
`EnvState` **iff the next step reads it back**.

**`StepOutput`** — per-transition, emitted and never read back: `step_rewards`, `step_dones`,
`step_truncated`, `is_success`, `original_queries`, `final_observation`, + component trace
fields (`step_endf`, `step_labels`, `terminal_category`, …). Things that exist only at the
moment an episode ends go here — in `EnvState` the auto-reset splice would overwrite them one
call later. Consumers read fields by name and tolerate absence (an env without the component
simply lacks the field).

**The observation** — the policy's censored view, built by `observation(state)`:
`{sub_index [B,1,A,W], derived_sub_indices [B,G,A,W], action_mask [B,G]}` plus component
`obs_extra` keys. Bookkeeping (depths, history, pointers, counters) is never exposed. Each
component also declares the matching `obs_space_extra` entry under the same condition, so the
declared space always equals the emitted obs.

## Extending it

Four tiers, shallow to deep. The living template for the first three is `examples/mnist/` —
a second real task (ternary atoms, a different engine) whose module mirrors the kge anatomy:
`data_loader.py`, `config.py`, `env.py`, `runner.py`, `tests/`.

### Tier 1 — new task or dataset: write a config

Implement the `base.data_loader.DataLoader` protocol — the symbolic program (`constants`,
`predicates`, `rules`, `max_arity`, `padding_idx`) plus `materialize(im, device)` returning the
KB tensors and query splits — and select it from a config:

```python
@dataclass
class MyConfig(BaseEnvConfig):
    def make_data_loader(self): return MyLoader()      # the only required hook
    # optional: engine_cls() → SLD for arity > 2 · core_cls() → your env subclass
    #           make_components() → extra features · special_predicates() → extra markers

core = build_env(MyConfig())
```

Mind the ground rules: flat atoms, single-atom queries (wrap conjunctions in a rule),
`max_steps` bounds proof depth, and the builder sizes `padding_states` (the action width) from
the engine's `max_children`.

### Tier 2 — new feature: write a component

Subclass `EnvComponent`, override only the hooks you need (all are no-ops), take your knobs as
constructor args, and append the instance in your config's `make_components()`. Declared state
fields ride the composed `EnvState` automatically — including cloning, freezing, and the
auto-reset splice. `FieldSpec(..., eval_roundtrip=False)` marks train-only fields the compiled
eval scorer may skip.

| Hook | Fires | Typical use |
|---|---|---|
| `setup(env)` | once, eager | cache constants / allocate static buffers |
| `declare_state_fields(env)` | build | recurrent `FieldSpec`s for the composed `EnvState` |
| `declare_terminal_pred_ids(env)` | build | predicate ids that end a proof (unioned into `env.terminal_pred_ids`) |
| `reset_seed_fields` / `reset_commit_fields` | reset | seed your fields from queries / draw seeds |
| `reset_draw_queries` | reset draw | transform the drawn queries (KGE corruption); return your seeds |
| `step_update_fields` | step, pre-derive | update fields from the just-entered state |
| `candidate_keep_mask` | reset + step | `[B,G]` veto over raw candidates (drop visited) |
| `candidate_refine` | reset + step | rewrite the compacted candidates (auto-advance, add a stop action) |
| `step_transform_candidates` | step, post-freeze | overwrite candidates from step context (depth-limit forcing) |
| `step_commit_fields` | step | commit field updates into the next state |
| `step_terminal_outcome` | step | extra terminal semantics → `(terminated, is_success, is_end)` |
| `obs_extra` / `obs_space_extra` | obs / build | expose a key to the policy + declare its space (same gate) |
| `declare_trace_fields` / `step_trace` | build / step | add `StepOutput` fields; `step_trace` receives `TransitionCtx` |

A complete component — a visited-step counter exposed to the policy:

```python
class DepthFeatureComponent(EnvComponent):
    name = "depth_feature"
    def declare_state_fields(self, env):
        return (FieldSpec("visited", lambda e, B: torch.zeros(B, dtype=torch.long, device=e.device)),)
    def step_update_fields(self, env, current, fields, mask):
        return {"visited": fields["visited"] + mask.long()}
    def obs_extra(self, env, state):
        return {"visited": state.visited}
    def obs_space_extra(self, env):
        return {"visited": gym.spaces.Box(0, 2 ** 31 - 1, shape=(), dtype=np.int64)}
```

**Rules for hook bodies** (everything except `setup` runs inside the compiled step): fixed
shapes only; no `.item()` / host syncs; never branch python-side on tensor *values* — use
`torch.where`. Branching on your own build-time config flags is fine (it specializes the trace).
“Optional” always means the component is present or absent at build — never a `None`-valued
tensor field at runtime.

### Tier 3 — new reward: override `_reward`

```python
class MyEnv(FuncEnv):
    def _reward(self, state, step_done, is_success, active):
        ...  # [B] float; read task fields off `state` (components put them there)
```

Return the class from `core_cls()` in your config. `KGEFuncEnv` is the reference (labels →
4-way reward). Subclass the env *only* for task infrastructure like this; behavior and state
belong in components — the MNIST agnosticism test enforces that its env subclass overrides
nothing.

### Tier 4 — new logic engine: implement `Grounder`

The engine is the layer below the env: anything satisfying `base.unification.Grounder` plugs in
via `config.engine_cls()`. One implementation ships: `SLD`, the arity-general single-step SLD
engine (atom width follows the program tensors — KGE runs W=3, the MNIST example W=6). The
protocol is four sizing attributes + one verb:

```python
max_children: int      # max successors per state → sizes padding_states
total_vocab_size: int  # token-id ceiling and collision-free hash base
n_vars: int            # runtime-variable table size
num_rules: int         # program size

derive(current_states [B,A,W], next_var [B], excluded=None)
    -> (derived [B,G,A,W], counts [B], next_var [B], derived_rule_idx [B,G])
```

Semantic contract (also in the protocol docstring): a successor with zero non-padding atoms
signals a completed proof (the env collapses it to a `TRUE` state); `counts[b] == 0` signals a
dead end (the env substitutes a `FALSE` state); `excluded` is the episode's root query, to be
skipped during fact unification (cycle prevention); atoms stay flat. Satisfy that and every
layer above — components, PPO, evaluators — runs unchanged.

## Tests that guard this folder

- `sb3/tests/` — behavioral parity against the frozen SB3 reference implementation (the
  don't-drift gate); `test_gym_vecenv_parity.py` pins the gym SAME_STEP contract.
- `examples/mnist/tests/test_component_compile.py` — every component combination must compile
  `fullgraph=True` with zero graph breaks: the contract that keeps components freely mixable.
- `examples/mnist/tests/test_mnist_agnostic.py` — a second task rides this env with zero edits
  here, and its env subclass overrides nothing.
- `kge/tests/` — end-to-end train/eval speed + MRR gates; the rollout/loss/eval compile
  `fullgraph=True`, so any component-induced graph break or recompile fails them too.

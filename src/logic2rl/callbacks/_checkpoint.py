"""CheckpointCallback — saves model checkpoints based on best metrics."""

import glob
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ._base import BaseCallback

logger = logging.getLogger(__name__)


class CheckpointCallback(BaseCallback):
    """
    Saves checkpoints based on best training reward and best evaluation metric.
    """

    def __init__(
        self,
        save_path: str,
        policy: Any,
        best_model_name_train: str = "best_model_train.pt",
        best_model_name_eval: str = "best_model_eval.pt",
        train_metric: str = "ep_rew_mean",
        eval_metric: str = "eval/success_rate",  # generic; the builder passes the task metric (e.g. KGE 'mrr_mean')
        verbose: bool = True,
        date: str = None,
        restore_best: bool = False,
        load_best_metric: str = 'eval',
        load_model: Any = False,
        save_mode: str = "best",
        single_file: bool = False,
    ):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.best_model_name_train = best_model_name_train
        self.best_model_name_eval = best_model_name_eval

        self.train_metric = train_metric
        self.eval_metric = eval_metric

        self.best_train_value = float('-inf')
        self.best_eval_value = float('-inf')
        self.verbose = verbose
        self.date = date
        self.restore_best = restore_best
        self.load_best_metric = load_best_metric
        self.load_model = load_model
        self.save_mode = save_mode
        self.single_file = single_file

        self._total_timesteps_config = 0 # Track for end-of-training behavior
        self._single_model_path = self.save_path / "model.safetensors"
        self._single_info_path = self.save_path / "model_info.json"

    def _infer_device(self) -> Any:
        try:
            return next(self.policy.parameters()).device
        except Exception:
            return 'cpu'

    def _write_single_model_info(
        self,
        *,
        metric_name: Optional[str],
        metric_value: Optional[float],
        iteration: int,
        global_step: Optional[int],
    ) -> None:
        payload = {
            "save_mode": self.save_mode,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "iteration": int(iteration),
            "global_step": int(global_step) if global_step is not None else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._single_info_path.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)

    def _save_single_model(
        self,
        *,
        metric_name: Optional[str],
        metric_value: Optional[float],
        iteration: int,
        global_step: Optional[int],
    ) -> None:
        torch.save(self.policy.state_dict(), self._single_model_path)
        self._write_single_model_info(
            metric_name=metric_name,
            metric_value=metric_value,
            iteration=iteration,
            global_step=global_step,
        )
        if self.verbose:
            logger.info(
                "[Checkpoint] Saved %s model to %s (%s=%s, step=%s)",
                self.save_mode,
                self._single_model_path,
                metric_name,
                metric_value,
                global_step,
            )

    def _load_single_model(self, path: Path, device: Any = None) -> bool:
        if not path.exists():
            logger.warning("Checkpoint not found at %s", path)
            return False
        if device is None:
            device = self._infer_device()
        state_dict = self._adapt_state_dict(torch.load(path, map_location=device))
        self.policy.load_state_dict(state_dict)
        return True

    def _adapt_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Hook to reconcile a loaded ``state_dict`` with the current policy.

        Generic base behavior is identity (shapes are expected to match).
        Tasks whose checkpoint vocab can differ from the live policy (e.g. the
        KGE entity/relation embedding tables) override this to resize the
        mismatched tensors. Base carries no task-specific resize logic.
        """
        return state_dict

    def on_training_start(self, total_timesteps: Optional[int] = None) -> None:
        """Called at the start of training to load an existing model if specified."""
        self._total_timesteps_config = total_timesteps or 0
        if self.load_model:
            if self.verbose:
                logger.info("[Checkpoint] Loading existing model at start of training (load_model=%s)...", self.load_model)

            if self.single_file:
                if isinstance(self.load_model, str) and self.load_model not in ["True", "eval", "train"]:
                    explicit_path = Path(self.load_model)
                    if explicit_path.exists():
                        if self.verbose:
                            logger.info("Loading model from path: %s", explicit_path)
                        self._load_single_model(explicit_path, device=self._infer_device())
                        return
                if self._single_model_path.exists():
                    if self.verbose:
                        logger.info("Loading run-local model from %s", self._single_model_path)
                    self._load_single_model(self._single_model_path, device=self._infer_device())
                    return

            # If load_model is a path string, load it directly
            if isinstance(self.load_model, str) and self.load_model not in ["True", "eval", "train"]:
                path = Path(self.load_model)
                if path.exists():
                    if self.verbose:
                        logger.info("Loading model from path: %s", path)
                    self.policy.load_state_dict(torch.load(path, map_location=next(self.policy.parameters()).device))
                    return

            # Otherwise use the standard load_best_model logic
            load_metric = 'eval'
            if isinstance(self.load_model, str) and self.load_model in ["eval", "train"]:
                load_metric = self.load_model

            self.load_best_model(load_metric=load_metric)

    def on_iteration_end(self) -> bool:
        """Consumer: read the shared metrics dict and save the best/last model."""
        self.check_and_save(self.model.last_metrics, self.model.iteration,
                            global_step=self.model.num_timesteps)
        return True

    def on_training_end(self) -> None:
        """Called at the end of training to optionally restore best model."""
        if self.restore_best and self._total_timesteps_config > 0:
            if self.verbose:
                logger.info("[Checkpoint] Restoring best model at end of training...")

            if self.single_file:
                self._load_single_model(self._single_model_path, device=self._infer_device())
                return

            # Determine device from policy
            device = None
            try:
                device = next(self.policy.parameters()).device
            except:
                pass

            self.load_best_model(load_metric=self.load_best_metric, device=device)

    def check_and_save(self, metrics: Dict[str, Any], iteration: int, global_step: Optional[int] = None) -> None:
        """Called manually or by Manager if we enhance the API."""

        if self.single_file:
            if self.save_mode == "best":
                if self.eval_metric not in metrics:
                    return
                val = metrics[self.eval_metric]
                if isinstance(val, (list, tuple, np.ndarray)):
                    val = np.mean(val)
                if isinstance(val, str):
                    try:
                        val = float(val.split()[0])
                    except Exception:
                        val = float('-inf')
                if val > self.best_eval_value:
                    self.best_eval_value = float(val)
                    self._save_single_model(
                        metric_name=self.eval_metric,
                        metric_value=float(val),
                        iteration=iteration,
                        global_step=global_step,
                    )
                return

            metric_name: Optional[str] = None
            metric_value: Optional[float] = None
            for candidate in (self.eval_metric, self.train_metric):
                if candidate in metrics:
                    metric_name = candidate
                    raw_value = metrics[candidate]
                    if isinstance(raw_value, (list, tuple, np.ndarray)):
                        raw_value = np.mean(raw_value)
                    if isinstance(raw_value, str):
                        try:
                            raw_value = float(raw_value.split()[0])
                        except Exception:
                            raw_value = None
                    metric_value = float(raw_value) if raw_value is not None else None
                    break
            self._save_single_model(
                metric_name=metric_name or self.eval_metric,
                metric_value=metric_value,
                iteration=iteration,
                global_step=global_step,
            )
            return

        # 1. Check Eval Metric (RankingCallback output)
        if self.eval_metric in metrics:
            val = metrics[self.eval_metric]
            # Handle formatted string/list if necessary, though typical Eval callbacks return floats for main metrics
            if isinstance(val, (list, tuple, np.ndarray)):
                 val = np.mean(val) # Should not happen usually
            if isinstance(val, str):
                try: val = float(val.split()[0])
                except: val = float('-inf')

            if val > self.best_eval_value:
                self.best_eval_value = val

                if self.date:
                    stem = self.best_model_name_eval.replace('.pt', '')
                    filename = f"{stem}_{self.date}.pt"
                else:
                    filename = self.best_model_name_eval

                path = self.save_path / filename
                torch.save(self.policy.state_dict(), path)

                # SAVE JSON INFO
                if self.date:
                    json_filename = f"info_best_eval_{self.date}.json"
                    json_path = self.save_path / json_filename

                    # Try to get simplified types for json
                    explained_var = metrics.get("explained_var", None)
                    if isinstance(explained_var, torch.Tensor):
                        explained_var = explained_var.item()
                    elif isinstance(explained_var, (np.float32, np.float64)):
                        explained_var = float(explained_var)

                    info = {
                        "metric": self.eval_metric,
                        "best_value": float(val),
                        "timesteps": iteration,
                        "explained_variance": explained_var
                    }
                    try:
                        with open(json_path, 'w') as f:
                            json.dump(info, f, indent=4)
                    except Exception as e:
                        logger.warning("Failed to save best model info json: %s", e)

                if self.verbose:
                    logger.info("[Checkpoint] New best eval model saved to %s (%s=%.4f)", path, self.eval_metric, val)

        # Always save "last" train model whenever checkpoint is called
        stem = "last_model_train"
        if self.date:
            filename = f"{stem}_{self.date}.pt"
        else:
            filename = f"{stem}.pt"

        path = self.save_path / filename
        torch.save(self.policy.state_dict(), path)
        if self.verbose:
            logger.info("[Checkpoint] Saved last train model to %s", path)

    def load_best_model(self, load_metric: str = 'eval', device: Any = None) -> bool:
        """
        Loads the best model based on the specified metric ('eval' or 'train').
        If the preferred model is not found, falls back to the other one if available.
        """
        if self.single_file:
            if isinstance(self.load_model, str) and self.load_model not in ["True", "eval", "train"]:
                path_to_load = Path(self.load_model)
            else:
                path_to_load = self._single_model_path
            if path_to_load.exists():
                logger.info("Restoring run-local %s model from %s", self.save_mode.upper(), path_to_load)
                return self._load_single_model(path_to_load, device=device)
            logger.warning("No run-local model found to restore at %s", path_to_load)
            return False

        path_to_load = None
        if self.date:
             stem_train = self.best_model_name_train.replace('.pt', '')
             name_train = f"{stem_train}_{self.date}.pt"
             stem_eval = self.best_model_name_eval.replace('.pt', '')
             name_eval = f"{stem_eval}_{self.date}.pt"
        else:
             name_train = self.best_model_name_train
             name_eval = self.best_model_name_eval

        best_model_path_train = self.save_path / name_train
        best_model_path_eval = self.save_path / name_eval

        if load_metric == 'train':
            path_to_load = best_model_path_train
            if not path_to_load.exists():
                 # Fallback to non-dated file
                 path_to_load = self.save_path / self.best_model_name_train

            if not path_to_load.exists():
                # Search for latest dated train model
                stem = self.best_model_name_train.replace('.pt', '')
                files = sorted(glob.glob(str(self.save_path / f"{stem}_*.pt")))
                if files:
                    path_to_load = Path(files[-1])

            if path_to_load.exists():
                logger.info("Restoring best TRAIN model (reward) from %s", path_to_load)
            else:
                 logger.warning("Best train model not found at %s or dated version.", path_to_load)
                 path_to_load = None
        else: # eval
            path_to_load = best_model_path_eval
            if not path_to_load.exists():
                 # Fallback to non-dated file
                 path_to_load = self.save_path / self.best_model_name_eval

            if not path_to_load.exists():
                # Search for latest dated eval model
                stem = self.best_model_name_eval.replace('.pt', '')
                files = sorted(glob.glob(str(self.save_path / f"{stem}_*.pt")))
                if files:
                    path_to_load = Path(files[-1])

            if path_to_load.exists():
                 logger.info("Restoring best EVAL model (MRR) from %s", path_to_load)
            else:
                 logger.warning("Best eval model not found at %s. Falling back to train model?", path_to_load)
                 path_to_load = best_model_path_train
                 if not path_to_load.exists():
                      path_to_load = self.save_path / self.best_model_name_train

                 if not path_to_load.exists():
                     # Search for latest dated train model as last resort
                     stem = self.best_model_name_train.replace('.pt', '')
                     files = sorted(glob.glob(str(self.save_path / f"{stem}_*.pt")))
                     if files:
                         path_to_load = Path(files[-1])

                 if path_to_load.exists():
                      logger.info("Restoring best TRAIN model instead from %s", path_to_load)
                 else:
                      path_to_load = None

        if path_to_load:
            if device is None:
                # Try to infer device from policy
                try:
                    device = next(self.policy.parameters()).device
                except:
                    device = 'cpu'

            state_dict = torch.load(path_to_load, map_location=device)
            state_dict = self._adapt_state_dict(state_dict)
            self.policy.load_state_dict(state_dict)

            # Load and print JSON info if available
            if self.date and load_metric == 'eval':
                json_filename = f"info_best_eval_{self.date}.json"
                json_path = self.save_path / json_filename
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            info = json.load(f)
                        logger.info("Loaded best model info: %s", info)
                    except Exception as e:
                        logger.warning("Failed to load info json: %s", e)

            return True
        else:
            logger.warning("No best model found to restore.")
            return False

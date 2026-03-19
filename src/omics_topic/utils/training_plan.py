from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
from pyro.infer import Trace_ELBO, TraceMeanField_ELBO
from scvi._constants import REGISTRY_KEYS
from scvi.train import PyroTrainingPlan

# Suppress Pyro's mean-field site ordering warning — our model/guide site order
# differs but the mean-field factorization is correct.
warnings.filterwarnings(
    "ignore",
    message="Failed to verify mean field restriction on the guide",
    module="pyro.infer.trace_mean_field_elbo",
)

logger = logging.getLogger(__name__)


class MultimodalLDAPyroTrainingPlan(PyroTrainingPlan):
    """
    Pyro training plan that also logs validation ELBO.

    The default `PyroTrainingPlan` logs the training ELBO via the SVI loss.
    This subclass mirrors that behaviour for the validation split by computing
    the ELBO on each validation batch and logging it as ``elbo_val``.

    Supports KL annealing (warmup) to prevent posterior collapse.
    """

    def __init__(
        self,
        *args: Any,
        n_obs_validation: int | None = None,
        kl_warmup_fraction: float = 0.25,
        kl_warmup_min: float = 0.01,
        **kwargs: Any,
    ) -> None:
        # Default to TraceMeanField_ELBO (analytic KL, lower variance gradients)
        if "loss_fn" not in kwargs:
            kwargs["loss_fn"] = TraceMeanField_ELBO()
        # Default to AdamW optimizer (better weight decay handling than Adam)
        if "optim" not in kwargs:
            import pyro.optim
            optim_kwargs = kwargs.pop("optim_kwargs", None) or {}
            if "lr" not in optim_kwargs:
                optim_kwargs["lr"] = 1e-3
            kwargs["optim"] = pyro.optim.AdamW(optim_args=optim_kwargs)
        super().__init__(*args, **kwargs)
        # Needed so Trace_ELBO scaling matches training behaviour
        self.n_obs_validation = n_obs_validation
        self.validation_step_outputs: list[dict[str, float]] = []

        # KL annealing parameters
        self.kl_warmup_fraction = kl_warmup_fraction  # Fraction of epochs for warmup
        self.kl_warmup_min = kl_warmup_min  # Starting KL weight (typically 0)
        self._kl_weight_final = getattr(self.module, 'kl_weight', 1.0)  # Store target

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _batch_library_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-modality library sizes for a mini-batch."""
        libs = []
        cursor = 0
        for F_m in getattr(self.module, "n_inputs_modalities", []):
            libs.append(x[:, cursor : cursor + F_m].sum(dim=1))
            cursor += F_m
        return torch.stack(libs, dim=1) if libs else torch.empty(0, device=x.device)

    def _log_validation_elbo(self, batch: dict[str, torch.Tensor]) -> float | None:
        """Compute and log validation ELBO (higher is better)."""
        x = batch[REGISTRY_KEYS.X_KEY]

        n_obs = self.n_obs_validation
        if n_obs is None:
            dm = getattr(self.trainer, "datamodule", None)
            val_indices = getattr(dm, "val_indices", None) if dm is not None else None
            n_obs = len(val_indices) if val_indices is not None else x.shape[0]

        # Use the module's batch parser so categorical/continuous covariates are preserved.
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        kwargs["n_obs"] = n_obs
        loss = Trace_ELBO().loss(self.module.model, self.module.guide, *args, **kwargs)
        # Keep sign consistent with elbo_train (logged as the raw loss)
        elbo = float(loss)
        self.log(
            "elbo_val",
            elbo,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
            add_dataloader_idx=False,
        )
        return elbo

    def _get_kl_weight(self, current_epoch: int, max_epochs: int) -> float:
        """
        Compute KL weight for current epoch using linear warmup.

        Parameters
        ----------
        current_epoch : int
            Current training epoch (0-indexed)
        max_epochs : int
            Total number of training epochs

        Returns
        -------
        float
            KL weight in range [kl_warmup_min, kl_weight_final]
        """
        if self.kl_warmup_fraction <= 0:
            return self._kl_weight_final

        warmup_epochs = max(1, int(max_epochs * self.kl_warmup_fraction))
        if current_epoch >= warmup_epochs:
            return self._kl_weight_final

        # Linear interpolation from min to final
        progress = current_epoch / warmup_epochs
        return self.kl_warmup_min + progress * (self._kl_weight_final - self.kl_warmup_min)

    # ------------------------------------------------------------------ #
    # Lightning hooks
    # ------------------------------------------------------------------ #
    def on_fit_start(self):
        n_val_batches = sum(self.trainer.num_val_batches) if hasattr(self.trainer, "num_val_batches") else 0
        # Initialize KL weight at start
        if self.kl_warmup_fraction > 0:
            kl_weight = self._get_kl_weight(0, self.trainer.max_epochs)
            self.module.kl_weight = kl_weight
            logger.info(f"KL annealing enabled: warmup over {self.kl_warmup_fraction*100:.0f}% of epochs")

    def on_train_epoch_start(self):
        """Update KL weight at start of each epoch for annealing."""
        if self.kl_warmup_fraction > 0:
            current_epoch = self.trainer.current_epoch
            max_epochs = self.trainer.max_epochs
            kl_weight = self._get_kl_weight(current_epoch, max_epochs)
            self.module.kl_weight = kl_weight
            # Log the current KL weight
            self.log("kl_weight", kl_weight, on_epoch=True, prog_bar=False, logger=True)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        """
        Validation step logs ``elbo_val`` (positive ELBO) and entropy to the progress bar/logger.

        Lightning will only run this if a validation dataloader exists; with
        ``validation_size=0`` or ``train_size=1.0`` the loop is skipped.
        """
        elbo = self._log_validation_elbo(batch)
        output = {}
        if elbo is not None:
            output["elbo_val"] = elbo

        # Log entropy term if available
        if hasattr(self.module, 'guide') and hasattr(self.module.guide, '_last_entropy'):
            entropy = self.module.guide._last_entropy
            if entropy is not None and self.module.entropy_weight > 0:
                self.log(
                    "entropy_mean_val",
                    entropy,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch[REGISTRY_KEYS.X_KEY].shape[0],
                )
                output["entropy_mean_val"] = float(entropy)

        # Log topic variance term if available
        if hasattr(self.module, 'guide') and hasattr(self.module.guide, '_last_topic_variance'):
            variance = self.module.guide._last_topic_variance
            if variance is not None and self.module.topic_variance_weight > 0:
                self.log(
                    "topic_variance_mean_val",
                    variance,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch[REGISTRY_KEYS.X_KEY].shape[0],
                )
                output["topic_variance_mean_val"] = float(variance)

        if output:
            self.validation_step_outputs.append(output)
        return None

    def on_validation_epoch_end(self):
        """Aggregate validation ELBO and entropy across batches."""
        if not self.validation_step_outputs:
            return
        elbos = [x["elbo_val"] for x in self.validation_step_outputs if "elbo_val" in x]
        if elbos:
            mean_elbo = sum(elbos) / len(elbos)
            self.log("elbo_val", mean_elbo, on_epoch=True, prog_bar=True, logger=True, batch_size=1)

        # Aggregate entropy if present
        entropies = [x["entropy_mean_val"] for x in self.validation_step_outputs if "entropy_mean_val" in x]
        if entropies:
            mean_entropy = sum(entropies) / len(entropies)
            self.log("entropy_mean_val", mean_entropy, on_epoch=True, prog_bar=False, logger=True, batch_size=1)

        # Aggregate topic variance if present
        variances = [x["topic_variance_mean_val"] for x in self.validation_step_outputs if "topic_variance_mean_val" in x]
        if variances:
            mean_variance = sum(variances) / len(variances)
            self.log("topic_variance_mean_val", mean_variance, on_epoch=True, prog_bar=False, logger=True, batch_size=1)

        self.validation_step_outputs.clear()

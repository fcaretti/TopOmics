from __future__ import annotations

import logging
from typing import Any

import torch
from scvi._constants import REGISTRY_KEYS
from scvi.train import PyroTrainingPlan

logger = logging.getLogger(__name__)


class MultimodalLDAPyroTrainingPlan(PyroTrainingPlan):
    """
    Pyro training plan that also logs validation ELBO.

    The default `PyroTrainingPlan` logs the training ELBO via the SVI loss.
    This subclass mirrors that behaviour for the validation split by computing
    the ELBO on each validation batch and logging it as ``elbo_val``.
    """

    def __init__(self, *args: Any, n_obs_validation: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Needed so Trace_ELBO scaling matches training behaviour
        self.n_obs_validation = n_obs_validation
        self.validation_step_outputs: list[dict[str, float]] = []

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
        libs = self._batch_library_tensor(x)
        if libs.numel() == 0:
            logger.debug("Skipping validation ELBO logging; could not build library tensor.")
            return None

        n_obs = self.n_obs_validation
        if n_obs is None:
            dm = getattr(self.trainer, "datamodule", None)
            val_indices = getattr(dm, "val_indices", None) if dm is not None else None
            n_obs = len(val_indices) if val_indices is not None else x.shape[0]

        # module.get_elbo returns the Pyro Trace_ELBO loss (negative ELBO)
        loss = self.module.get_elbo(x, libs, n_obs)
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

    # ------------------------------------------------------------------ #
    # Lightning hooks
    # ------------------------------------------------------------------ #
    def on_fit_start(self):
        n_val_batches = sum(self.trainer.num_val_batches) if hasattr(self.trainer, "num_val_batches") else 0

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        """
        Validation step logs ``elbo_val`` (positive ELBO) to the progress bar/logger.

        Lightning will only run this if a validation dataloader exists; with
        ``validation_size=0`` or ``train_size=1.0`` the loop is skipped.
        """
        elbo = self._log_validation_elbo(batch)
        if elbo is not None:
            self.validation_step_outputs.append({"elbo_val": elbo})
        return None

    def on_validation_epoch_end(self):
        """Aggregate validation ELBO across batches."""
        if not self.validation_step_outputs:
            return
        elbos = [x["elbo_val"] for x in self.validation_step_outputs if "elbo_val" in x]
        if elbos:
            mean_elbo = sum(elbos) / len(elbos)
            self.log("elbo_val", mean_elbo, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.validation_step_outputs.clear()

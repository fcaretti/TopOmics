# multimodal_lda_module.py
"""Multimodal Amortized Latent Dirichlet Allocation (MM-LDA)

================================================================
A full Pyro / **scvi-tools** implementation that
* keeps **one shared topic mixture** per cell, *θₙ*,
* gives **each modality its own topic-by-feature distribution** ϕₖ,ₘ and its **own likelihood**
  (Multinomial for sparse discrete counts, Gamma-Poisson/Negative Binomial for RNA, etc.),
* uses the **encode-then-mix** inference strategy (one encoder per modality, mixed Gaussian
  parameters),
* is drop-in compatible with the scvi-tools training loop (inherits `PyroBaseModuleClass`).

The file declares three public objects
-------------------------------------
* `MultimodalLDAPyroModel`   – generative process with modality plate & mixed likelihoods.
* `MultimodalLDAPyroGuide`   – per-modality encoders + combined θₙ posterior + per-modality ϕₖ,ₘ posterior.
* `MultimodalAmortizedLDAPyroModule` – thin wrapper pairing the two above and exposing
  helper utilities (`topic_by_feature`, `get_topic_distribution`, `get_elbo`).

A higher-level `ModelClass` wrapper (*MultimodalAmortizedLDA*) is provided at the bottom so that
users get the usual `train()`/`get_latent_representation()`/etc. API.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import Trace_ELBO
from pyro.nn import PyroModule
from scvi._constants import REGISTRY_KEYS
from scvi.module.base import PyroBaseModuleClass, auto_move_data
from scvi.nn import Encoder

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Helper utils
# --------------------------------------------------------------------------------------------------


def logistic_normal_approximation(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Laplace approximation parameters (μ, σ) of Logistic-Normal ≈ Dirichlet(α)."""
    K = alpha.shape[-1]
    mu = torch.log(alpha) - torch.log(alpha).sum() / K
    sigma = torch.sqrt((1 - 2 / K) / alpha + torch.sum(1 / alpha) / K**2)
    return mu, sigma


def masked_softmax(weights: torch.Tensor, mask: torch.Tensor, dim: int = 0):
    """Softmax **ignoring** masked entries (mask == 0)."""
    weights = weights.masked_fill(~mask.bool(), -1e9)
    return torch.softmax(weights, dim=dim)


class CategoricalBoW(dist.Multinomial):
    """Multinomial that tolerates −inf logits for zero counts (identical to scvi implementation)."""

    def log_prob(self, value):  # type: ignore[override]
        if self._validate_args:
            self._validate_sample(value)
        logits, value = dist.util.broadcast_all(self.logits, value)
        logits = logits.clone(memory_format=torch.contiguous_format)
        logits[(value == 0) & (logits == -math.inf)] = 0
        return (logits * value).sum(-1)


# --------------------------------------------------------------------------------------------------
# Generative model with modality plate
# --------------------------------------------------------------------------------------------------


class MultimodalLDAPyroModel(PyroModule):
    """Generative process

    For k in {1..K}, m in {1..M}::
        ϕₖ,ₘ  ~ LogisticNormal(α)          # topic by feature for modality m
    For cell n::
        θₙ ~ LogisticNormal(β)
        For modality m::
            xₙ,ₘ ~   Multinomial(Nₙ, θₙ·ϕₘ)    if likelihood[m] == "multinomial"
                    GammaPoisson(μ, rₘ)        if likelihood[m] == "gamma_poisson"
    """

    def __init__(
        self,
        n_inputs_modalities: list[int],
        likelihoods: list[str],
        n_topics: int,
        cell_topic_prior: torch.Tensor,
        topic_feature_priors: list[torch.Tensor],  # one tensor per modality (len = M)
        dispersion_rna: float = 10.0,  # global NB dispersion for Gamma-Poisson modalities
    ) -> None:
        super().__init__("multimodal_lda")

        assert len(n_inputs_modalities) == len(likelihoods) == len(topic_feature_priors)
        self.n_modalities = len(n_inputs_modalities)
        self.n_topics = n_topics
        self.n_inputs_modalities = n_inputs_modalities
        self.likelihoods = likelihoods
        self.dispersion_rna = dispersion_rna  # used for Gamma-Poisson / NB

        # Pre-compute Logistic-Normal approximations for priors
        cell_mu, cell_sigma = logistic_normal_approximation(cell_topic_prior)
        self.register_buffer("cell_mu", cell_mu)
        self.register_buffer("cell_sigma", cell_sigma)

        self.topic_prior_mus = torch.nn.ParameterList()
        self.topic_prior_sigmas = torch.nn.ParameterList()
        for t_prior in topic_feature_priors:
            mu_m, sig_m = logistic_normal_approximation(t_prior)
            self.topic_prior_mus.append(torch.nn.Parameter(mu_m, requires_grad=False))
            self.topic_prior_sigmas.append(torch.nn.Parameter(sig_m, requires_grad=False))

        # Populated by training plan for full-batch ELBO scaling
        self.n_obs = None
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # for device

    # ---------- data-loader helper ----------
    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, torch.Tensor]):
        x = tensor_dict[REGISTRY_KEYS.X_KEY]  # concatenated features
        libraries = []
        cursor = 0
        for n_f in tensor_dict["n_inputs_modalities"]:  # injected by wrapper
            libraries.append(x[:, cursor : cursor + n_f].sum(dim=1))
            cursor += n_f
        return (x, torch.stack(libraries, dim=1)), {}

    # ---------- forward ----------
    @auto_move_data
    def forward(
        self,
        x: torch.Tensor,  # (B, ΣF)
        libraries: torch.Tensor,  # (B, M) – per-modality lib sizes
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        # ----- topic-feature distributions (per modality) -----
        topic_feature_dists = []  # will store φₖ,ₘ tensors
        with pyro.plate("topics", self.n_topics):
            for m in range(self.n_modalities):
                # mu_m = self.topic_prior_mus[m]
                # sig_m = self.topic_prior_sigs[m]
                mu_m = self.topic_prior_mus[m].unsqueeze(0).expand(self.n_topics, -1)
                sig_m = self.topic_prior_sigmas[m].unsqueeze(0).expand(self.n_topics, -1)

                with poutine.scale(scale=kl_weight):
                    log_phi = pyro.sample(f"log_topic_feature_dist_{m}", dist.Normal(mu_m, sig_m).to_event(1))
                topic_feature_dists.append(F.softmax(log_phi, dim=-1))  # (K, Fₘ)

        # ----- cells plate -----
        with pyro.plate("cells", size=n_obs or self.n_obs, subsample_size=x.shape[0]):
            # shared θₙ
            with poutine.scale(scale=kl_weight):
                log_theta = pyro.sample("log_cell_topic_dist", dist.Normal(self.cell_mu, self.cell_sigma).to_event(1))
            theta = F.softmax(log_theta, dim=-1)

            # likelihood per modality
            cursor = 0
            for m, (F_m, L_m) in enumerate(zip(self.n_inputs_modalities, self.likelihoods, strict=False)):
                x_m = x[:, cursor : cursor + F_m]
                lib_m = libraries[:, m]
                phi_m = topic_feature_dists[m]  # (K, F_m)
                rate_m = theta @ phi_m  # (B, F_m)

                if L_m == "multinomial":
                    N_max = int(lib_m.max().item())
                    pyro.sample(
                        f"feature_counts_{m}",
                        CategoricalBoW(N_max, rate_m),
                        obs=x_m,
                    )
                elif L_m in {"gamma_poisson", "nb"}:
                    # mean scaled by lib; dispersion shared globally per feature set
                    mu = rate_m * lib_m.unsqueeze(-1)
                    r = torch.tensor(self.dispersion_rna, device=mu.device)
                    pyro.sample(
                        f"feature_counts_{m}",
                        dist.NegativeBinomial(total_count=r, probs=mu / (mu + r)).to_event(1),
                        obs=x_m,
                    )
                else:
                    raise ValueError(f"Unknown likelihood {L_m}")
                cursor += F_m


# --------------------------------------------------------------------------------------------------
# Guide with encode-then-mix
# --------------------------------------------------------------------------------------------------


class MultimodalLDAPyroGuide(PyroModule):
    def __init__(
        self,
        n_inputs_modalities: list[int],
        n_topics: int,
        n_hidden: int,
        weight_mode: str = "equal",
        max_n_obs: int | None = None,
    ) -> None:
        super().__init__("multimodal_lda_guide")
        self.n_modalities = len(n_inputs_modalities)
        self.n_inputs_modalities = n_inputs_modalities
        self.n_topics = n_topics

        self.encoders = torch.nn.ModuleList(
            [
                Encoder(n_in, n_topics, distribution="ln", return_dist=True, n_hidden=n_hidden)
                for n_in in n_inputs_modalities
            ]
        )

        # per-modality topic-feature posterior params
        self.topic_feature_posterior_mu = torch.nn.ParameterList()
        self.unconstrained_topic_feature_posterior_sigma = torch.nn.ParameterList()
        for F_m in n_inputs_modalities:
            mu_m, sig_m = logistic_normal_approximation(torch.ones(F_m))
            self.topic_feature_posterior_mu.append(torch.nn.Parameter(mu_m.repeat(n_topics, 1)))
            self.unconstrained_topic_feature_posterior_sigma.append(torch.nn.Parameter(sig_m.repeat(n_topics, 1)))

        if weight_mode == "equal":
            self.mod_w = None
        elif weight_mode == "universal":
            self.mod_w = torch.nn.Parameter(torch.ones(self.n_modalities))
        elif weight_mode == "cell":
            if max_n_obs is None:
                raise ValueError("Specify `max_n_obs` when weight_mode=='cell'.")
            self.mod_w = torch.nn.Parameter(torch.ones(max_n_obs, self.n_modalities))
        else:
            raise ValueError("weight_mode must be 'equal'|'universal'|'cell'.")

        self.weight_mode = weight_mode
        self.n_obs = None

    @staticmethod
    def _softplus(t: torch.Tensor) -> torch.Tensor:
        return F.softplus(t)

    def _mix_gaussians(
        self, mus: torch.Tensor, vars_: torch.Tensor, masks: torch.Tensor, cell_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mix (*M,B,K*) Gaussians using weights w_{m,c}.

        Parameters
        ----------
        mus/vars_ : (M , B , K)
        masks     : (M , B)  (0 = modality absent)
        """
        if self.mod_w is None:  # equal
            w = torch.ones_like(masks)
        elif self.mod_w.dim() == 1:  # universal
            w = self.mod_w.view(-1, 1).expand_as(masks)
        else:  # cell
            w = self.mod_w[cell_idx, :]  # (B , M) -> transpose
            w = w.T  # (M , B)

        w = masked_softmax(w, masks, dim=0).unsqueeze(-1)  # (M,B,1)
        mu = (w * mus).sum(0)
        var = (w * vars_).sum(0)
        return mu, var

    # property for sigmas
    @property
    def topic_feature_posterior_sigma(self) -> list[torch.Tensor]:
        return [self._softplus(u) for u in self.unconstrained_topic_feature_posterior_sigma]

    # ---------- forward ----------
    @auto_move_data
    def forward(
        self,
        x: torch.Tensor,  # (B , ΣF)
        libraries: torch.Tensor,  # unused – kept for signature parity
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        B = x.shape[0]
        cell_idx = torch.arange(B, device=x.device)  # needed only if weight_mode=="cell"

        # ϕₖ,ₘ variational dists
        for m in range(self.n_modalities):
            with pyro.plate(f"topics_{m}", self.n_topics):
                with poutine.scale(scale=kl_weight):
                    pyro.sample(
                        f"log_topic_feature_dist_{m}",
                        dist.Normal(self.topic_feature_posterior_mu[m], self.topic_feature_posterior_sigma[m]).to_event(
                            1
                        ),
                    )

        # θₙ variational
        xs = torch.split(x, self.n_inputs_modalities, dim=1)  # (B,Fₘ)
        mus, vars_, masks = [], [], []
        for enc, x_m in zip(self.encoders, xs, strict=False):
            q_m, _ = enc(x_m)  # q(z|x)
            mus.append(q_m.loc)
            vars_.append(q_m.scale**2)
            masks.append((x_m.sum(1) > 0).float())
        mus = torch.stack(mus)  # (M,B,K)
        vars_ = torch.stack(vars_)  # (M,B,K)
        masks = torch.stack(masks)  # (M,B)

        muθ, varθ = self._mix_gaussians(mus, vars_, masks, cell_idx)

        with pyro.plate("cells", size=n_obs or self.n_obs, subsample_size=B), poutine.scale(None, kl_weight):
            pyro.sample("log_cell_topic_dist", dist.Normal(muθ, torch.sqrt(varθ)).to_event(1))


# --------------------------------------------------------------------------------------------------
# Wrapper module pairing model & guide
# --------------------------------------------------------------------------------------------------


class MultimodalAmortizedLDAPyroModule(PyroBaseModuleClass):
    def __init__(
        self,
        n_inputs_modalities: list[int],
        likelihoods: list[str],
        n_topics: int,
        n_hidden: int,
        cell_topic_prior: float | Sequence[float] | None = None,
        topic_feature_prior: float | Sequence[float] | None = None,
        weight_mode: str = "equal",
        max_n_obs: int | None = None,
    ):
        super().__init__()
        assert len(n_inputs_modalities) == len(likelihoods)
        self.n_inputs_modalities = n_inputs_modalities
        self.likelihoods = likelihoods
        self.n_topics = n_topics
        self.n_hidden = n_hidden
        self.n_modalities = len(n_inputs_modalities)
        self.weight_mode = weight_mode

        if cell_topic_prior is None:
            cell_topic_prior_tensor = torch.full((n_topics,), 1 / n_topics)
        elif isinstance(cell_topic_prior, float):
            cell_topic_prior_tensor = torch.full((n_topics,), cell_topic_prior)
        else:
            cell_topic_prior_tensor = torch.tensor(cell_topic_prior)

        topic_feature_priors = []
        for F_m in n_inputs_modalities:
            if topic_feature_prior is None:
                topic_feature_priors.append(torch.full((F_m,), 1 / n_topics))
            elif isinstance(topic_feature_prior, float):
                topic_feature_priors.append(torch.full((F_m,), topic_feature_prior))
            else:
                raise ValueError("Pass list/None/float for topic_feature_prior, not sequence")

        self._model = MultimodalLDAPyroModel(
            n_inputs_modalities,
            likelihoods,
            n_topics,
            cell_topic_prior_tensor,
            topic_feature_priors,
        )
        self._guide = MultimodalLDAPyroGuide(
            n_inputs_modalities,
            n_topics,
            n_hidden,
            weight_mode=weight_mode,
            max_n_obs=max_n_obs,
        )

        # We need this method so scvi training plan can create data-loader args
        def _args_from_batch(tdict):
            tdict["n_inputs_modalities"] = n_inputs_modalities  # inject for library calc
            return self._model._get_fn_args_from_batch(tdict)

        self._get_fn_args_from_batch = _args_from_batch

    # proxies
    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    # utilities
    def topic_by_feature(self, n_samples: int = 5_000):
        out = {}
        for m, (mu, sig) in enumerate(
            zip(self.guide.topic_feature_posterior_mu, self.guide.topic_feature_posterior_sigma, strict=False)
        ):
            tbf = torch.mean(
                F.softmax(dist.Normal(mu.detach().cpu(), sig.detach().cpu()).sample((n_samples,)), dim=2),
                dim=0,
            )
            out[m] = tbf  # user can map index→ modality name externally
        return out

    """@auto_move_data
    @torch.inference_mode()
    def get_topic_distribution(self, x: torch.Tensor, n_samples: int = 5_000):
        mu, var = self.guide.encode_combined(x)
        samples = dist.Normal(mu, torch.sqrt(var)).sample((n_samples,))
        return F.softmax(samples, dim=2).mean(0)"""

    def get_topic_distribution(self, x: torch.Tensor, n_samples: int = 5_000) -> torch.Tensor:
        device = next(self._guide.parameters()).device  # cuda:0 or cpu
        x = x.to(device, non_blocking=True)

        B = x.shape[0]
        xs = torch.split(x, self.model.n_inputs_modalities, dim=1)
        # run encoders
        mus, vars_, masks = [], [], []
        for enc, x_m in zip(self.guide.encoders, xs, strict=False):
            q_m, _ = enc(x_m)
            mus.append(q_m.loc)
            vars_.append(q_m.scale**2)
            masks.append((x_m.sum(1) > 0).float())
        mus = torch.stack(mus)
        vars_ = torch.stack(vars_)
        masks = torch.stack(masks)
        muθ, varθ = self.guide._mix_gaussians(mus, vars_, masks, torch.arange(B, device=x.device))
        samps = dist.Normal(muθ, torch.sqrt(varθ)).sample((n_samples,))
        return torch.softmax(samps, -1).mean(0)

    @auto_move_data
    @torch.inference_mode()
    def get_elbo(self, x: torch.Tensor, libs: torch.Tensor, n_obs: int):
        device = next(self._guide.parameters()).device  # cuda:0 or cpu
        x = x.to(device, non_blocking=True)
        return Trace_ELBO().loss(self.model, self.guide, x, libs, n_obs=n_obs)


# --------------------------------------------------------------------------------------------------
# High-level scvi ModelClass
# --------------------------------------------------------------------------------------------------

'''class MultimodalAmortizedLDA(PyroSviTrainMixin, BaseModelClass):
    _module_cls = MultimodalAmortizedLDAPyroModule

    def __init__(
        self,
        adata,
        n_inputs_modalities: List[int],
        likelihoods: List[str],
        n_topics: int = 20,
        n_hidden: int = 128,
        cell_topic_prior: float | Sequence[float] | None = None,
        topic_feature_prior: float | Sequence[float] | None = None,
        weight_mode: str = 'equal',
    ) -> None:
        pyro.clear_param_store()
        super().__init__(adata)
        if sum(n_inputs_modalities) != self.summary_stats.n_vars:
            raise ValueError("Sum of modality feature counts must equal adata.n_vars")
        self.module = self._module_cls(
            n_inputs_modalities,
            likelihoods,
            n_topics,
            n_hidden,
            cell_topic_prior,
            topic_feature_prior,
        )

    # ---------- anndata setup ----------
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(cls, adata, layer: str | None = None, **kwargs):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata_manager = AnnDataManager(
            fields=[LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True)],
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)'''

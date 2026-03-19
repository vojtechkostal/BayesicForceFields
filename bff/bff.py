from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Union,
)

from .bayes.inference import lgp_hyperopt, initialize_mcmc_sampler
from .bayes.gaussian_process import LGPCommittee
from .structures import (
    Specs, MCMCResults, QoIDataset
)
from .io.logs import Logger, print_progress_mcmc
from .tools import rdf_sigmoid_mean

import numpy as np
import torch


# ---- Type aliases (keep signatures readable) ----
PathLike = Union[str, Path]
SpecsLike = Union[str, Path, Dict[str, Any], Specs]
ArrayLike = Union[np.ndarray, torch.Tensor]


class BFFLearner:

    def __init__(
        self,
        *datasets: Sequence[QoIDataset],
        logger: Optional[Logger] = None,
    ) -> None:
        self.datasets = datasets
        self.logger = logger or Logger("BFFLearn")
        self.models: Dict[str, LGPCommittee] = {}

    def train(
        self,
        y_means: Optional[Dict[str, ArrayLike | float | str]] = None,
        n_hyper_max: int = 200,
        committee_size: int = 1,
        test_fraction: float = 0.2,
        device: str = "cuda",
        **kwargs,
    ) -> None:

        y_means = y_means or {}

        self.logger.info("=== Optimizing LGP surrogates ===", level=0)
        self.logger.info("", level=0)

        for dataset in self.datasets:
            qoi = dataset.name

            if not isinstance(dataset, QoIDataset):
                raise ValueError(
                    f"Invalid dataset type: {type(dataset)}. "
                    "Expected QoIDataset or path to a QoIDataset file."
                )

            self.logger.info(f"QoI: {qoi}", level=1)

            fn_model = Path(f"{qoi}.lgp").resolve()
            if fn_model.exists():
                model = LGPCommittee.load(fn_model)
                self.models[qoi] = model

                self.logger.info(
                    f"Model already trained. | MAPE = {model.error:.2f}", level=2)
                self.logger.info("", level=0)
                continue

            mean = y_means.get(qoi, 0)
            if qoi == "rdf" and mean == "sigmoid":
                n_bins = dataset.settings.get("n_bins")
                r_range = dataset.settings.get("r_range")
                mean = rdf_sigmoid_mean(n_bins, r_range, dataset.outputs_ref)

            self.models[qoi] = lgp_hyperopt(
                X=dataset.inputs,
                y=dataset.outputs,
                y_mean=mean,
                test_fraction=test_fraction,
                n_hyper=n_hyper_max,
                committee=committee_size,
                observations=dataset.n_observations,
                nuisance=dataset.nuisance,
                fn_out=fn_model,
                device=device,
                logger=self.logger,
                opt_kwargs=kwargs,
            )

            self.logger.info("", level=0)

    def run(
        self,
        qoi: Optional[Sequence[str]] = None,
        constraint: Optional[Callable] = None,
        priors_disttype: str = "normal",
        max_iter: int = 1000,
        n_walkers: Optional[int] = None,
        fn_chain: PathLike = "./mcmc.h5",
        fn_priors: Optional[PathLike] = "./priors.yaml",
        fn_tau: Optional[PathLike] = "./tau.yaml",
        restart: bool = True,
        device: str = "cuda",
        **kwargs,
    ) -> MCMCResults:

        qoi = set(qoi) if qoi else set(self.models.keys())
        missing = qoi - self.models.keys()
        if missing:
            raise ValueError(
                f"Requested QoI(s) not found in trained models: "
                f"{', '.join(repr(q) for q in missing)}"
            )

        self.logger.info("=== Parameter Learning ===", level=0)
        self.logger.info("", level=0)

        y_ref = {
            dataset.name: dataset.outputs_ref
            for dataset in self.datasets
            if dataset.name in qoi
        }

        p0, priors, sampler = initialize_mcmc_sampler(
            surrogate=self.models,
            y_true=y_ref,
            constraint=constraint,
            n_walkers=n_walkers,
            priors_disttype=priors_disttype,
            device=device,
        )

        print_progress_mcmc(
            sampler,
            p0,
            max_iter=max_iter,
            logger=self.logger,
            restart=restart,
            fn_chain=fn_chain,
            **kwargs
        )

        # tau = sampler.get_autocorr_time(tol=0)
        # mcmc = MCMCResults(chain_src=sampler, priors_src=priors, tau_src=tau)

        # if fn_priors:
        #     mcmc.write_priors(fn_priors)
        # if fn_tau:
        #     mcmc.write_tau(fn_tau)

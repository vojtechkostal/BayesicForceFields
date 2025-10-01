from pathlib import Path

from .bayes.inference import lgp_hyperopt, initialize_mcmc_sampler
from .structures import Specs, OptimizationResults
from .io.logs import Logger, print_progress_mcmc


class BFFOptimizer:
    """
    Main optimization class for Bayesian Force Field (BFF) parameter optimization
    employing Local Gaussian Processes as surrogate models.

    Parameters
    ----------
    train_data : tuple
        Training datasets, each containing QoI (Quantity of Interest).
    logger : Logger, optional
        An instance of a Logger for logging information.
        If not provided, a default Logger with the name "BFF" is used.

    Attributes
    ----------
    train_data : tuple
        The training datasets provided during initialization.
    logger : Logger
        The logger instance used for logging.
    surrogate : dict, optional
        A dictionary to hold the optimized surrogate models for each QoI.
    QoI : list[str], optional
        A list of Quantity of Interest names that will be optimized.

    Methods
    -------
    setup_LGP
        Sets up the LGP surrogates for the specified QoIs
        using the provided training data.
    run
        Runs the MCMC sampling process using
        the optimized surrogates and specified parameters.
    """
    def __init__(
        self, *train_data, specs: str | Path | dict | Specs, logger: Logger = None
    ) -> None:

        self.train_data = train_data
        self.logger = logger or Logger("BFF")
        self.surrogate = None
        self.QoI = None
        self.specs = Specs(specs) if not isinstance(specs, Specs) else specs

        self.logger.info("=== Bayesian Force Field Optimization ===\n", level=0)

    @property
    def _all_qoi(self) -> list[str]:
        """
        Get a list of all unique QoI names from the training datasets.
        """
        return list({qoi for d in self.train_data for qoi in d.qoi_names})

    def setup_LGP(
        self,
        QoI: list[str] = None,
        n_max: int = 200,
        fn_hyper: dict[str | Path] = None,
        means: dict = {'rdf': 'sigmoid'},
        committee: int = 1,
        test_fraction: float = 0.2,
        device: str = 'cuda:0',
        obs_factor: int = 1,
        **kwargs
    ):
        """
        Set up Local Gaussian Process (LGP) surrogates for the specified QoIs.

        Parameters
        ----------
        QoI : list of str, optional
            List of Quantity of Interest names to optimize.
            If None, all QoIs from the training data are used.
        n_max : int, optional
            Maximum number of data points used for hyperparameter optimization
            (default is 200).
        fn_hyper : dict of str or Path, optional
            Dictionary mapping QoI names to file paths for
            loading/saving optimized hyperparameters.
            If None, hyperparameters are optimized from scratch (default is None).
        means : dict, optional
            Dictionary mapping QoI names to their mean values for centering.
            Default is {'rdf': 'sigmoid'}.
        committee : int, optional
            Number of LGP models in the ensemble (default is 1).
        test_fraction : float, optional
            Fraction of the data to reserve for testing (default is 0.2).
        device : str, optional
            Device to perform computations on (e.g., 'cpu' or 'cuda:0')
            (default is 'cuda:0').
        obs_factor : int, optional
            Factor to multiply the number of observations by (default is 1).
        **kwargs
            Additional arguments passed to the MAP optimizer.
        """

        self.QoI = QoI or self._all_qoi
        means = means or {}
        fn_hyper = fn_hyper or {}

        self.logger.info("Optimizing LGP surrogates", level=0)
        self.logger.info("-------------------------", level=0)

        surrogate = {}
        for qoi in self.QoI:

            self.logger.info(f"QoI: {qoi}", level=1)

            trainset = next(d for d in self.train_data if qoi in d.qoi_names)
            N = (trainset.observations.get(qoi) or 0) * obs_factor

            if qoi == "rdf" and means.get("rdf") == "sigmoid":
                y_mean = trainset.rdf_sigmoid_mean
            else:
                y_mean = means.get(qoi, 0.0)

            lgp = lgp_hyperopt(
                X=trainset.X,
                y=trainset.y[qoi],
                y_mean=y_mean,
                fn_hyperparams=fn_hyper.get(qoi),
                test_fraction=test_fraction,
                n_hyper=n_max,
                committee=committee,
                observations=N,
                device=device,
                logger=self.logger,
                opt_kwargs=kwargs
            )

            surrogate[qoi] = lgp

        self.surrogate = surrogate

        self.logger.info("", level=0)

    def run(
        self,
        priors_type: str = 'normal',
        max_iter: int = 100000,
        n_walkers: int = None,
        fn_backend: str | Path = './mcmc.h5',
        fn_priors: str | Path = './priors.yaml',
        fn_tau: str | Path = './tau.npy',
        restart: bool = True,
        device: str = 'cuda:0',
        **kwargs
    ) -> OptimizationResults:

        """
        Run MCMC posterior sampling using the optimized LGP surrogates.

        Parameters
        ----------
        priors_type : str, optional
            Type of prior distributions to use ('normal' or 'uniform')
            (default is 'normal').
        max_iter : int, optional
            Maximum number of MCMC iterations (default is 100000).
        n_walkers : int, optional
            Number of MCMC walkers (default is 5 times the number of parameters).
        fn_backend : str or Path, optional
            Filename for the MCMC backend storage (default is './mcmc.h5').
        fn_priors : str or Path, optional
            Filename to save the priors (default is './priors.yaml').
        fn_tau : str or Path, optional
            Filename to save the autocorrelation times (default is './tau.npy').
        restart : bool, optional
            Whether to restart from the last MCMC state if available (default is True).
        device : str, optional
            Device to perform computations on (e.g., 'cpu' or 'cuda:0')
            (default is 'cuda:0').
        **kwargs
            Additional arguments passed to the progress printing function.

        Returns
        -------
        OptimizationResults
            An object containing the MCMC sampling results, including samples, priors,
            and autocorrelation times.
        """
        if not self.surrogate:
            raise ValueError("LGP surrogate not set up. Call 'setup_LGP' first.")

        y_true = {
            qoi: d.y_ref[qoi]
            for d in self.train_data
            for qoi in self.QoI
            if qoi in d.qoi_names
        }

        p0, priors, sampler = initialize_mcmc_sampler(
            self.surrogate,
            self.specs,
            self.QoI,
            y_true,
            n_walkers,
            priors_type,
            fn_backend,
            restart,
            device
        )

        self.logger.info("MCMC posterior sampling", level=0)
        self.logger.info("-----------------------", level=0)

        print_progress_mcmc(sampler, p0, max_iter, logger=self.logger, **kwargs)
        tau = sampler.get_autocorr_time(tol=0)
        results = OptimizationResults(sampler, priors, tau, self.specs.data)

        if fn_priors:
            results.save_priors(fn_priors)
        if fn_tau:
            results.save_tau(fn_tau)

        return results

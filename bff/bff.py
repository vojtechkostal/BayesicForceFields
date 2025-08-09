import time
import warnings
import multiprocessing as mp

from pathlib import Path
from .bayes.inference import lgp_hyperopt, initialize_mcmc_sampler
from .evaluation.trajectory import (
    analyze_all_trajectories,
    analyze_trajectories_wrapper,
    get_all_settings
)

from .structures import TrainData, OptimizationResults, Specs
from .topology import check_topols
from .io.logs import Logger, print_progress, print_progress_mcmc, format_time


warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="MDAnalysis.coordinates.DCD"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="MDAnalysis.coordinates.XDR"
)


class Optimizer:
    """
    Class optimizing the fixed-charge force field's parameters.
    """

    def __init__(
        self,
        *train_dirs: str | Path,
        verbose: bool = True,
        fn_log: str = None
    ) -> None:
        """
        Initialize the MDSampler class.

        Parameters
        ----------
        train_dirs : str | Path
            Directories containing the training data.
        verbose : bool, optional
            If True, enables verbose logging. Default is True.
        fn_log : str or None, optional
            Path to the log file. If None, logs are not saved to a file.
            Default is None.
        """

        self.reference = None
        self.surrogate = None
        self.loss_components = []

        # Initialize the logger
        self.logger = Logger(fn_log, verbose=verbose)

        self.logger.info('=========================================')
        self.logger.info('     Byaesian Force Field Optimizer      ')
        self.logger.info('=========================================')
        self.logger.info('')
        self.logger.info('> loading training data: in progress...', overwrite=True)

        t0 = time.time()
        self.train_data = TrainData(*train_dirs)
        t1 = time.time()
        self.logger.info(
            f'> loading training sets: {len(train_dirs)} | Done. '
            f'({format_time(t1 - t0)})'
        )
        self.logger.info('')
        self.specs = Specs(self.train_data.fn_specs)

        self.logger.info(f'> molecule: {self.specs.mol_resname}')
        self.logger.info('> parameters:')
        for param in self.specs.bounds_explicit.params:
            self.logger.info(f'  {param}')
        self.logger.info('')

    def load_reference(
        self,
        fn_topol: list[str | Path],
        fn_coord: list[str | Path],
        fn_trjs: list[str | Path],
        start: int = 0,
        stop: int = None,
        step: int = 1,
        **kwargs
    ) -> None:
        """
        fn_topol : List[Union[str, Path]]
            Paths to the .itp topology files.
        Parameters
        ----------
        fn_topol : str
            Path to the .itp topology file.
        fn_coord : str
            Path to the .gro/.pdb coordinate file.
        fn_trjs : list | str
            Paths to the trajectory file(s).
        start : int, optional
            Frame index to start at. Default is 0.
        stop : int, optional
            Frame index to finish at. Default is None.
        step : int, optional
            Frame stride for sampling frames. Default is 1.
        """

        self.logger.info(
            '> loading reference trajectories: in progress',
            overwrite=True
        )

        # Make sure that the atomtype in reference and training set match
        for top_ref, top_train in zip(fn_topol, self.train_data.fn_topol):
            check_topols(top_ref, top_train)

        # Load settings to match the training set
        settings = self.train_data.settings or kwargs

        # Analyze the reference trajectories
        t0 = time.time()
        reference = analyze_all_trajectories(
            fn_topol, fn_coord, fn_trjs, self.train_data.restraints,
            self.specs.mol_resname,
            start, stop, step, **settings)
        self.train_data.load_reference(reference)
        t1 = time.time()
        self.logger.info(
            f'> loading reference trajectories: Done. ({format_time(t1 - t0)})'
        )
        self.logger.info('')

    def load_train(
        self,
        workers: int = 1,
        fn_in: str | Path = None,
        fn_out: str | Path = None,
        start: int = 0,
        stop: int = None,
        step: int = 1,
        progress_stride: int = 10,
        **kwargs
    ) -> None:
        """
        Analyze the training set trajectories.

        Parameters
        ----------
        workers : int, optional
            Number of parallel workers. Default is 1.
        fn_out : str, optional
            Path to the output JSON file to save the results.
        start : int, optional
            Frame index to start at. Default is 0.
        stop : int, optional
            Frame index to finish at. Default is None.
        step : int, optional
            Frame stride for sampling frames. Default is 1.
        """

        # Load the analyzed samples from a file
        t0 = time.time()
        if fn_in:
            self.logger.info(
                '> loading training trajectories: from file', overwrite=True
            )
            self.train_data.load_features(fn_in)

        # Analyze the samples from scratch
        else:
            args_list = [
                (
                    self.train_data.fn_topol,
                    self.train_data.fn_coords,
                    trj,
                    self.train_data.restraints,
                    self.specs.mol_resname,
                    start,
                    stop,
                    step,
                    kwargs
                )
                for trj in self.train_data.trajectories
            ]

            # Determine whether to parallelize the analysis
            n_samples = self.train_data.n_samples
            workers = mp.cpu_count() if workers == -1 else workers
            if workers > 1:
                with mp.Pool(workers, maxtasksperchild=1) as pool:
                    iterator = pool.imap(analyze_trajectories_wrapper, args_list)
                    features = list(
                        print_progress(
                            iterator,
                            total=n_samples,
                            logger=self.logger,
                            stride=progress_stride
                        )
                    )
                    pool.close()
                    pool.join()
            else:
                iterator = print_progress(
                    args_list,
                    total=n_samples,
                    logger=self.logger,
                    stride=progress_stride
                )
                features = [analyze_trajectories_wrapper(args) for args in iterator]

            settings = get_all_settings(kwargs)
            self.train_data.load_features(features, settings)

        t1 = time.time()
        self.logger.info(
            "> loading training trajectories: "
            f"{self.train_data.n_samples}/{self.train_data.n_samples} (100%) | "
            f"Done. ({format_time(t1 - t0)})"
        )

        # Save the analyzed samples
        if fn_out:
            self.logger.info('> saving samples into file: in progress', overwrite=True)

            self.train_data.write_features(fn_out)
            t2 = time.time()
            self.logger.info(
                f'> Saving samples into file: {self.train_data.n_samples} '
                f'| Done. ({format_time(t2 - t1)})'
            )
        self.logger.info('')

    def setup_lgp(
        self,
        QoI: list[str] = None,
        n_hyper: int = 200,
        fn_hyper: dict[str | Path] = None,
        means: dict = {'rdf': 'sigmoid'},
        committee: int = 1,
        test_fraction: float = 0.2,
        device: str = 'cuda:0',
        **kwargs
    ) -> None:

        """
        Set up Local Gaussian Process models for the training data.

        Parameters
        ----------
        QoI : list[str], optional
            List of quantities of interest (QoI) to optimize. If None, uses all
            available QoI from the training data.
        n_hyper : int, optional
            Number of hyperparameter samples for optimization. Default is 200.
        fn_hyper : dict[str | Path], optional
            Dictionary mapping QoI to file paths for hyperparameter optimization.
            If None, hyperparameters are optimized and results are saved.
        max_iter : int, optional
            Maximum number of iterations for the optimization. Default is 20000.
        means : dict, optional
            Dictionary specifying the mean values for each QoI. Default is
            {'rdf': 'sigmoid'}, which uses a sigmoid mean for the RDF.
            The rest of the QoI will use a mean of 0.0.
        comittee : int, optional
            Number of LGP models in the committee for each QoI. Default is 1.
        device : str, optional
            Device to use for computations ('cuda:0', 'cpu' or 'mps').
            Default is 'cpu' (faster for single-input batch during
            gradient-based optimizartion).
        kwargs : dict, optional
            Additional keyword arguments for the `optimize_lgp` funciton.
        """

        self.QoI = QoI or self.train_data.y_slices.keys()
        means_local = means.copy()
        if means.get('rdf') == 'sigmoid':
            means_local['rdf'] = self.train_data.rdf_sigmoid_mean

        # lgp_committees = []
        lgp_committees = {}
        for q in self.QoI:
            self.logger.info(f'> Optimizing LGP hyperparameters: {q}')
            sl = self.train_data.y_slices.get(q, slice(None))
            committee_qoi = lgp_hyperopt(
                X=self.train_data.X,
                y=self.train_data.y[:, sl],
                y_mean=means_local.get(q, 0.0),
                fn_hyperparams=fn_hyper.get(q, None),
                test_fraction=test_fraction,
                n_hyper=n_hyper,
                committee=committee,
                observations=self.train_data.observations.get(q, None),
                device=device,
                logger=self.logger,
                opt_kwargs=kwargs,
            )

            # Store the model
            # lgp_committees.append(committee_qoi)
            lgp_committees[q] = committee_qoi
            self.logger.info('')

        # self.surrogate = CommitteeWrapper(lgp_committees)
        self.surrogate = lgp_committees

    def run(
        self,
        n_max: int = 50000,
        n_walkers: int = None,
        priors_disttype: str = 'normal',
        fn_backend: str = './backend.h5',
        fn_priors: str = './priors.yaml',
        fn_tau: str = './tau.npy',
        fn_specs: str = './specs.yaml',
        restart: bool = True,
        device: str = 'cuda:0',
        **kwargs
    ) -> OptimizationResults:

        """
        Run the Bayesian inference to optimize the force field parameters.

        Parameters
        ----------
        n_max : int, optional
            Maximum number of MCMC samples to generate. Default is 50000.
        n_walkers : int, optional
            Number of walkers in the MCMC sampler. If None, defaults to 5 * n_params.
            Default is None.
        priors_disttype : str, optional
            Distribution type for the priors. Options are 'normal' or 'uniform'.
            Default is 'normal'.
        fn_backend : str, optional
            Path to the backend file for the MCMC sampler. Default is './backend.h5'.
        fn_priors : str, optional
            Path to save the prior distributions. Default is './priors.yaml'.
        fn_tau : str, optional
            Path to save the autocorrelation time. Default is './tau.npy'.
        fn_specs : str, optional
            Path to save the specifications of the model. Default is './specs.yaml'.
        restart : bool, optional
            If True, restarts the MCMC sampler from the backend file.
            If False or the file does not exist,
            initializes a new sampler. Default is True.
        """

        # Construct the MCMC sampler
        if not self.surrogate:
            raise ValueError("Surrogate model is not set up. "
                             "Call `setup_lgp` before running the MCMC sampler.")

        # Construct the LGP committee
        p0, priors, sampler = initialize_mcmc_sampler(
            self.surrogate, self.specs, self.QoI,
            self.train_data.y_true, self.train_data.y_slices,
            n_walkers, priors_disttype, fn_backend, restart, device
        )

        # Run the MCMC sampler
        self.logger.info(f'> Optimizing parameters: {self.specs.mol_resname}')
        print_progress_mcmc(sampler, p0, n_max, logger=self.logger, **kwargs)
        tau = sampler.get_autocorr_time(tol=0)
        results = OptimizationResults(sampler, priors, tau, self.specs.data)

        # Save the results
        if fn_priors:
            results.save_priors(fn_priors)
        if fn_tau:
            results.save_tau(fn_tau)
        if fn_specs:
            self.specs.save(fn_specs)

        return results

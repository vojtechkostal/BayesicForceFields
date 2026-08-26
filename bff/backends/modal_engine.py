"""
Modal Cloud Serverless Backend for BayesicForceFields.
======================================================
Provides scalable serverless molecular dynamics execution across 100+ concurrent
worker containers using GROMACS with OpenMP AVX2 acceleration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class ModalGromacsBackend:
    """Serverless Modal Cloud runner for high-throughput GROMACS MD campaigns."""

    def __init__(self, app_name: str = "bff-gromacs-campaign", max_containers: int = 100):
        self.app_name = app_name
        self.max_containers = max_containers

    @staticmethod
    def is_available() -> bool:
        try:
            import modal  # noqa: F401
            return True
        except ImportError:
            return False

    def run_campaign(
        self,
        top_text: str,
        gro_text: str,
        mdp_text: str,
        parameter_samples: np.ndarray,
        sample_ids: List[str],
        threads_per_worker: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Dispatches parameter batches to Modal Cloud serverless workers.
        """
        if not self.is_available():
            raise RuntimeError("modal package is required. Install via: pip install modal")

        from new_optimizations.modal_bff_campaign import run_gromacs_simulation, app

        items = []
        for sid, params in zip(sample_ids, parameter_samples):
            items.append({
                "sample_id": sid,
                "top_text": top_text,
                "gro_text": gro_text,
                "mdp_text": mdp_text,
                "n_steps": 100000,
                "dt_ps": 0.002,
            })

        with app.run():
            results = list(run_gromacs_simulation.map(items))

        return results

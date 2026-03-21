import subprocess
from pathlib import Path


class Slurm:
    """
    A class to generate and save SLURM job submission scripts.

    Parameters
    ----------
    job_name : str
        Name of the SLURM job.
    nodes : int
        Number of nodes to allocate.
    ntasks_per_node : int
        Number of tasks per node.
    cpus_per_task : int
        Number of CPUs per task.
    time : str
        Maximum wall time in the format 'HH:MM:SS'.
    partition : str
        Partition or queue name.
    mem : str
        Memory allocation (e.g., '4G', '16G').
    output : str, optional
        Path to the output file, default is './slurm-files/slurm-%x.%j.out'.

    Attributes
    ----------
    commands : list
        List of shell commands to execute in the SLURM script.
    """

    def __init__(self, **sbatch):
        """
        Initialize the class with SBATCH parameters.

        Parameters
        ----------
        **sbatch : dict
            Arbitrary keyword arguments representing SBATCH parameters.
            Each key-value pair will be formatted as '#SBATCH --key=value'.
        """

        self.sbatch = [
            f"#SBATCH --{key.replace('_', '-')}={value}"
            for key, value in sbatch.items()
        ]
        self.commands = []
        self.fname = None

    def add_command(self, command: str) -> None:
        """
        Add a shell command to the script.

        Parameters
        ----------
        command : str
            The shell command to add.
        """
        self.commands.append(command)

    def generate(self) -> str:
        """
        Generate the SLURM script content as a string.

        Returns
        -------
        str
            The complete SLURM script.
        """
        script = "#!/bin/bash\nset -euo pipefail\n"
        if self.sbatch:
            script += "\n".join(self.sbatch) + "\n"
        if self.commands:
            script += "\n" + "\n".join(self.commands) + "\n"
        return script

    def save(self, filename: str | Path) -> None:
        """
        Save the SLURM script to a file.

        Parameters
        ----------
        filename : str
            Path to save the SLURM script.
        """
        self.fname = Path(filename)
        with open(self.fname, "w") as file:
            file.write(self.generate())

    def submit(self, filename: str | Path) -> int:
        """
        Submit the SLURM script to the queue.
        """
        self.save(filename)
        out = subprocess.run(
            ["sbatch", str(self.fname)],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode != 0:
            message = out.stderr.strip() or out.stdout.strip() or "unknown sbatch error"
            raise RuntimeError(f"SLURM submission failed: {message}")

        tokens = out.stdout.strip().split()
        if not tokens:
            raise RuntimeError(
                "SLURM submission failed: could not parse job ID from sbatch output."
            )
        try:
            return int(tokens[-1])
        except ValueError as exc:
            raise RuntimeError(
                "SLURM submission failed: could not parse job ID from "
                f"sbatch output {out.stdout!r}."
            ) from exc

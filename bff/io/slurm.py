import subprocess


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
            f'#SBATCH --{key}={value}'
            for key, value in sbatch.items()
        ]
        self.commands = []
        self.fname = None

    def add_command(self, command):
        """
        Add a shell command to the script.

        Parameters
        ----------
        command : str
            The shell command to add.
        """
        self.commands.append(command)

    def generate(self):
        """
        Generate the SLURM script content as a string.

        Returns
        -------
        str
            The complete SLURM script.
        """
        script = "#!/bin/bash" + '\n'
        script += '\n'.join(self.sbatch) + "\n"
        script += "\n".join(self.commands)
        return script

    def save(self, filename):
        """
        Save the SLURM script to a file.

        Parameters
        ----------
        filename : str
            Path to save the SLURM script.
        """
        self.fname = filename
        with open(filename, 'w') as file:
            file.write(self.generate())

    def submit(self, filename):
        """
        Submit the SLURM script to the queue.
        """
        if not self.fname:
            self.save(filename)
        out = subprocess.run(
            ['sbatch', self.fname], capture_output=True, text=True
        )
        job_id = out.stdout.split()[-1]

        return job_id

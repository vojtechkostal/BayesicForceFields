from collections import OrderedDict


class MDP:
    """
    Class to read, modify, and write GROMACS .mdp files.

    The `.mdp` file format consists of key-value pairs for simulation
    settings, interspersed with comments (`;`) and blank lines.
    This class preserves the order of the file content,
    allowing for editing and writing the file back while maintaining
    the original structure.

    Attributes
    ----------
    content : OrderedDict
        Stores the .mdp file content with comments and
        blank lines labeled for preservation.

    Methods
    -------
    write(fn_out):
        Writes the current content to a new .mdp file, maintaining formatting.
    """

    def __init__(self, fn_mdp: str) -> None:
        """
        Initialize the MDP object and load the content from a file.

        Parameters
        ----------
        fn_mdp : str
            Path to the .mdp file to read.
        """
        self.content = self._read(fn_mdp)

    def _read(self, fn: str) -> OrderedDict:
        """
        Read an .mdp file and parse its content.

        This method separates comments, blank lines, and key-value pairs.
        Comments and blank lines are assigned unique keys for later reference.

        Parameters
        ----------
        fn : str
            Path to the .mdp file to read.

        Returns
        -------
        OrderedDict
            An ordered dictionary representing the .mdp file content.
        """
        content = OrderedDict()
        i_comment = 0
        i_blank = 0
        with open(fn, 'r') as f:
            for line in f:
                if line.startswith(';'):  # Comments
                    content[f'C{i_comment:03d}'] = line
                    i_comment += 1
                elif line.strip() == '':  # Blank lines
                    content[f'B{i_blank:03d}'] = line
                    i_blank += 1
                else:  # Key-value pairs
                    key, value = line.split('=', 1)
                    content[key.strip()] = value.strip()
        return content

    def write(self, fn_out: str) -> None:
        """
        Write the current content to an .mdp file.

        This method preserves comments, blank lines, and formatting
        from the original file.

        Parameters
        ----------
        fn_out : str
            Path to the output .mdp file.
        """
        with open(fn_out, 'w') as f:
            for key, value in self.content.items():
                if key.startswith('C') or key.startswith('B'):
                    f.write(value)
                else:  # Key-value pairs
                    f.write(f'{key:<25} = {value}\n')


def get_n_frames_target(fn_mdp):
    """Extracts the expected number of frames in the resulting trajectory."""
    mdp_data = MDP(fn_mdp).content
    n_steps = int(mdp_data.get('nsteps'))
    if n_steps:
        stride = int(mdp_data['nstxout-compressed'])
        return int(n_steps / stride), stride
    else:
        return None, None


def get_restraints(fn_mdp):
    """Reads the restraints from the MDP file."""

    mdp = MDP(str(fn_mdp))
    mdp_data = mdp.content

    # Check if the pull code is present
    if "pull-ncoords" not in mdp_data or "pull-ngroups" not in mdp_data:
        restraints = []

    else:
        restraints = [
            {
                "atoms": " ".join(
                    mdp_data[f"pull-group{group_idx}-name"]
                    for group_idx in mdp_data[
                        f"pull-coord{coord_idx}-groups"
                    ].split()
                ),
                "x0": float(mdp_data[f"pull-coord{coord_idx}-init"]),
                "k": float(mdp_data[f"pull-coord{coord_idx}-k"]),
            }
            for coord_idx in range(1, int(mdp_data["pull-ncoords"]) + 1)
        ]

    return restraints

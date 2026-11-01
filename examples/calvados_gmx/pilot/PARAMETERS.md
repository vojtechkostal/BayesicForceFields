# CALVADOS2 pilot parameters (Hst5 @ M1_24 optimized lambdas)

Sources: upstream `simulate.py`, `analyse.py`, `residues.csv` 
(pinned commit, checksum-verified by fetch_data.py).
Review every ADAPT-marked number.

- temperature: 293.0 K (Hst5 in analyse.py)
- pH: 7.5 (HIS charge = 1/(1+10^(pH-6)) = 0.0307 e)
- ionic strength: 0.15 M
- water dielectric series: epsw = 80.31
- Bjerrum length lB = 0.7101 nm, RT = 2.4361 kJ/mol
- Debye kappa = 1.2697 1/nm (Debye length 0.788 nm)
- eps_HI = 0.8368000000000001 kJ/mol, AH cutoff = 2.4 nm (energy-shifted)
- Yukawa cutoff = 4.0 nm (energy-shifted)
- bonds: k = 8033.0 kJ/mol/nm^2, r0 = 0.38 nm
- dt = 0.01 ps, Langevin gamma = MW/100 (sd tau-t 1.0 ADAPT)
- exclusions: bonded pairs only (nrexcl 2); no angles

| type | three | sigma (nm) | lambda (M1 opt) | charge (e) | MW (Da) |
| --- | --- | --- | --- | --- | --- |
| A | ALA | 0.5040 | 0.2949 | +0.0000 | 71.07 |
| E | GLU | 0.5920 | 0.0030 | -1.0000 | 129.11 |
| F | PHE | 0.6360 | 0.8798 | +0.0000 | 147.18 |
| G | GLY | 0.4500 | 0.6830 | +0.0000 | 57.05 |
| H | HIS | 0.6080 | 0.4468 | +0.0307 | 137.14 |
| K | LYS | 0.6360 | 0.1371 | +1.0000 | 128.17 |
| R | ARG | 0.6560 | 0.6846 | +1.0000 | 156.19 |
| S | SER | 0.5180 | 0.4702 | +0.0000 | 87.08 |
| X | ASP | 0.5580 | 0.0833 | +0.0000 | 117.09 (terminal bead, charge and mass corrected) |
| Y | TYR | 0.6460 | 0.9708 | +0.0000 | 163.18 |
| Z | TYR | 0.6460 | 0.9708 | -1.0000 | 179.18 (terminal bead, charge and mass corrected) |

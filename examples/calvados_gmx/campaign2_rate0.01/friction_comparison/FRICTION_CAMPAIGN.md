# Campaign2 at two collision rates (same 500 points)

| | 1.16 /ps (campaign2) | 0.01 /ps (upstream) |
|---|---|---|
| tau_int of Rg, median (ps) | 172 | 17 |
| SEM of one 50 ns mean, median (nm) | 0.0189 | 0.0059 |
| SD of the 500 means (nm) | 0.0262 | 0.0185 |
| noise share of that variance | 55% | 11% |
| best R2 any surrogate can reach | 0.45 | 0.89 |

Equilibrium: mean paired difference -0.0005 +- 0.0009 nm; chi2 609 on 500 (p = 0.00); |z| > 2 for 5.6% of points (expected 4.6%); dependence on the stickiness groups F = 0.48 (p = 0.79). Counting the SEMs' own uncertainty (10 blocks each, median Welch dof 11): chi2 505 (p = 0.43); |z| > 2 expected for 7.1% of points.

Cost at equal noise: 5.2 ns at 0.01 /ps match campaign2's 50 ns (9.6x). Check: the first 6 ns of each new run have RMS SEM 0.0179 nm against 0.0194.

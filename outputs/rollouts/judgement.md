| run | n | succ | appr | grasp | transp | place | |e_p|@close cm | |e_R|@close deg | cos_appr | idle |
|---|---|---|---|---|---|---|---|---|---|---|
| base_s101 | 100 | 0.43 | 0.34 | 0.05 | 0.18 | 0.00 | 11.21 | 23.6 | 0.315 | 0.10 |
| base_s111 | 100 | 0.53 | 0.23 | 0.08 | 0.12 | 0.04 | 9.42 | 20.6 | 0.293 | 0.10 |
| ge_s101 | 100 | 0.54 | 0.21 | 0.05 | 0.18 | 0.02 | 6.19 | 12.4 | 0.276 | 0.13 |
| ge_s111 | 100 | 0.56 | 0.27 | 0.05 | 0.10 | 0.02 | 8.48 | 16.2 | 0.288 | 0.14 |

success rate by initial ||e_R(0)|| quartile (deg):
| run | 85-89 | 89-95 | 95-104 | 104-117 |
|---|---|---|---|---|
| base_s101 | 0.36 (n=25) | 0.58 (n=24) | 0.48 (n=25) | 0.31 (n=26) |
| base_s111 | 0.52 (n=25) | 0.56 (n=25) | 0.54 (n=24) | 0.50 (n=26) |
| ge_s101 | 0.60 (n=25) | 0.42 (n=24) | 0.64 (n=25) | 0.50 (n=26) |
| ge_s111 | 0.60 (n=25) | 0.60 (n=25) | 0.60 (n=25) | 0.44 (n=25) |

baseline success 0.480 ± 0.050 | grasp_err success 0.550 ± 0.010

pre-registered criteria:
  [FAIL] 1. success +10%p and both exp runs > both baseline runs
  [PASS] 2. grasp-failure ratio down and |e_R|@close down
  [PASS] 3. gain larger for ||e_R(0)|| >= 95 deg (+0.089) than below (+0.051)
  [FAIL] 4. approach-failure ratio down and cos_approach up (only if approach was the baseline cause)

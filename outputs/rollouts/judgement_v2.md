## frame: grasp_err_ee vs grasp_err (base)
| run | n | succ | appr | grasp | transp | place | |e_p|@close cm | |e_R|@close deg | cos_appr | idle | rel_out | xy@rel cm (med) | v@rel cm/s (med) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ge_s101 | 100 | 0.54 | 0.21 | 0.05 | 0.18 | 0.02 | 6.19 | 12.4 | 0.276 | 0.13 | 0.16 | 9.7 | 4.5 |
| ge_s111 | 100 | 0.56 | 0.27 | 0.05 | 0.10 | 0.02 | 8.48 | 16.2 | 0.288 | 0.14 | 0.09 | 9.9 | 4.9 |
| geee_s101 | 100 | 0.54 | 0.28 | 0.04 | 0.13 | 0.01 | 5.54 | 11.9 | 0.323 | 0.12 | 0.13 | 12.1 | 5.7 |
| geee_s111 | 100 | 0.56 | 0.22 | 0.07 | 0.11 | 0.04 | 7.67 | 15.2 | 0.281 | 0.12 | 0.10 | 9.0 | 5.5 |

success rate by initial ||e_R(0)|| quartile (deg):
| run | 85-89 | 89-95 | 95-104 | 104-117 |
|---|---|---|---|---|
| ge_s101 | 0.60 (n=25) | 0.42 (n=24) | 0.64 (n=25) | 0.50 (n=26) |
| ge_s111 | 0.60 (n=25) | 0.60 (n=25) | 0.60 (n=25) | 0.44 (n=25) |
| geee_s101 | 0.44 (n=25) | 0.52 (n=25) | 0.62 (n=24) | 0.58 (n=26) |
| geee_s111 | 0.60 (n=25) | 0.52 (n=25) | 0.68 (n=25) | 0.44 (n=25) |

baseline success 0.550 ± 0.010 | grasp_err success 0.550 ± 0.010

pre-registered criteria (frame):
  [FAIL] F1. success +5%p and both exp runs > both baseline runs
  [FAIL] F2. approach-failure ratio down
  [PASS] F3. |e_R|@close down

## place: grasp_err_ee + place_err_ee vs grasp_err_ee
| run | n | succ | appr | grasp | transp | place | |e_p|@close cm | |e_R|@close deg | cos_appr | idle | rel_out | xy@rel cm (med) | v@rel cm/s (med) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geee_s101 | 100 | 0.54 | 0.28 | 0.04 | 0.13 | 0.01 | 5.54 | 11.9 | 0.323 | 0.12 | 0.13 | 12.1 | 5.7 |
| geee_s111 | 100 | 0.56 | 0.22 | 0.07 | 0.11 | 0.04 | 7.67 | 15.2 | 0.281 | 0.12 | 0.10 | 9.0 | 5.5 |
| gepl_s101 | 100 | 0.50 | 0.29 | 0.02 | 0.16 | 0.03 | 6.47 | 15.3 | 0.297 | 0.12 | 0.18 | 12.4 | 6.6 |
| gepl_s111 | 100 | 0.59 | 0.20 | 0.03 | 0.17 | 0.01 | 9.56 | 17.4 | 0.263 | 0.11 | 0.19 | 10.6 | 5.0 |

success rate by initial ||e_R(0)|| quartile (deg):
| run | 85-89 | 89-95 | 95-104 | 104-117 |
|---|---|---|---|---|
| geee_s101 | 0.44 (n=25) | 0.52 (n=25) | 0.62 (n=24) | 0.58 (n=26) |
| geee_s111 | 0.60 (n=25) | 0.52 (n=25) | 0.68 (n=25) | 0.44 (n=25) |
| gepl_s101 | 0.68 (n=25) | 0.52 (n=25) | 0.48 (n=25) | 0.32 (n=25) |
| gepl_s111 | 0.67 (n=24) | 0.56 (n=25) | 0.64 (n=25) | 0.50 (n=26) |

baseline success 0.550 ± 0.010 | grasp_err success 0.545 ± 0.045

pre-registered criteria (place):
  [FAIL] P1. success +5%p and both exp runs > both baseline runs
  [FAIL] P2. released-outside-box ratio down
  [FAIL] P3. median truck-box xy at release down
  [FAIL] P4. median EE speed at release down

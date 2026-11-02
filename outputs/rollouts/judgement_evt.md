## primary: bc_event (ready) vs gepl
| run | n | success | lifted | miss_close | multi_switch | rel_in_box | release_out |
|---|---|---|---|---|---|---|---|
| gepl_s101 | 100 | 0.500 | 0.720 | 0.820 | 0.300 | 0.459 | 0.180 |
| gepl_s111 | 100 | 0.590 | 0.820 | 0.810 | 0.360 | 0.493 | 0.190 |
| evr_s101 | 100 | 0.530 | 0.710 | 0.420 | 0.070 | 0.603 | 0.120 |
| evr_s111 | 100 | 0.570 | 0.720 | 0.260 | 0.070 | 0.646 | 0.100 |

baseline success 0.545 | exp success 0.550
  [FAIL] B1. success +5%p (+0.005) and both exp runs > both baseline runs
  [PASS] B2. non-lifting closes per episode down
  [PASS] B2'. episodes with > 2 gripper switches down
  [FAIL] B3. episodes that lift the truck: ratio not lower

## secondary: ready_inbox vs ready (release)
| run | n | success | lifted | miss_close | multi_switch | rel_in_box | release_out |
|---|---|---|---|---|---|---|---|
| evr_s101 | 100 | 0.530 | 0.710 | 0.420 | 0.070 | 0.603 | 0.120 |
| evr_s111 | 100 | 0.570 | 0.720 | 0.260 | 0.070 | 0.646 | 0.100 |
| evi_s101 | 100 | 0.450 | 0.730 | 0.070 | 0.030 | 0.905 | 0.050 |
| evi_s111 | 100 | 0.460 | 0.750 | 0.100 | 0.010 | 0.909 | 0.060 |

baseline success 0.550 | exp success 0.455
  [PASS] R1. first release with truck-box xy < 10 cm: ratio up
  [PASS] R2. released-outside-box ratio down
  [FAIL] R3. success not lower

# 닫기 영역 규칙 e_xy < 4.0 cm 그리고 e_z < 3.0 cm, 3 step 연속 (자동 생성: tools_c/check_region_rule.py, 계획: plan_R45.md)

| 지표 | evr (evr_s101, evr_s111) | 참고: gepl (gepl_s101, gepl_s111) |
|---|---|---|
| 버틴 에피소드 중 닫음 | 13 | 2 |
| 버틴 에피소드 | 13 | 2 |
| 트럭을 든 에피소드 | 133 | 119 |
| 규칙이 처음 닫는 거리 중앙값 (cm) | 2.272 | 2.427 |
| 기록보다 먼저 닫는 비율 | 0.752 | 0.975 |
| 들어 올림 기대값: 규칙 | 0.922 | 0.908 |
| 들어 올림 기대값: 기록된 닫힘 | 0.937 | 0.931 |
| 위험 구역(e_z≥2, |e_p|≥3)에서 닫는 비율 | 0.060 | 0.109 |
| 들어 올린 비율 < 0.8인 칸에서 닫는 비율 | 0.068 | 0.118 |
| 기록된 닫힘까지 한 번도 발동 안 한 비율 | 0.248 | 0.025 |
| 기록된 닫힘 순간 영역 밖이라 계속 기다릴 비율 | 0.015 | 0.025 |
| 못 든 에피소드 중 규칙이라면 닫았을 건수 | 15 | 8 |

- 들어 올림 기대값: 닫은 순간의 (|e_p|, e_z) 칸(--lateral이면 (e_xy, e_z) 칸)에서 실제로 들어 올린 비율(gepl+evr+evi 닫힘)의 평균. 규칙과 기록된 닫힘에 같은 잣대를 쓴다.

- [PASS] R1. stuck episodes: closes in >= 12 of 13
- [PASS] R2. expected lift (rule) not more than 2%p below the recorded closes
- [PASS] R3'. first closes in low-lift cells (< 0.8) <= 10%
- [PASS] R4'. lifted episodes where the rule would still be waiting at the recorded close <= 10%

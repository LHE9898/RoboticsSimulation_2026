# 닫기 영역 규칙 |e_p| < 4.5 cm, 3 step 연속 (자동 생성: tools_c/check_region_rule.py, 계획: plan_R45.md)

| 지표 | evr (evr_s101, evr_s111) | 참고: gepl (gepl_s101, gepl_s111) |
|---|---|---|
| 버틴 에피소드 중 닫음 | 13 | 2 |
| 버틴 에피소드 | 13 | 2 |
| 트럭을 든 에피소드 | 133 | 119 |
| 규칙이 처음 닫는 거리 중앙값 (cm) | 3.663 | 3.616 |
| 기록보다 먼저 닫는 비율 | 0.857 | 0.992 |
| 들어 올림 기대값: 규칙 | 0.848 | 0.850 |
| 들어 올림 기대값: 기록된 닫힘 | 0.925 | 0.939 |
| 위험 구역(e_z≥2, |e_p|≥3)에서 닫는 비율 | 0.391 | 0.479 |
| 못 든 에피소드 중 규칙이라면 닫았을 건수 | 17 | 8 |

- 들어 올림 기대값: 닫은 순간의 (|e_p|, e_z) 칸에서 실제로 들어 올린 비율(gepl+evr+evi 닫힘)의 평균. 규칙과 기록된 닫힘에 같은 잣대를 쓴다.

- [PASS] R1. stuck episodes: closes in >= 12 of 13
- [FAIL] R2. expected lift (rule) not more than 2%p below the recorded closes
- [FAIL] R3. first closes with e_z >= 2 cm and |e_p| >= 3 cm <= 15%

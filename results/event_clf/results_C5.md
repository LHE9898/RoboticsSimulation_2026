# 잡기 정답 5 cm 검증 결과 (자동 생성: tools_c/check_close5.py, 계획: plan_C5.md)

| label | model | tau | train_on_time | train_early | stuck_fired | stuck_n | lifted_n | fire_dist_median_cm | beyond5 | beyond10 | exp_lift | no_fire_before_close |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ready | CPU seed 0 | 0.990 | 0.719 | 0.120 | 2 | 13 | 138 | 2.763 | 0.058 | 0.000 | 0.891 | 0.609 |
| ready | CPU seed 1 | 0.970 | 0.734 | 0.141 | 4 | 13 | 138 | 2.596 | 0.072 | 0.000 | 0.883 | 0.630 |
| abs5 | CPU seed 0 | 0.980 | 0.962 | 0.015 | 11 | 13 | 138 | 3.436 | 0.094 | 0.000 | 0.837 | 0.283 |
| abs5 | CPU seed 1 | 0.980 | 0.940 | 0.038 | 12 | 13 | 138 | 3.409 | 0.109 | 0.000 | 0.832 | 0.232 |
| ready | evr GPU seed101 | 0.980 | nan | nan | 1 | 13 | 138 | 2.661 | 0.065 | 0.000 | 0.886 | 0.754 |
| ready | evr GPU seed111 | 0.990 | nan | nan | 1 | 13 | 138 | 2.567 | 0.058 | 0.000 | 0.895 | 0.775 |

- stuck_fired: 버틴 13건 중 목표 3 cm 안에서 닫기 판단이 발동한 건수
- fire_dist / beyond5 / beyond10 / exp_lift: evr이 트럭을 든 에피소드에서 처음 닫기 판단이 나온 순간의 잡기 목표 거리와, 그 거리로 환산한 들어 올림 기대 비율
- no_fire_before_close: 기록된 닫힘 이전에 한 번도 발동하지 않은 비율 (이 경우 기록된 닫힘 거리를 쓴다; 실제로는 늦게 닫거나 버틸 수 있음)

- [PASS] V. control (ready, CPU) closes in <= 5 of the stuck episodes
- [PASS] C1. abs5 closes near the target in >= 10 of the stuck episodes (every seed)
- [FAIL] C2. abs5 first close beyond 5 cm <= 5% and expected lift >= 0.93
- [PASS] C3. abs5 first close beyond 10 cm <= 2%

## 해석 (사후 메모)

- 사전 기준대로면 C2 실패이므로 계획에 따라 GPU 재학습은 하지 않는다.
- 다만 C2의 기준값 0.93은 잘못 정했다. 이 환산은 실제로 트럭을 든 에피소드에도 거리별 평균 비율을 적용하므로, 대조(ready)와 실제 evr 분류기도 0.88–0.90에 그친다. 기준을 넘을 수 있는 모델이 없었다.
- 대조와 나란히 보면 abs5는 버티기 13건 중 11–12건을 해소한다(대조 2–4건, 실제 evr 1건). 대신 처음 닫는 거리 중앙값이 2.6–2.8 cm에서 3.4 cm로 당겨지고, 5 cm 밖에서 닫는 비율이 6–7%에서 9–11%로 늘어 들어 올림 기대 비율이 약 5%p 낮다(0.83 대 0.88–0.89).
- 성공률 환산(들어 올린 뒤 성공 0.77): 버티기 해소로 +8–9건, 일찍 닫기로 −5건, 순 +3건 정도(+1.5%p)로 본다.

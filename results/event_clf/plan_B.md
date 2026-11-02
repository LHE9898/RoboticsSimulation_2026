# B단계 계획: 그리퍼 사건 분류를 붙인 정책의 시뮬레이션 평가

학습 전에 작성했다. A단계(오프라인)에서 사건 분류기가 먼 곳 오작동을 줄이고(0.11 → 0.01회/에피소드) 잡을 닫힘과 헛닫힘을 가려냈으므로(ROC-AUC 0.41–0.48 → 0.75–0.76), 이를 정책에 붙여 실제 성공률이 오르는지 확인한다.

## 모델

| 이름 | 관측 | 팔 6D | 그리퍼 | 사건 정답 |
|---|---|---|---|---|
| `gepl` (대조, 기존 결과 재사용) | 32D | GMM | GMM의 7번째 차원 (매 step 무작위 mode) | - |
| `evr` = `bc_event_ready` | 같은 32D | 같은 GMM | 사건 분류기 + 상태 기계 | `grip_event_ready` |
| `evi` = `bc_event_ready_inbox` | 같은 32D | 같은 GMM | 사건 분류기 + 상태 기계 | `grip_event_ready_inbox` |

- 알고리즘: `tools_c/bc_event.py` (robomimic `bc_event`). 팔은 gepl과 같은 BC-RNN-GMM(LSTM 2×400, 길이 10, GMM 5 modes)이고 그리퍼만 GMM에서 뺀다.
- 사건 분류기: 입력 [관측 32D, 직전 step 대비 변화량], MLP 256×2, 출력 2개(열려 있을 때 "지금 닫기", 닫혀 있을 때 "지금 열기"), 양성 가중 BCE.
- 정답 (`tools_c/add_grip_event_to_hdf5.py`):
  - `ready`: 실제 전환 직전 5 step에 더해, EE가 이미 전환 위치에 있던 구간(닫기: 잡기 목표 거리가 전환 순간 +1 cm 이내, 열기: 박스 xy가 전환 순간 +1 cm 이내이고 높이 차 1 cm 이내)을 최대 40 step까지 양성으로 둔다. A단계에서 "사람이 키를 누르기 전 이미 잡을 자리에 있던 구간"을 일찍으로 셌던 문제를 반영했다.
  - `ready_inbox`: `ready`와 같되, 열기 양성은 트럭–박스 xy < 10 cm에서 놓은 시범에서만 둔다.
- 실행 규칙: 그리퍼 상태를 유지하다가 p > τ[상태]가 3 step 연속이면 바꾼다. 무작위 뽑기 없음.
- 기준값 τ: 학습이 끝난 뒤 **학습 데이터의 시범에서만** 고른다(`tools_c/calibrate_event_tau.py`, 후보 0.5–0.99 중 "정답 구간 안에서 처음 바꾸는 비율"이 가장 높은 값). 평가 rollout을 보기 전에 `event_tau.json`으로 고정한다.

## 공통 조건

- 데이터: `datasets/mimic_500_evt.hdf5` = `mimic_500_rel.hdf5`(gepl 학습 데이터)에 그리퍼 상태·사건 정답만 추가. 시범, 행동, 관측은 같다.
- 학습: 2000 epoch, 마지막 체크포인트, 학습 seed 101과 111 (gepl과 같음).
- 평가: seed 42, rollout 100회, horizon 800, 영상 기록.

## 사전 판정 기준

**주 비교: `evr` 대 `gepl`** (모두 충족하면 지지)
- **B1.** 성공률 평균 +5%p 이상, 그리고 `evr` 2 run 모두 > `gepl` 2 run
- **B2.** 트럭을 들지 못한 닫힘 횟수(에피소드당) 감소
- **B2′.** 그리퍼를 2번 넘게 바꾼 에피소드 비율 감소
- **B3.** 트럭을 들어 올린 에피소드 비율이 gepl 이상

**부 비교: `evi` 대 `evr`** (놓기 정답 거르기의 효과)
- **R1.** 처음 놓은 순간 트럭–박스 xy < 10 cm 비율 증가
- **R2.** 박스 밖 놓기 비율 감소
- **R3.** 성공률이 낮아지지 않음

판정은 `tools_c/judge_event.py`로 자동 계산한다(`outputs/rollouts/judgement_evt.md`).

## 해석 규칙

- B2와 B2′만 통과하고 B1이 실패하면: 그리퍼 결정은 고쳐졌지만 성공률의 병목이 다른 곳(접근 동작, 데이터의 놓기 위치 분포)에 있다는 근거로 쓴다.
- B3이 실패하면(잡는 비율 감소): 기준값이 너무 높아 닫아야 할 때 못 닫는 것인지, 사후 분석으로 확인한다.
- R1–R3이 실패하면: 놓기 위치는 정답을 거르는 것만으로는 바뀌지 않고, 시범 자체(놓는 위치)를 바꿔야 한다는 근거로 쓴다.

## 실행 순서 (GPU가 비면)

```bash
conda activate lerobot-arena && cd ~/IsaacLab_RS
bash tools_c/run_c_pipeline.sh train_evt   # 2 변형 × seed 101, 111 (GPU, Isaac Sim 실행)
bash tools_c/run_c_pipeline.sh calib_evt   # 기준값 보정 (CPU) -> <run>/event_tau.json
bash tools_c/run_c_pipeline.sh eval_evt    # 평가 + 영상 + 판정 -> outputs/rollouts/judgement_evt.md
```

이미 끝난 준비: `convert_evt`(데이터 생성), `smoke_evt`(CPU 동작 확인: 학습 → 저장 → 다시 불러오기 → rollout 방식 행동 → 기준값 보정).

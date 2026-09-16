# 실행 검증 — 2026-09-16

## 판정

**실제 데이터 기반 closed-loop 연구 프로토타입이 실행됩니다.** 전체 선택 뉴런 그래프, 독립 brain state, 실제 NeuroMechFly 몸, 먹이, 접촉 손상, 몸 복구, 선택적 가소성, 대조군과 기록/분석 경로를 검증했습니다.

**요청의 모든 과학적 성공 조건을 충족했다고 판정하지 않습니다.** 특히 aggression 전략, 검증된 neuron→muscle mapping, winner/loser effect와 일반화된 학습 성능은 미검증입니다. 짧은 실행에서 attack onset은 0이었으며, 아래의 damage는 물리적 몸/다리 접촉에서 발생했습니다.

## Phase별 결과

| 단계 | 실제 실행 및 결과 | 메모리 / 한계 |
|---|---|---|
| 데이터 및 loader | 공식 MaleCNS v1.0 Feather 3개 다운로드, SHA-256 기록. 뉴런 166,700 / 연결 25,582,938 / type 11,751 | CSR 196.45MiB. 변환 완료 RSS 0.687GiB. 원본 segment 연결 151,856,684개 중 126,273,746개는 선택 뉴런 집합 밖이므로 제외 |
| Sparse CNS | 두 brain, 각각 100 timestep. 전체 그래프 자극/전파. synthetic 및 실제 데이터 subset/full tests | smoke RSS 0.514GiB. 신경 VRAM 0. 높은 회로 발화율은 미보정 문제 |
| 독립 상태/공유 | graph/weights 공유, 각 brain의 voltage/delta/eligibility/RNG 분리 | topology 복사 없음. reset A/B/C와 state 격리 단위 테스트 통과 |
| 물리 환경 | FlyGym 2.1.0 / MuJoCo 3.9.0. 2마리, 134 joints, 96 actuators. 0.5초 보행 검증에서 실제 접촉 기록 1,502개 | 접촉 기록은 control interval당 최대 64개 표본이며 생물학적 접촉 횟수가 아님. 힘/손상 적분은 모든 접촉에서 수행 |
| 부위별 손상/복구 | 실제 접촉 손상, 앞다리 파괴 시 해당 actuator gain=0, reset으로 복구 테스트 | impulse 적분의 timestep 독립성 테스트 통과. 조직 상수는 생물학적으로 보정되지 않음 |
| 행동불능 | 매우 약한 조직을 설정한 3회 회귀 실행: 0.15초 locomotion_loss, 0.12초 locomotion_loss, 0.16초 critical_thorax | 기본 생물학적 조건이 아니라 명시적 경로 검증 설정. 다음 경기에도 약 2,700~3,000개 plastic delta가 유지/변화 |
| 반복/대조군 | quick 설정의 ON 100회 + OFF 100회 완료. 동일 matchup/seed 순서 확인 | ON 659.89초, OFF 651.65초. 동시 실행한 벽시계 수치이며 단독 성능 추정으로 사용하지 말 것 |
| 다개체 pool | 20개 brain으로 무작위 2회 실행 완료 | RSS 0.716GiB, OS peak 0.778GiB. 동시에 20개 몸을 물리 실행하는 benchmark는 아님 |
| GUI/영상 | native MuJoCo GUI 실행/종료 통과. 기본 2초 경기 영상 960×640, H.264, 25fps, 길이 2.04초 저장 및 frame 확인 | 2초 simulated time에 75.02초 wall time. peak sampled RSS 0.762GiB, OS peak 0.767GiB |
| checkpoint | 100회 checkpoint로 101번째 경기 실행. 모든 초기 plastic delta가 저장본과 정확히 일치 | RNG round-trip 단위 테스트 통과. 완전한 운영체제/버전 간 bitwise 재현성은 보장하지 않음 |
| 일반화 평가 경로 | 동결한 trained delta vs 2개의 naive 상대 pool에서 2회 smoke 완료 | 평가 경로 실행 검증이며 일반화 성공의 증거가 아님 |
| 분석 | summary Parquet, 9-panel overview, ON/OFF 비교, weight histogram 생성 및 시각 검토 | 초기 hunger가 고정되어 hunger sweep을 하지 않았음을 그래프에 명시 |

## 100회 paired run 관찰

`results/paired_verification.json`:

- 각 조건 100경기, 각 0.15초 simulated time.
- 두 조건 matchup/seed 순서 동일.
- OFF의 modified synapse 수는 모든 경기에서 0.
- ON 마지막 경기 참가 개체들의 modified synapse 수: 3,726 / 3,878.
- 총 food 섭취량: ON 1.2, OFF 1.2 (모델 energy 단위).
- ON 누적 damage: 1.0032 (모든 개체/경기 합, 모델 integrity 단위).
- ON sampled RSS 약 0.635GiB, OFF 약 0.635GiB.
- ON 약 45.46 brain steps/s, OFF 약 46.04 brain steps/s. 2개 brain 계산을 합친 단계 수이며 물리 substep 처리율과 다릅니다.
- 신경 계산 VRAM 0. `nvidia-smi` GPU 수치는 desktop/다른 앱 및 rendering을 포함합니다.

가중치 변화와 결과 궤적의 차이는 확인됐습니다. **전체 food 섭취량의 개선이나 공격 전략 학습은 입증되지 않았습니다.** 두 개체의 PCA는 최대 rank 1이며 population 행동군집을 해석할 수 없습니다.

## 기본 설정 2초 경기

`results/demo_full_2s`:

- 두 개체 이동거리 약 14.53 / 18.00mm.
- food 섭취량 0.0376 / 0.
- 각 개체 접촉 damage 약 0.000958.
- modified synapse 3,504 / 2,980.
- attack onset 0 / 0; time limit으로 종료.

영상은 물리적 상호작용과 신경 계산이 연결된 실행 증거입니다. 모델이 실제 싸움 전략을 습득했다는 영상이 아닙니다.

## 성능 분리 측정

`scripts/benchmark.py`, `results/benchmark.json` (JIT warmup 후, 2개 brain × 100단계):

| 조건 | brain steps/s | 두 brain world steps/s |
|---|---:|---:|
| 무자극 | 369.48 | 184.74 |
| 감각 뉴런 128개 자극 | 150.09 | 75.04 |

이 benchmark에는 물리/GUI가 포함되지 않습니다. renderer 경기와 같은 시간에 실행되어 CPU 경합의 영향이 있을 수 있습니다. O(active outgoing edges) 전파라서 발화량에 따라 시간이 달라집니다. RAM peak 약 0.503GiB. CUDA 가속은 구현하지 않았습니다.

## 테스트

최종 기록 `results/tests.txt`: **9 passed**.

대상: LIF propagation/refractory, STDP 시간 순서와 OFF 불변성, state 공유/독립 및 A/B/C reset, impulse 적분, hunger/단일 food 분배, 실제 전체 MaleCNS, 실제 induced subset, 실제 MuJoCo 충돌/actuator 약화/복구, checkpoint/RNG와 body reset.

## 재현성 범위와 다음 과학적 작업

모든 run은 config, seed, dataset/provenance, dependency versions, CPU/OS 및 timestamp를 저장했습니다. 초기 workspace는 git repository가 아니므로 git commit은 null입니다. 이후 run에는 Python source SHA-256도 추가했습니다. 초기 100회 run의 과학적 모델 경로는 같지만, 이후 GUI/프로파일링/평가 기능이 추가됐으므로 초기 run이 최종 코드의 모든 기능을 검증하는 것은 아닙니다. 최종 테스트, GUI, 2초 영상, resume smoke는 해당 추가 후 실행했습니다.

남은 작업은 신경 및 NT/receptor parameter calibration, 실험 근거가 있는 sensory/DN mapping, 실제 lunge/wing/grapple biomechanics, contact impulse의 감각 부호화 확장, 여러 seed·hunger 조건·held-out 상대에서 충분한 길이의 실험입니다. 현재 짧은 반복 검증만으로 winner/loser effect나 행동 전략 분화를 주장하지 않습니다.

## GUI 끊김 수정 검증

기존 GUI는 영상 캡처와 같은 callback에서 가상 시간 0.04초마다만 sync했습니다. 전체 계산이 실시간보다 수십 배 느려 화면이 실제 시간 기준으로 오래 정지했습니다. GUI 전용 실행도 불필요한 offscreen rendering을 수행했습니다.

수정 후 GUI는 별도 모델/상태를 사용하며 벽시계 기준 30Hz로 갱신합니다. 실제 계산 결과 사이의 자세를 화면에서만 보간합니다. sparse 전파 커널은 계산 중 GIL을 해제해 화면 스레드가 동작할 수 있게 했으며, 연산 순서/신경·물리 timestep은 바꾸지 않았습니다.

`scripts/profile_gui.py`의 실제 전체 그래프, 2개체, 0.3초 paired 실행 결과 (`results/gui_regression.json`):

- GUI 상태 갱신 평균 30.00회/초, 간격 중앙값 33.33ms, 95백분위 48.92ms, 최대 54.05ms.
- 시뮬레이션 snapshot 31회로 GUI 356회 갱신. 이 수치는 GUI에 전달한 상태 갱신률이며, 모니터의 실제 present FPS를 측정한 값은 아닙니다.
- GUI의 offscreen renderer 생성 없음, video 처리 시간 0.
- headless와 GUI의 모든 episode agent 결과 및 checkpoint 배열이 정확히 일치.
- GUI 실행 RSS 약 0.938GiB, OS peak 약 0.939GiB. 신경 계산 VRAM 0; 다른 앱을 포함한 총 GPU 사용량 약 1,192MiB.
- GUI 경기 구간의 신경 처리 약 4.32초, 물리 처리 약 6.57초. 따라서 실제 계산은 여전히 슬로모션이며, 이 수정이 전체 모델을 실시간으로 가속했다고 해석하지 않습니다.

추가 회귀 테스트는 quaternion 보간, episode reset 시 보간 금지, GUI 상태 격리, 물리 진행 없이도 화면 스레드가 계속 동작하는지 확인합니다.

## 학습 운동 출력층 추가 검증 (2026-09-16)

이후 기본 decoder에 별도 공학적 actor/critic 363개 파라미터를 추가했습니다. 전체 MaleCNS와 실제 20,000개 이하 CNS 가소성 연결은 유지합니다. 보상은 섭취량에서 실제 순 에너지 변화로 수정했습니다. 앞다리의 독립 관절/부착 출력을 추가했으며 공격 전략은 제공하지 않습니다. 따라서 위의 과거 고정 decoder 실험과 아래 실험은 서로 다른 모델 버전입니다. 구조와 실행법: [LEARNING.md](LEARNING.md).

### 국소 학습 알고리즘의 동작

`scripts/verify_motor_learning.py`, `results/motor_learning_benchmark.json`:

- 합성 actuator calibration, 5개 seed, 각 1,500회 학습 후 탐색/학습을 끄고 새 rollout 평가.
- 같은 두 관측 상황을 훈련과 평가에 사용하며, 미지의 상황에 대한 일반화 테스트는 아닙니다.
- 동결 출력층 MSE 0.4161, 학습 출력층 0.0117–0.0177, 관측 상황과 보상 목표의 연결을 섞은 대조군 0.2259–0.4646.
- 이 결과는 보상에 따른 출력 선택 개선을 검증합니다. **실제 MaleCNS, 먹이 탐색, 싸움 학습의 결과가 아닙니다.**

### 실제 MaleCNS / MuJoCo 통합

`results/motor_learning_on`, `motor_learning_off`: 동일 seed/상대의 0.15초 경기 3회씩.

- ON: 각 개체의 actor 가중치 330개 변화, 6회 갱신. CNS 연결 변화 3,047 / 2,888개.
- OFF: CNS 및 actor 가중치 변화 없음. 탐색은 양쪽 모두 켬.
- 총 먹이 획득량은 ON/OFF 모두 0.0360. **성능 향상을 보이지 않았습니다.**
- `motor_learning_continuous`의 연속 4회와 `motor_learning_resumed`의 3회 후 재개 결과: 마지막 경기 지표, 모든 뇌/운동 상태 배열, RNG가 정확히 일치.
- `motor_learning_frozen`: 2회의 frozen 평가 후 CNS delta 및 actor/critic 가중치가 원본 checkpoint와 정확히 일치.
- 검증 스크립트 `scripts/verify_learning_integration.py`, 결과 `results/motor_learning_integration.json`.

기본 조건의 `results/motor_learning_body_2s`: 두 몸이 걷고 접촉하며 독립 앞다리 출력을 사용한 2초 실행 및 영상 저장. 먹이 0.0448 / 0.0072. 실제 시간 약 69.92초 (신경 28.58초, 물리 39.04초, 영상 1.90초), RSS 약 0.755GiB. 순간 자세를 확인했고 비정상 수치 없이 종료했습니다. 이는 물리 실행 점검이며 학습된 전투 영상이 아닙니다.

새 출력층에서도 `scripts/profile_gui.py`의 GUI/headless 경기 결과와 **모든 checkpoint 항목**이 정확히 일치했습니다. 화면 상태 갱신 약 29.97회/초, 95백분위 간격 46.09ms, 최대 50.69ms. OS peak RSS 약 0.933GiB, neural VRAM 0. 렌더링을 포함한 전체 계산은 여전히 실시간보다 느립니다.

`pytest -q`: **19 passed**. 보상 지연의 행동 credit, OFF 불변성, A/B/C 운동 가중치 보존, CNS/출력층 독립 ablation, checkpoint/난수 복구, frozen 가중치 이전, 실제 앞다리 제어·손상·복구를 포함합니다.

새 모델의 공격 횟수/성공률은 `null`로 기록합니다. 독립 관절 동작이나 충돌을 공격으로 자동 해석하지 않습니다. 자발적 전투, 경쟁 전략 개선, 장기 훈련의 수렴, 낯선 상대에 대한 성능 향상은 아직 검증하지 않았습니다.

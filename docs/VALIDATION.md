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

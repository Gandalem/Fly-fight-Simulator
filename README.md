# FlyFight — MaleCNS embodied neural simulation

실제 MaleCNS v1.0 연결망을 공유하는 여러 독립 LIF 뇌를 NeuroMechFly / MuJoCo 몸에 연결하는 연구용 프로토타입입니다. 먹이 경쟁, 허기, 부위별 손상, 몸의 복구와 신경 가소성 보존을 실험합니다.

**실제 초파리 뇌 전체를 완벽히 재현한 것은 아닙니다.** 실제 연결 구조와 annotation을 사용하지만, 신경 파라미터·감각/운동 mapping·생리/부상 모델은 검증되지 않은 가정입니다. 학습 상태 변화가 관찰되어도 생각·성격·고통 또는 실제 winner/loser effect를 입증하지 않습니다.

## 구현된 경로

```text
MaleCNS outgoing CSR (공유, 읽기 전용)
  → 개체별 LIF voltage/spikes/trace/선택적 plastic delta
  → descending activity → 개체별 학습 운동 출력층
  → NeuroMechFly 보행 CPG + 독립 앞다리 관절/부착 제어
  → 같은 arena의 접촉·먹이·위치 변화
  → vision/contact/food/internal feature encoder → LIF
```

전체 adjacency를 dense로 만들거나 BPTT/Adam을 사용하지 않습니다. CPU event-driven 전파가 활성 뉴런의 outgoing edge만 방문합니다. 최대 20,000개 실제 연결만 plastic하며 전체 topology와 기본 가중치는 모든 개체가 공유합니다. 추가된 363개 운동 출력층 파라미터는 **공학적 adapter로 실제 MaleCNS 시냅스와 구분**합니다.

**스스로 싸우는 전략은 아직 검증되지 않았습니다.** 현재는 행동을 탐색하고 에너지 회복·부상 비용으로 운동 출력을 학습하는 단계입니다. 구조, 대조군, 저장된 뇌를 보는 방법은 [docs/LEARNING.md](docs/LEARNING.md)에 있습니다.

## 설치 — Windows

native CPython 3.12 이상을 사용하세요. 이 PC의 기본 `python`은 MSYS Python이므로, 이미 구성한 프로젝트 가상환경을 직접 실행하는 것이 안전합니다.

```powershell
cd 'D:\fly fight'
.\.venv\Scripts\python.exe -m pytest -q
# 새 환경에서는:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Linux/WSL2:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

`requirements-lock.txt`는 이 Windows 실행의 정확한 버전입니다. Linux에서는 플랫폼별 dependency 차이 때문에 pyproject 설치를 사용하고 run.json의 실제 버전을 확인하세요. Linux headless rendering은 작동하는 OpenGL/EGL 설정이 필요합니다. 신경 및 headless 물리 계산은 GPU 없이 실행합니다.

## 데이터 준비

[공식 MaleCNS 다운로드](https://male-cns.janelia.org/download/)에서 약 1.11GB의 세 파일만 사용합니다. 데이터 라이선스: CC-BY. 데이터셋/논문 및 API 조사: [docs/RESEARCH.md](docs/RESEARCH.md).

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m flyfight prepare-connectome
.\.venv\Scripts\python.exe -m flyfight inspect-connectome
.\.venv\Scripts\python.exe -m flyfight smoke --steps 100 --agents 2
```

이미 로컬에 있는 공식 파일은 `configs/default.yaml`의 `dataset.raw` 아래에 두세요. 다른 processed graph는 `--data 경로`로 지정합니다. 원본 annotation의 `bodyId`, NT의 `body`, 연결의 `body_pre/body_post/weight`를 사용하며, 예상 schema와 다르면 실패합니다. 데이터가 없을 때 synthetic 데이터로 대체하지 않습니다.

원본은 151,856,684개의 segment 연결을 포함합니다. 현재 모델은 **annotation에 superclass가 있는 166,700개 뉴런 사이의 25,582,938개 연결**을 사용합니다. CSR 배열은 약 196.45MiB입니다. 데이터 해시·제외 수·변환 설정은 manifest/provenance에 남습니다. 기본 가중치 설정을 바꾸면 반드시 `prepare-connectome`을 다시 실행하세요.

## 실행

```powershell
# 한 경기 및 영상 (fight.mp4, 화면 overlay 포함)
.\.venv\Scripts\python.exe -m flyfight fight --agents 2 --render
# MuJoCo 라이브 GUI (실제 계산 속도는 창 상단에 표시)
.\.venv\Scripts\python.exe -m flyfight fight --gui
# 경험을 보존하며 반복 훈련/관찰 (경기당 가상 10초; 실제 수 분 가능)
.\.venv\Scripts\python.exe -m flyfight train --config configs/selfplay.yaml --episodes 100 --gui --output results/selfplay_01
# 저장된 두 뇌를 추가 학습/탐색 없이 관찰
.\.venv\Scripts\python.exe -m flyfight fight --config configs/selfplay.yaml --resume results/selfplay_01/checkpoint.npz --plasticity off --exploration off --gui
# 기본 2초 경기 100회: 실제 실행 시간은 하드웨어/활동량에 따라 오래 걸립니다.
.\.venv\Scripts\python.exe -m flyfight train --episodes 100 --headless
# 짧은 파이프라인 검증용 경기 100회 (장기 학습 실험이 아님)
.\.venv\Scripts\python.exe -m flyfight train --config configs/quick.yaml --episodes 100 --output results/my_on
.\.venv\Scripts\python.exe -m flyfight train --config configs/quick.yaml --episodes 100 --plasticity off --output results/my_off
.\.venv\Scripts\python.exe scripts/verify_run.py results/my_on results/my_off
# 독립 상태 20개를 보관하고 매 경기 2마리를 무작위 선택
.\.venv\Scripts\python.exe -m flyfight train --population 20 --episodes 10000 --random-matchmaking --headless
# 여러 마리를 동시에 배치
.\.venv\Scripts\python.exe -m flyfight fight --population 4 --agents 4
```

출력 경로가 이미 차 있으면 덮어쓰지 않습니다. 각 run의 `episodes.sqlite`, `run.json`, `mapping.json`, `performance.jsonl`, `initial.npz`, `checkpoint.npz`가 저장됩니다. neural spike raster는 `smoke`에서만 제한된 길이로 저장합니다. timestep별 CSV를 무제한 생성하지 않습니다.

### GUI 끊김과 계산 속도

`fight --gui`는 별도의 표시용 MuJoCo 상태를 사용해 기본 초당 30회 갱신합니다 (`arena.gui_fps`). 계산 결과 사이의 관절/몸 위치만 화면에서 보간하며, 원래 물리 상태나 신경/학습 timestep을 변경하지 않습니다. 마우스 힘·상태 조작도 표시용 복사본에만 적용됩니다. GUI만 사용하는 경우 영상 저장용 offscreen renderer는 만들지 않습니다.

**화면 갱신 속도와 시뮬레이션 계산 속도는 다릅니다.** 전체 MaleCNS와 두 몸을 CPU로 계산하면 실제 시간보다 훨씬 느립니다. 창 상단의 `Compute speed 0.03x real time`은 실제 1초에 가상 시간이 약 0.03초 진행된다는 뜻입니다. 이 경우 움직임은 부드러운 슬로모션으로 표시됩니다. 보간에는 최대 약 한 계산 결과 간격의 표시 지연이 있으며, 다음 계산 결과가 오지 않으면 마지막 상태에서 기다립니다.

`viewer_metrics.json`에 화면 상태 갱신 간격을, episode summary의 `timings`에 신경·물리·영상 계산 시간을 저장합니다. 회귀 검증은 `python scripts/profile_gui.py`로 실행합니다. 같은 seed의 GUI/headless 결과와 checkpoint가 정확히 같은지 비교합니다.

## 계산 속도와 진행 표시

기본 설정은 컴파일한 LIF/STDP 커널, 독립 뇌 2개 병렬 계산, FlyGym 관절/감각 mapping 캐시와 원래 spline 계수의 일괄 평가를 사용합니다. **전체 MaleCNS, 물리·신경 timestep, 학습 규칙은 유지합니다.** 참고 경로와 경기 결과/체크포인트가 정확히 같은지 비교하는 스크립트는 `scripts/benchmark_speed.py`입니다.

이 PC의 실제 10초 평가 한 경기에서 약 **311초 → 155초**로 줄었습니다. 이 비교의 모든 과학적 episode 기록은 일치했습니다. 동시 실행 프로그램에 따라 속도는 달라지며 아직 실시간은 아닙니다. 최초 커널 컴파일은 추가 시간이 들 수 있습니다. 이미 실행 중인 프로세스는 기존 코드를 사용하며 새 실행부터 최적화가 적용됩니다.

터미널에는 기본 10초마다 `progress_percent`(진행률), `compute_speed`(가상 시간/실제 시간), `eta_seconds`(시간 제한까지 예상 남은 실제 초)가 표시됩니다. 예상 시간은 현재 속도 기준으로 바뀌며 행동불능이 발생하면 일찍 끝날 수 있습니다. 매 경기 끝에는 `episode_complete`와 실제 소요 시간을 출력합니다.

설정: `runtime.fast_neural: true`, `runtime.brain_workers: 2`, `arena.controller_backend: cached`, `runtime.progress_seconds: 10.0`. 참고 구현을 비교하려면 각각 `false`, `1`, `reference`를 사용합니다. CPU 자원이 부족하면 `brain_workers`를 1로 낮출 수 있습니다.

## 뇌 보존과 재개

- `--reset-mode A`: 몸 복구 + 신경 상태/가중치 초기화.
- `--reset-mode B` (기본): 몸 복구 + membrane, synaptic current, plastic delta, CNS eligibility/trace 및 학습 운동 가중치 보존.
- `--reset-mode C`: 몸 복구 + plastic delta, CNS trace, 학습 운동 가중치를 `partial_decay` 비율로 감소.

몸은 경기 시작 시 joint state, 위치/방향, integrity, hemolymph, fatigue, injury를 복구합니다. Energy/Hunger는 `body.reset_energy`로 유지 여부를 정합니다. 행동불능이 발생하면 경기 전체를 종료하고 다음 경기에서 복구합니다.

```powershell
.\.venv\Scripts\python.exe -m flyfight train --resume results/my_on/checkpoint.npz --config configs/quick.yaml --episodes 20 --output results/continued
```

재개할 때 원래 config, population, dataset, seed를 유지하세요. 추가 episode 수를 지정합니다. 체크포인트는 match RNG와 각 brain RNG를 저장합니다. 중단된 경기의 부분 상태는 완료 상태로 저장하지 않습니다. 초기 및 주기적 checkpoint에서 재개할 수 있으며, 마지막 checkpoint 이후 완료한 경기 일부는 재실행될 수 있습니다.

## 분석과 일반화 평가

```powershell
.\.venv\Scripts\python.exe -m flyfight analyze results/my_on --compare results/my_off
.\.venv\Scripts\python.exe -m flyfight evaluate --checkpoint results/my_on/checkpoint.npz --trained-agent 0 --naive-opponents 5 --episodes 20 --config configs/quick.yaml --output results/naive_evaluation
```

분석은 win/retreat, damage 후 다음 interval retreat, 이전 승패와 다음 경기, 개체별 unlabeled PCA, 가중치 변화, 상대 다양성을 그림과 Parquet으로 저장합니다. 새 연속 운동 decoder의 공격 행동은 아직 분류하지 않으므로 attack 관련 값은 누락으로 표시합니다. `evaluate`는 학습된 CNS delta와 운동 출력층 가중치를 가져와 plasticity/탐색을 끄고 새로운 naive 상대와 비교합니다. 순간적 신경 상태는 매 경기 초기화합니다. 먹이 획득량/행동불능에서 계산한 winner는 분석용이고 CNS 입력이나 reward로 사용하지 않습니다. 상대 다양성 그래프 자체는 일반화의 인과적 증거가 아닙니다.

## 설정

| 파일 | 내용 |
|---|---|
| default.yaml | seed, 경로, 메모리 한도, population, 반복 횟수 |
| neural.yaml | LIF dt/decay/refractory, NT sign, count→weight 변환, encoder/decoder gain |
| learning.yaml | plastic 후보, 최대 수, STDP/eligibility, homeostasis reward, decay |
| body.yaml | Energy/Fatigue, part integrity, 손상·출혈·행동불능 조건 |
| arena.yaml | 물리/제어 dt, 시작 거리, arena 크기, 먹이 양·범위·재생 |
| quick.yaml | 짧은 통합 회귀 검증용 override |
| selfplay.yaml | 가상 10초 반복 학습, 매 경기 checkpoint 저장 |
| incapacity_regression.yaml | 매우 약한 가상 조직으로 death/reset 경로를 검증하는 override |

`--config 파일.yaml`은 기본 5개 설정을 깊은 병합으로 덮어씁니다. 손상은 `max(force-threshold,0) × physics_dt × multiplier`를 매 physics substep 적분합니다. 힘은 MuJoCo/FlyGym 모델의 단위이며 생물 조직 측정치로 보정되지 않았습니다.

## 32GB RAM / 8GB VRAM 권장

- 기본 CPU sparse backend와 float32 neural state를 사용하세요. **CUDA 신경 backend는 아직 구현되지 않았으며**, `device`가 다른 값이면 명시적으로 CPU로 fallback합니다. 신경 계산은 VRAM 0을 사용합니다.
- topology는 mmap으로 한 번 로드하고 공유합니다. 20개 brain pool도 전체 graph 20벌을 만들지 않습니다.
- 기본 RAM safety limit은 28GiB이며 profile에 RSS, CPU, 단계/초, 경기/초, nvidia-smi GPU 전체 사용량을 기록합니다. GPU 값은 다른 앱을 포함합니다.
- rendering은 headless 학습과 분리하세요. nvidia-smi 전체 사용량이 8GiB에서 256MiB 이내로 접근하면 offscreen renderer를 끄고 CPU 계산을 계속합니다. OS/다른 앱의 VRAM 사용까지 이 프로그램이 제한할 수는 없습니다. GUI 창 자체는 별도의 OpenGL context를 사용하므로 메모리가 제한적이면 headless를 선택하세요.
- 긴 실험 전 `smoke`, `quick.yaml`, 짧은 평가로 메모리와 발화율을 확인하세요. 10,000회 CLI 지원은 해당 횟수의 완료/학습 성능을 검증했다는 뜻이 아닙니다.

## 알려진 한계

실제 MaleCNS 구조를 사용하지만 NeuroMechFly의 female morphology를 몸의 대용으로 사용합니다. feature sensory mapping과 DN shard decoder는 heuristic입니다. DAN의 activity와 보상 scalar는 별개이며, 실제 도파민 compartment별 작동은 재현하지 않습니다. NT sign 역시 receptor를 고려하지 못합니다. 강한 recurrent excitation과 과도한 발화가 나타날 수 있어 생리적 calibration이 필요합니다.

다리 관절/보행 손상은 actuator에 반영됩니다. 양 앞다리의 관절과 부착은 독립 연속 출력으로 제어합니다. 날개·더듬이 integrity는 보관하고 더듬이/머리 손상은 감각 입력에 반영하지만, wing/antenna articulation, 비행, 구애 노래, 검증된 lunge/grapple biomechanics는 아직 구현하지 않았습니다. 과거 고정 decoder의 `lunge`는 보행 출력 증폭일 뿐이며 기본 학습 decoder는 이를 사용하지 않습니다. 음식을 먹는 실제 구기 운동도 없습니다.

이 버전은 작동하는 실험 파이프라인입니다. 안정된 전략의 출현, winner/loser effect, 생물학적으로 타당한 학습, 여러 seed에서의 통계적 유의성은 아직 입증되지 않았습니다. 검증 결과는 [docs/VALIDATION.md](docs/VALIDATION.md)를 참고하세요.

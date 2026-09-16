# 스스로 행동을 배우는 경로

## 현재 상태

완성된 학습형 전투 모델이 아니라 **학습 실험이 가능한 프로토타입**입니다. 기존 버전은 실제 연결 일부의 STDP 가중치가 달라졌지만, 고정 운동 decoder 때문에 행동 선택을 개선하는 경로가 부족했습니다. 100회의 아주 짧은 실행은 학습 성능을 증명하지 못했습니다.

현재는 전체 166,700개 뉴런/25,582,938개 연결의 LIF 시뮬레이션을 그대로 사용하고, descending activity 뒤에 개체별 작은 학습 출력층을 추가했습니다. 이 출력층의 363개 파라미터는 **공학적 adapter이며 MaleCNS에서 관찰된 시냅스가 아닙니다.** 실제 연결의 최대 20,000개 R-STDP delta와 별도로 저장·보고합니다. 전체 신경망의 BPTT/Adam은 사용하지 않습니다.

## 행동과 학습 신호

```text
감각 feature → 전체 MaleCNS LIF → 32개 descending activity 묶음
   → 학습 가능한 연속 운동 출력
   → 왼쪽/오른쪽 보행 + 양 앞다리의 관절 3개씩 + 발 부착 정도
   → 실제 관절/접촉/먹이 섭취
   → 에너지 변화와 부상 → 최근 운동 선택의 eligibility 갱신
```

- 기본 0.1초마다 운동 출력을 선택합니다. 훈련에서는 Gaussian noise로 탐색하며 다음 선택까지 유지합니다. `--exploration off`는 탐색을 끕니다.
- 입력은 실제 descending spike count를 평활화한 값과 bias입니다. 출력층에 상대 위치·먹이 방향·승패 정보를 직접 주지 않습니다.
- actor는 어떤 운동을 할지, 작은 critic은 그 신경 활동 상태에서 예상되는 보상을 학습합니다. 시간 차 보상 오차와 eligibility trace로 국소 가중치를 갱신합니다. CNS를 역전파하거나 과거 trajectory를 무한히 저장하지 않습니다.
- arena 보상은 `5 × 실제 Energy 변화 − 2 × 새로 받은 injury`입니다. Energy 변화에는 섭취, 기초 대사, 이동·관절 동작 비용, 최대 에너지 제한이 포함됩니다. 초당 modulation을 ±5로 제한하고 시간 적분합니다.
- 공격, 접촉 횟수, 상대에게 가한 피해, 승리에 대한 별도 보상은 없습니다. **싸우지 않고 먹이를 확보하는 방법도 유효합니다.**
- 앞다리에는 관절별 offset 범위와 부착 제어만 제공합니다. `상대가 가까우면 공격` 규칙, lunge/boxing 타임라인, 공격/후퇴 상태 머신은 없습니다. 기존 보행 CPG 자체는 제공된 운동 primitive입니다.
- 앞다리 target은 물리 substep에서 부드럽게 연결하고 joint range로 제한합니다. 부상에 따른 actuator 약화도 그대로 적용됩니다. wings/antennae articulation, 구기 섭식, 검증된 lunge biomechanics는 아직 없습니다.

방법 참고: [Degris, Pilarski & Sutton, Model-Free Reinforcement Learning with Continuous Action in Practice](https://people.bordeaux.inria.fr/degris/papers/DegrisACC2012.pdf)의 연속 행동 actor–critic/eligibility 접근을 참고했습니다. 이 구현의 feature, 보상, 시간 상수, 한계값은 별도 공학적 가정이며 그 논문의 결과를 재현했다고 주장하지 않습니다.

## 실행

반복 훈련하면서 관찰하려면:

```powershell
.\.venv\Scripts\python.exe -m flyfight train --config configs/selfplay.yaml --episodes 100 --gui --output results/selfplay_01
```

GUI 없이 같은 학습은 `--gui` 대신 `--headless`와 새 출력 경로를 지정합니다. `selfplay.yaml`은 경기당 가상 시간 10초, 매 경기 checkpoint 저장 설정입니다. 짧은 `quick.yaml`은 실행 점검용입니다. **100회면 싸움을 배운다는 보장은 없습니다.** 전체 계산은 아직 실시간보다 느립니다. 가상 10초가 실제 수 분 걸릴 수 있으며 GUI의 Compute speed를 확인하세요. 화면 보간은 학습이나 물리를 가속하지 않습니다.

중단 후 이어서 훈련하려면 새 출력 경로를 지정합니다:

```powershell
.\.venv\Scripts\python.exe -m flyfight train --config configs/selfplay.yaml --resume results/selfplay_01/checkpoint.npz --episodes 100 --gui --output results/selfplay_02
```

저장된 두 개체를 추가 학습·탐색 없이 관찰하려면:

```powershell
.\.venv\Scripts\python.exe -m flyfight fight --config configs/selfplay.yaml --resume results/selfplay_01/checkpoint.npz --plasticity off --exploration off --gui
```

`fight --gui`만 실행하면 **새 뇌로 한 경기**를 실행합니다. 기존 checkpoint를 자동으로 고르지 않습니다. `--resume`은 population/config가 일치해야 합니다. 새 버전 이전 checkpoint에는 학습 출력층이 없으므로 경고 후 해당 출력층은 미학습 상태로 시작합니다.

## 저장·대조군

- `checkpoint.npz`: 실제 CNS 상태/가소성과 motor actor/critic, trace, RNG를 각각 저장합니다.
- Mode B: CNS 상태와 학습 출력층 가중치 보존. Mode A: 둘 다 초기화. Mode C: 둘의 학습 가중치를 설정 비율로 감소.
- 몸을 복구해 위치를 바꾸는 것은 행동 결과가 아니므로 motor 출력층의 짧은 실행 trajectory/trace는 경기 경계에서 초기화합니다. 기존 CNS의 membrane/eligibility 보존 규칙은 바꾸지 않습니다.
- `--plasticity off`: CNS 및 motor 출력층 가중치를 모두 동결합니다. 탐색은 그대로 두므로 ON/OFF는 동일한 탐색 조건에서 비교할 수 있습니다.
- config의 `learning.enabled: false`: CNS R-STDP만 끕니다. `motor_learning.plasticity: false`: 같은 연속 운동 출력 구조를 유지하며 출력층 학습만 끕니다. 두 요인의 기여를 분리하는 실험에 사용합니다.
- `motor_learning.enabled: false`: 과거 고정 decoder로 돌아가는 재현용 옵션입니다. 현재 모델과 행동 공간이 다르므로 공정한 출력층 학습 대조군으로 쓰면 안 됩니다.
- `evaluate`: 학습 가중치만 가져와 낯선 naive 상대와 평가합니다. 매 경기 신경의 순간 상태를 초기화하고 탐색과 가중치 갱신을 모두 끕니다. 단순 시청과 다른 평가 절차입니다.

## 무엇을 검증했는가

1. 합성 actuator calibration에서 상황별 연속 출력을 학습: 여러 seed에서 동결/상황을 섞은 보상 대조군과 비교. **실제 초파리나 먹이 경쟁 실험이 아닙니다.**
2. 실제 전체 MaleCNS + 두 MuJoCo 몸: 출력층·CNS 가중치 변화, OFF 불변성, 실제 앞다리 관절 제어, 저장/복구, frozen 평가를 검증.
3. 먹이 탐색·경쟁 전략 개선, 여러 상대에 대한 일반화, 자발적 전투와 실시간 계산: **아직 입증하지 못했습니다.**

앞다리가 움직이거나 몸이 부딪쳤다는 이유로 공격 성공을 기록하지 않습니다. 새 decoder의 `attack_attempts`와 `successful_contacts`는 `null`, `attack_measurement`는 `unclassified`입니다. 대신 앞다리 운동 시간·실제 접촉 시간·힘/손상·먹이·에너지·운동 가중치를 기록합니다. 공격 분류기는 물리 trajectory와 관찰 영상을 근거로 별도로 검증해야 합니다.

다음 성능 판단은 충분히 긴 학습 후 **동결한 정책을 같은 조건의 미학습 정책과 비교**해야 합니다. 독립 seed, 학습에 쓰지 않은 상대·시작 위치에서 먹이 확보/순 에너지/부상 비용을 측정해야 하며 훈련 중 우연한 탐색 동작이나 가중치 변화만으로 학습 성공을 선언하면 안 됩니다.

from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parent.parent

def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a

def load_config(path=None):
    c = {}
    for name in ['default', 'neural', 'learning', 'body', 'arena']:
        merge(c, yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text(encoding='utf-8')))
    if path:
        merge(c, yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {})
    n, a = c['neural'], c['arena']
    if n['dt'] <= 0 or a['control_dt'] <= 0 or a['physics_dt'] <= 0:
        raise ValueError('Timesteps must be positive')
    if not 1 <= a['gui_fps'] <= 120 or a['render_fps'] <= 0:
        raise ValueError('gui_fps must be 1..120 and render_fps must be positive')
    for small in [n['dt'], a['physics_dt']]:
        if abs(a['control_dt']/small-round(a['control_dt']/small)) > 1e-6:
            raise ValueError('control_dt must be an integer multiple of neural/physics dt')
    if c['learning']['max_synapses'] < 0 or not 0 <= c['learning']['partial_decay'] <= 1:
        raise ValueError('Invalid plasticity configuration')
    m=c['motor_learning']
    if m['feature_groups']<2 or m['feature_groups']%2:
        raise ValueError('Motor feature_groups must be a positive even number >= 2')
    for key in ['decision_dt','feature_tau','exploration_std','actor_rate','critic_rate',
                'discount_tau','eligibility_tau','td_clip','trace_clip','weight_bound','actuator_smoothing_tau']:
        if m[key]<=0: raise ValueError(f'motor_learning.{key} must be positive')
    if m['decision_dt']<a['control_dt'] or abs(m['decision_dt']/a['control_dt']-round(m['decision_dt']/a['control_dt']))>1e-6:
        raise ValueError('motor decision_dt must be an integer multiple of control_dt')
    if len(m['joint_offset_radians'])!=3 or any(x<0 for x in m['joint_offset_radians']):
        raise ValueError('Expected three nonnegative motor joint offsets')
    return c

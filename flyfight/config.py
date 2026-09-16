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
    for small in [n['dt'], a['physics_dt']]:
        if abs(a['control_dt']/small-round(a['control_dt']/small)) > 1e-6:
            raise ValueError('control_dt must be an integer multiple of neural/physics dt')
    if c['learning']['max_synapses'] < 0 or not 0 <= c['learning']['partial_decay'] <= 1:
        raise ValueError('Invalid plasticity configuration')
    return c

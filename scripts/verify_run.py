"""Verify completed ON/OFF experiments without interpreting biological meaning."""
import argparse
import json
import numpy as np
from flyfight.logging.episode_logger import read

p=argparse.ArgumentParser(); p.add_argument('on'); p.add_argument('off'); p.add_argument('--episodes',type=int,default=100)
a=p.parse_args()
on,off=read(a.on),read(a.off)
assert len(on)==len(off)==a.episodes
assert [r['seed'] for r in on]==[r['seed'] for r in off]
assert [(r['fly_a_id'],r['fly_b_id']) for r in on]==[(r['fly_a_id'],r['fly_b_id']) for r in off]
for ep in off:
    assert all(x['plasticity']['modified_synapses']==0 for x in ep['agents'])
assert any(x['plasticity']['modified_synapses']>0 for ep in on for x in ep['agents'])
for records in [on,off]:
    for r in records:
        assert all(np.isfinite(x['distance_traveled']) and x['food_consumed']>=0 for x in r['agents'])
result={'episodes_each':len(on),'paired_seeds':True,'control_weights_unchanged':True,
        'on_food':sum(x['food_consumed'] for ep in on for x in ep['agents']),
        'off_food':sum(x['food_consumed'] for ep in off for x in ep['agents']),
        'on_damage':sum(x['damage_received'] for ep in on for x in ep['agents']),
        'on_final_modified':[x['plasticity']['modified_synapses'] for x in on[-1]['agents']]}
print(json.dumps(result,indent=2))

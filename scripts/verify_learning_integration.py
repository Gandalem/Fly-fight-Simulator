"""Audit actual paired arena runs, resume, and frozen transfer without combat claims."""
import argparse
import json
from pathlib import Path
import numpy as np
from flyfight.logging.episode_logger import read


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--on',required=True)
    p.add_argument('--off',required=True)
    p.add_argument('--continuous',required=True)
    p.add_argument('--resumed',required=True)
    p.add_argument('--frozen',required=True)
    p.add_argument('--output',default='results/motor_learning_integration.json')
    args=p.parse_args()
    on,off,continuous,resumed=map(read,[args.on,args.off,args.continuous,args.resumed])
    assert len(on)==len(off)
    assert [x['seed'] for x in on]==[x['seed'] for x in off]
    assert [(x['fly_a_id'],x['fly_b_id']) for x in on]==[(x['fly_a_id'],x['fly_b_id']) for x in off]
    assert all(a['motor_learning']['changed_actor_weights']==0 and a['plasticity']['modified_synapses']==0
               for ep in off for a in ep['agents'])
    assert all(a['motor_learning']['changed_actor_weights']>0 for a in on[-1]['agents'])
    assert resumed[-1]['episode_id']==continuous[-1]['episode_id']
    assert resumed[-1]['seed']==continuous[-1]['seed']
    assert resumed[-1]['agents']==continuous[-1]['agents']
    with np.load(Path(args.continuous)/'checkpoint.npz') as a,np.load(Path(args.resumed)/'checkpoint.npz') as b:
        for key in a.files:
            if key!='metadata': np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        ma,mb=json.loads(str(a['metadata'])),json.loads(str(b['metadata']))
        assert ma['match_rng']==mb['match_rng']
        for aa,bb in zip(ma['agents'],mb['agents']):
            assert aa['rng']==bb['rng'] and aa['motor']['rng']==bb['motor']['rng']
    with np.load(Path(args.on)/'checkpoint.npz') as trained,np.load(Path(args.frozen)/'checkpoint.npz') as frozen:
        for suffix in ('delta','motor_weights','motor_value_weights'):
            np.testing.assert_array_equal(trained['0_'+suffix],frozen['0_'+suffix])
    report=dict(scope='Full MaleCNS and real MuJoCo bodies; integration checks, not evidence of learned combat.',
                paired_episodes=len(on),frozen_control_weights_unchanged=True,
                exact_resume_arrays_outcomes_and_rng=True,frozen_evaluation_weights_unchanged=True,
                on_food=sum(a['food_consumed'] for ep in on for a in ep['agents']),
                off_food=sum(a['food_consumed'] for ep in off for a in ep['agents']),
                motor_updates=[a['motor_learning'] for a in on[-1]['agents']],
                inputs={k:getattr(args,k) for k in ('on','off','continuous','resumed','frozen')})
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

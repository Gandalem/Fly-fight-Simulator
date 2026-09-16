from pathlib import Path
import json
import numpy as np
import warnings
from ..embodiment.motor_learning import MotorLearningState

ARRAYS=['voltage','current','refractory','spikes','pre_trace','post_trace','delta','eligibility','total_spikes']

def save(path,pool,episode,rng):
    path=Path(path); values={}
    meta=dict(version=2,episode=episode,match_rng=rng.bit_generator.state,agents=[])
    for fly in pool:
        meta['agents'].append(dict(id=fly.id,rng=fly.brain.rng.bit_generator.state,energy=fly.internal.energy,
                                  motor=fly.brain.motor.metadata()))
        for key in ARRAYS: values[f'{fly.id}_{key}']=getattr(fly.brain,key)
        for key in MotorLearningState.ARRAY_NAMES: values[f'{fly.id}_motor_{key}']=getattr(fly.brain.motor,key)
    values['metadata']=np.array(json.dumps(meta))
    values['neuron_ids']=pool[0].brain.graph.metadata.neuron_id.to_numpy()
    values['plastic_edges']=pool[0].brain.plastic.edges
    values['weight_config']=np.array(json.dumps(pool[0].brain.graph.provenance.get('neural_weight_config',{}),sort_keys=True))
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**values); temp.replace(path)

def restore(path,pool,rng):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if len(meta['agents'])!=len(pool): raise ValueError('Checkpoint population mismatch')
        np.testing.assert_array_equal(data['neuron_ids'],pool[0].brain.graph.metadata.neuron_id.to_numpy())
        np.testing.assert_array_equal(data['plastic_edges'],pool[0].brain.plastic.edges)
        if 'weight_config' in data:
            if json.loads(str(data['weight_config']))!=pool[0].brain.graph.provenance.get('neural_weight_config',{}):
                raise ValueError('Checkpoint base-weight configuration mismatch')
        for fly,stored in zip(pool,meta['agents']):
            if fly.id!=stored['id']: raise ValueError('Checkpoint agent ID mismatch')
            for key in ARRAYS: getattr(fly.brain,key)[:]=data[f'{fly.id}_{key}']
            if 'motor' in stored:
                validate_motor_config(stored['motor']['config'],fly.brain.motor.config)
                for key in MotorLearningState.ARRAY_NAMES:
                    getattr(fly.brain.motor,key)[:]=data[f'{fly.id}_motor_{key}']
                fly.brain.motor.restore_metadata(stored['motor'])
            else:
                warnings.warn('Legacy checkpoint: CNS state restored; motor readout starts untrained.')
            fly.brain.rng.bit_generator.state=stored['rng']; fly.internal.energy=stored['energy']
        rng.bit_generator.state=meta['match_rng']
        return meta['episode']

def transfer_weights(path,fly,source_agent=0):
    """Frozen evaluation transfers plastic weights, starts transient activity anew."""
    with np.load(path,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['neuron_ids'],fly.brain.graph.metadata.neuron_id.to_numpy())
        np.testing.assert_array_equal(data['plastic_edges'],fly.brain.plastic.edges)
        if 'weight_config' in data and json.loads(str(data['weight_config']))!=fly.brain.graph.provenance.get('neural_weight_config',{}):
            raise ValueError('Checkpoint base-weight configuration mismatch')
        fly.brain.delta[:]=data[f'{source_agent}_delta']
        if f'{source_agent}_motor_weights' in data:
            meta=json.loads(str(data['metadata']))
            stored=next(x for x in meta['agents'] if x['id']==source_agent)
            validate_motor_config(stored['motor']['config'],fly.brain.motor.config)
            fly.brain.motor.weights[:]=data[f'{source_agent}_motor_weights']
            fly.brain.motor.value_weights[:]=data[f'{source_agent}_motor_value_weights']
        else:
            warnings.warn('Legacy checkpoint has no learned motor readout.')


def validate_motor_config(stored,current):
    # Exploration is deliberately disabled for frozen viewing/evaluation.
    for key in set(stored)|set(current):
        if key not in ('exploration','plasticity') and stored.get(key)!=current.get(key):
            raise ValueError(f'Checkpoint motor configuration mismatch: {key}')

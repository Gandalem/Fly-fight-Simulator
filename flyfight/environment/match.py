import numpy as np
from .food import Food
from ..neural.lif import lif_step
from ..neural.plasticity import update,statistics
from ..neural.neuromodulation import homeostatic_signal
from ..combat.death import incapacity
from ..logging.neural_logger import summarize

def run_match(env,flies,sensory,motor,config,episode_id,seed,render=None,progress=None):
    a=config['arena']; dt=a['control_dt']; neural_dt=config['neural']['dt']
    rng=np.random.default_rng(seed)
    for fly in flies:
        fly.reset_body(config)
        fly.brain.reset(config['training']['reset_mode'],config['learning']['partial_decay'])
        if config.get('evaluation'):
            learned=fly.brain.delta.copy()
            fly.brain.reset('A')
            fly.brain.delta[:]=learned
    env.reset(seed)
    food=Food(config)
    fields=['food_consumed','food_ownership_time','distance_traveled','attack_attempts','successful_contacts',
            'damage_dealt','damage_received','retreat_count','approach_count','energy_spent']
    stats=[dict(fly_id=f.id,**{k:0. for k in fields},hunger_before=f.internal.hunger) for f in flies]
    previous=env.positions(); velocities=np.zeros_like(previous)
    contact=[{} for f in flies]; modulation=np.zeros(len(flies)); causes=[None]*len(flies)
    last_attack=np.zeros(len(flies),bool); last_direction=np.zeros(len(flies),np.int8)
    prev_damage=np.zeros(len(flies)); damage_retreat_events=np.zeros(len(flies)); damaged_intervals=np.zeros(len(flies))
    neural_steps=0
    for tick in range(max(1,round(a['duration']/dt))):
        positions=env.positions(); headings=env.headings(); commands=[]
        for i,f in enumerate(flies):
            features=sensory.features(i,positions,headings,velocities,f.body,f.internal,contact[i],float(food.amount>0))
            drive=sensory.encode(features,f.body)
            counts=np.zeros(f.brain.graph.n,np.uint16)
            for _ in range(round(dt/neural_dt)):
                counts+=lif_step(f.brain,drive,config)
                update(f.brain,modulation[i],config,config['training']['plasticity'] and config['learning']['enabled'])
                neural_steps+=1
            cmd=motor.decode(counts,dt); commands.append(cmd)
            attack=cmd['lunge']>.5
            stats[i]['attack_attempts']+=int(attack and not last_attack[i]); last_attack[i]=attack
        result=env.step(commands,[f.body for f in flies],[f.internal for f in flies])
        current=env.positions(); velocities=(current-previous)/dt
        intake,owner=food.consume(current,[f.internal for f in flies],dt,rng)
        for i,f in enumerate(flies):
            movement=float(np.mean(np.abs(commands[i]['legs'])))
            spent=f.internal.step(dt,movement,commands[i]['lunge'],intake[i],f.body.hemolymph,config)
            f.body.update(dt,config,float(result['distance'][i]/dt))
            modulation[i]=homeostatic_signal(intake[i],result['damage_received'][i],dt,config)
            contact[i]=result['part_force'][i]
            s=stats[i]
            for key,val in [('food_consumed',intake[i]),('food_ownership_time',dt if owner==i else 0),
                            ('distance_traveled',result['distance'][i]),('damage_dealt',result['damage_dealt'][i]),
                            ('damage_received',result['damage_received'][i]),('energy_spent',spent)]: s[key]+=float(val)
            s['successful_contacts']+=int(result['contact_active'][i] and commands[i]['lunge']>.5)
            others=[j for j in range(len(flies)) if j!=i]
            old_dist=np.min(np.linalg.norm(previous[others,:2]-previous[i,:2],axis=1))
            new_dist=np.min(np.linalg.norm(current[others,:2]-current[i,:2],axis=1))
            direction=1 if new_dist<old_dist-1e-3 else -1 if new_dist>old_dist+1e-3 else 0
            s['approach_count']+=int(direction==1 and last_direction[i]!=1)
            s['retreat_count']+=int(direction==-1 and last_direction[i]!=-1)
            if prev_damage[i]>0:
                damaged_intervals[i]+=1; damage_retreat_events[i]+=int(direction==-1)
            last_direction[i]=direction
            causes[i]=incapacity(f.body,config)
        prev_damage=result['damage_received'].copy(); previous=current
        if render and tick % max(1,round(1/(a['render_fps']*dt)))==0:
            frame=env.frame()
            if frame is not None: render(frame,flies,episode_id,(tick+1)*dt)
        if progress: progress(neural_steps)
        if any(causes): break
    duration=(tick+1)*dt
    # Apply final interval's reward to eligibility; no extra physics or neural tick.
    for i,f in enumerate(flies):
        if config['training']['plasticity'] and config['learning']['enabled']:
            bound=np.abs(f.brain.plastic.base)*config['learning']['max_relative_change']
            f.brain.delta[:]=np.clip(f.brain.delta+config['learning']['learning_rate']*modulation[i]*dt*f.brain.eligibility,-bound,bound)
        stats[i].update(hunger_after=f.internal.hunger,death_cause=causes[i],
            body_parts_damaged=[k for k,p in f.body.parts.items() if p.current_integrity<p.max_integrity],
            body_parts_destroyed=[k for k,p in f.body.parts.items() if p.current_integrity<=0],
            integrity={k:p.current_integrity for k,p in f.body.parts.items()},hemolymph=f.body.hemolymph,
            damage_then_retreat_probability=float(damage_retreat_events[i]/damaged_intervals[i]) if damaged_intervals[i] else None,
            neural=summarize(f.brain,duration),plasticity=statistics(f.brain))
    # Analysis-only outcome: sole capable agent, else strictly most food, else draw.
    alive=[i for i,c in enumerate(causes) if c is None]
    food_values=np.array([s['food_consumed'] for s in stats])
    if len(alive)==1: winning=alive[0]; outcome='incapacity'
    elif food_values.max()>0 and np.sum(np.isclose(food_values,food_values.max()))==1: winning=int(np.argmax(food_values)); outcome='food_acquisition'
    else: winning=None; outcome='draw'
    winner=flies[winning].id if winning is not None else None
    return dict(episode_id=episode_id,fly_a_id=flies[0].id,fly_b_id=flies[1].id,
                winner=winner,loser=next((f.id for f in flies if f.id!=winner),None) if winner is not None and len(flies)==2 else None,
                duration=duration,termination='incapacity' if any(causes) else 'time_limit',outcome=outcome,
                food_remaining=food.amount,agents=stats,neural_steps=neural_steps,seed=seed)

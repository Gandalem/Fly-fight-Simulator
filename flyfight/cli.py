import argparse
import json
from pathlib import Path
from .config import load_config
from .connectome import prepare_connectome, load_connectome
from .runtime import Monitor

def main():
    p=argparse.ArgumentParser(description='MaleCNS embodied sparse spiking prototype')
    sub=p.add_subparsers(dest='command',required=True)
    for command in ['prepare-connectome','inspect-connectome','smoke','fight','train','evaluate','analyze']:
        s=sub.add_parser(command)
        s.add_argument('--config')
        if command in ['prepare-connectome','inspect-connectome','smoke','fight','train','evaluate']: s.add_argument('--data')
        if command == 'smoke':
            s.add_argument('--steps',type=int,default=100)
            s.add_argument('--agents',type=int,default=2)
            s.add_argument('--stimulate',nargs='*',type=int)
        if command in ['train','fight','evaluate']:
            s.add_argument('--episodes',type=int)
            s.add_argument('--population',type=int)
            s.add_argument('--agents',type=int,default=2)
            s.add_argument('--render',action='store_true')
            s.add_argument('--gui',action='store_true')
            s.add_argument('--headless',action='store_true')
            s.add_argument('--random-matchmaking',action='store_true')
            s.add_argument('--plasticity',choices=['on','off'])
            s.add_argument('--reset-mode',choices=['A','B','C'])
            s.add_argument('--seed',type=int)
            s.add_argument('--output')
            s.add_argument('--resume')
        if command=='evaluate':
            s.add_argument('--checkpoint',required=True)
            s.add_argument('--trained-agent',type=int,default=0)
            s.add_argument('--naive-opponents',type=int,default=5)
        if command=='analyze':
            s.add_argument('run')
            s.add_argument('--compare')
    args=p.parse_args()
    c=load_config(args.config)
    if args.command=='analyze':
        from .analysis.plots import analyze
        analyze(args.run,args.compare); return
    monitor=Monitor(c)
    path=args.data or c['dataset']['processed']
    if args.command=='prepare-connectome':
        graph=prepare_connectome(c['dataset']['raw'],path,c)
    else: graph=load_connectome(path)
    if args.command in ['prepare-connectome','inspect-connectome']:
        print(json.dumps({'graph':graph.stats(), 'measured':monitor.sample()},indent=2)); return
    if args.command=='smoke':
        from .neural.simulator import smoke
        print(json.dumps(smoke(graph,c,args.steps,args.agents,args.stimulate),indent=2)); return
    from .training.trainer import train
    if args.seed is not None: c['seed']=args.seed
    if args.episodes is not None: c['training']['episodes']=args.episodes
    if args.population is not None: c['training']['population']=args.population
    if args.plasticity: c['training']['plasticity']=args.plasticity=='on'
    if args.reset_mode: c['training']['reset_mode']=args.reset_mode
    if args.command=='fight': c['training']['episodes']=1
    if args.random_matchmaking: c['training']['random_matchmaking']=True
    if args.command=='evaluate':
        if args.naive_opponents<1: p.error('--naive-opponents must be positive')
        c['evaluation']={'checkpoint':args.checkpoint,'agent':args.trained_agent,'protocol':'frozen learned weights vs naive, transient state reset each episode'}
        c['training'].update(plasticity=False,reset_mode='B',population=args.naive_opponents+1)
        args.agents=2
    print(train(graph,c,args.output,args.render and not args.headless,args.agents,args.resume,args.gui and not args.headless))

"""Run repeatable causal credit tests; this does not establish fly learning."""
import argparse
import json
from pathlib import Path
from flyfight.training.motor_benchmark import run_benchmark


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=1500)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2,7,19,31,43])
    parser.add_argument('--output',default='results/motor_learning_benchmark.json')
    args=parser.parse_args()
    records=[run_benchmark(seed,args.trials) for seed in args.seeds]
    result=dict(scope='Synthetic readout credit assignment only; not MaleCNS, food seeking, combat or real-time validation.',
                records=records,passes=all(x['learned_mse']<.4*x['frozen_mse'] and x['reward_shuffled_mse']>2*x['learned_mse'] for x in records))
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passes']: raise SystemExit(1)


if __name__=='__main__': main()

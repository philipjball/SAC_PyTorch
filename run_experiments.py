import argparse
import datetime
import multiprocessing
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='HalfCheetah-v2')
parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--seeds5to9', dest='seeds5to9', action='store_true')
parser.add_argument('--total_steps', type=int, default=int(3e6))
parser.set_defaults(seeds5to9=False)

args = parser.parse_args()
params = vars(args)

experiment_id = 'runs/' + params['experiment_name']
num_experiments = 5
seeds_5to9 = params['seeds5to9']
lower = 0
upper = num_experiments

main_experiment = ["python", "train_agent.py", "--env", params['env'], "--experiment_name", experiment_id, "--n_evals", str(10), "--seed"]

if seeds_5to9:
    lower += 5
    upper += 5

all_experiments = [main_experiment + [str(i)] for i in range(lower, upper)]

def run_experiment(spec):
    subprocess.run(spec, check=True)

def run_all_experiments(specs):
    pool = multiprocessing.Pool()
    pool.map(run_experiment, specs)

run_all_experiments(all_experiments)

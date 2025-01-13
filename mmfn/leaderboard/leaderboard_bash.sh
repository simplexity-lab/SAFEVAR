conda activate mmfn

# << Leaderboard setting
# ===> pls remeber to change this one
export CODE_FOLDER=/home/kin/mmfn
export CARLA_ROOT=/home/kin/CARLA_0.9.10.1
# ===> pls remeber to change this one
export SCENARIO_RUNNER_ROOT=${CODE_FOLDER}/scenario_runner
export LEADERBOARD_ROOT=${CODE_FOLDER}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg":"${CODE_FOLDER}/team_code":${PYTHONPATH}

python run_steps/phase0_run_eval.py port=2010 towns="['Town02', 'Town04', 'Town05', 'Town10']" resume=True if_open_carla=True

# if you have big GPU memory, you can start with these two simultaneously.
python run_steps/phase0_run_eval.py port=2000 towns="['Town01', 'Town03', 'Town06', 'Town07']" resume=True if_open_carla=True
```
python run_steps/phase0_run_eval.py --config-anme=eval

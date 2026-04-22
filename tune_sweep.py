"""
Ray Tune grid search sweep with W&B logging.

Launch from the head node after `ray up ray_cluster.yaml`:
    python tune_sweep.py --wandb-project minesweeper-rl --max-clicks 2000000
"""
import argparse
import ray
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback

from common_constants import *
from training_config import TrainingConfig
from train import run_training, MODEL_NAME


# ---------------------------------------------------------------------------
# Search grid — add or remove axes as needed
# ---------------------------------------------------------------------------
SEARCH_GRID = {
    'learn_rate':                     tune.grid_search([1e-3, 5e-3, 1e-2]),
    'discount':                       tune.grid_search([0.05, 0.1, 0.3, 0.6, 0.9]),
    'train_every_n_clicks':           tune.grid_search([50, 100, 250, 500, 750, 1000, 1500, 2000]),
    'batch_size':                     tune.grid_search([64, 128, 256, 512]),
    'update_target_every_n_episodes': tune.grid_search([1, 5, 10]),
    'reward_progress':                tune.grid_search([0.1, 0.3]) 
}


def train_trial(config: dict) -> None:
    trial_id = ray.train.get_context().get_trial_id()
    cfg = TrainingConfig(**config)
    run_training(
        cfg,
        model_name=f'sweep_{trial_id}',
        max_clicks=config['_max_clicks'],
        prompt=False,
        handle_sigint=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-project', default='minesweeper-rl')
    parser.add_argument('--max-clicks', type=int, default=2_000_000)
    args = parser.parse_args()

    ray.init(
        address='auto',
        runtime_env={
            'working_dir': '.',
            'excludes': ['models/', 'replay/', 'logs/', '.git/', '__pycache__/', 'notes', '.venv/', 'img/'],
            # Suppress pygame display errors on headless workers
            'env_vars': {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'},
        },
    )

    # Inject max_clicks into each trial's config dict
    param_space = {**SEARCH_GRID, '_max_clicks': args.max_clicks}

    tuner = tune.Tuner(
        tune.with_resources(train_trial, resources={'cpu': 1, 'gpu': 1}),
        #param_space=param_space,
        param_space= {
            'learn_rate': LEARN_RATE,
            'discount': DISCOUNT,
            'train_every_n_clicks': TRAIN_EVERY_N_CLICKS,
            'batch_size': BATCH_SIZE,
            'update_target_every_n_episodes': UPDATE_TARGET_EVERY_N_EPISODES,
            'reward_progress': REWARD_PROGRESS,
            '_max_clicks': args.max_clicks
        }
        tune_config=tune.TuneConfig(num_samples=1),  # grid_search already expands all combos
        run_config=ray.train.RunConfig(
            name='minesweeper_sweep',
            callbacks=[WandbLoggerCallback(project=args.wandb_project)],
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric='eval_progress_avg', mode='max')
    print('Best config:', best.config)
    print('Best metrics:', best.metrics)


if __name__ == '__main__':
    main()
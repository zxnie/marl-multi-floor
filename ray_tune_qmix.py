from gym.spaces import Tuple
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.registry import register_env
import ray.rllib.algorithms.qmix as qmix
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from env_v29_ray import GridWorldAGV

algorithm = 'qmix'
num_agents = 10
n_floors, dim_x, dim_y = 3, 3, 4
reconf_timestep = [2000, 4000, 6000, 8000]
max_ep = 100
seed = 50

env_config = {
    'dim_x': dim_x,
    'dim_y': dim_y,
    'n_floors': n_floors,
    'num_agent': num_agents,
    'reconf_timestep': reconf_timestep,
    'max_ep': max_ep,
    'seed': seed
}

env = GridWorldAGV(env_config)


def env_creator(env_config_dict):
    return GridWorldAGV(env_config_dict)


group_0 = []
for i in range(num_agents):
    group_0.append(str(i))
grouping = {"group_0": group_0}
obs_space = Tuple([env.observation_space] * num_agents)
act_space = Tuple([env.action_space] * num_agents)
register_env(
    "grid_world",
    lambda config: GridWorldAGV(config).with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)
)


class callback_custom_metrics(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics['delay'] = float(episode.last_info_for('group_0')['_group_info'][0]['delay_avg'])
        episode.custom_metrics['energy'] = float(episode.last_info_for('group_0')['_group_info'][0]['energy_avg'])


config = (
    qmix.QMixConfig()
    .environment('grid_world', env_config=env_config)
    .framework(framework='torch')
    .rollouts(num_rollout_workers=2, batch_mode="truncate_episodes")
    .callbacks(callbacks_class=callback_custom_metrics)
    # .training(num_steps_sampled_before_learning_starts=100)
    .training(lr=0.01, mixer='qmix', train_batch_size=128)
)

stop = {
    # "episodes_total": 1,
    # "training_iteration": max_eps,
    "timesteps_total": 200000,
}

tuner = tune.Tuner(
    'QMIX',
    run_config=air.RunConfig(
        name='QMIX',
        local_dir="grid_world",
        stop=stop,
        verbose=1,
        progress_reporter=CLIReporter(
            metric_columns={
                "training_iteration": "iter",
                "time_total_s": "time_total_s",
                "timesteps_total": "ts",
                "snapshots": "snapshots",
                "episodes_this_iter": "train_episodes",
                "episode_reward_mean": "reward_mean",
            },
            sort_by_metric=True,
            max_report_frequency=30, ),
    ),
    param_space=config.to_dict(),
)

results = tuner.fit()

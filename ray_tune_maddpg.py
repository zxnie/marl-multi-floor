from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.registry import register_env
import ray.rllib.algorithms.maddpg as maddpg
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from env_v29_ray import GridWorldAGV

algorithm = 'maddpg'
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


def env_creator(env_config_dict):
    return GridWorldAGV(env_config_dict)


register_env('grid_world', env_creator)


def gen_policy(i):
    config_agent = {
        'agent_id': i
    }
    return PolicySpec(config=config_agent)


policies = {str(i): gen_policy(i) for i in range(num_agents)}
policy_ids = list(policies.keys())


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id


class callback_custom_metrics(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics['delay'] = float(episode.last_info_for('0')['delay_avg'])
        episode.custom_metrics['energy'] = float(episode.last_info_for('0')['energy_avg'])


config = (
    maddpg.MADDPGConfig()
    .environment('grid_world', env_config=env_config)
    .framework(framework='tf')
    .rollouts(num_rollout_workers=0)
    .callbacks(callbacks_class=callback_custom_metrics)
    .training(num_steps_sampled_before_learning_starts=100)
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
)

stop = {
    # "episodes_total": 1,
    # "training_iteration": max_eps,
    "timesteps_total": num_agents * 20000,
}

tuner = tune.Tuner(
    'MADDPG',
    run_config=air.RunConfig(
        name='MADDPG',
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

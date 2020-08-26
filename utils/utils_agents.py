import gym
import pandas as pd
import datetime
from recogym.agents import OrganicUserEventCounterAgent, organic_user_count_args
from recogym import build_agent_init
from recogym.agents import Agent
from recogym import Configuration
from recogym import (
    gather_agent_stats,
    AgentStats
)
from recogym import env_1_args


def produce_agent_stats(
    env,
    std_env_args,
    agent: Agent,
    num_products: int,
    num_organic_users_to_train: int,
    num_users_to_train: int,
    num_users_to_score: int,
    random_seed: int,
    agent_class,
    agent_configs,
    agent_name: str,
    with_cache: bool,
):
    stat_epochs = 1
    stat_epochs_new_random_seed = True
    training_data_samples = tuple([num_users_to_train])
    testing_data_samples = num_users_to_score

    time_start = datetime.datetime.now()
    agent_stats = gather_agent_stats(
        env,
        std_env_args,
        {
            'agent': agent,
        },
        {
            **build_agent_init(
                agent_name,
                agent_class,
                {
                    **agent_configs,
                    'num_products': num_products,
                }
            ),
        },
        training_data_samples,
        testing_data_samples,
        stat_epochs,
        stat_epochs_new_random_seed,
        num_organic_users_to_train,
        with_cache
    )



    q0_025 = []
    q0_500 = []
    q0_975 = []
    for agent_name in agent_stats[AgentStats.AGENTS]:
        agent_values = agent_stats[AgentStats.AGENTS][agent_name]
        q0_025.append(agent_values[AgentStats.Q0_025][0])
        q0_500.append(agent_values[AgentStats.Q0_500][0])
        q0_975.append(agent_values[AgentStats.Q0_975][0])

    time_end = datetime.datetime.now()
    seconds = (time_end - time_start).total_seconds()

    return pd.DataFrame(
        {
            'q0.025': q0_025,
            'q0.500': q0_500,
            'q0.975': q0_975,
            'time': [seconds],
        }
    )

def create_agent_and_env_sess_pop(
    num_products: int,
    num_organic_users_to_train: int,
    num_users_to_train: int,
    num_users_to_score: int,
    random_seed: int,
    latent_factor: int,
    num_flips: int,
    log_epsilon: float,
    sigma_omega: float,
    agent_class,
    agent_configs,
    agent_name: str,
    with_cache: bool,
    reverse_pop=False
):

    std_env_args = {
        **env_1_args,
        'random_seed': random_seed,
        'num_products': num_products,
        'K': latent_factor,
        'sigma_omega': sigma_omega,
        'number_of_flips': num_flips
    }

    env = gym.make('reco-gym-v1')

    sess_pop_agent = OrganicUserEventCounterAgent(Configuration({
                **organic_user_count_args,
                **std_env_args,
                'select_randomly': True,
                'epsilon': log_epsilon,
                'num_products': num_products,
                'reverse_pop': reverse_pop
            }))

    return env, std_env_args, sess_pop_agent

def eval_against_session_pop(
    num_products: int,
    num_organic_users_to_train: int,
    num_users_to_train: int,
    num_users_to_score: int,
    random_seed: int,
    latent_factor: int,
    num_flips: int,
    log_epsilon: float,
    sigma_omega: float,
    agent_class,
    agent_configs,
    agent_name: str,
    with_cache: bool,
):
    env, std_env_args, agent = create_agent_and_env_sess_pop(num_products,
        num_organic_users_to_train,
        num_users_to_train,
        num_users_to_score,
        random_seed,
        latent_factor,
        num_flips,
        log_epsilon,
        sigma_omega,
        agent_class,
        agent_configs,
        agent_name,
        with_cache,
    )
    return produce_agent_stats(env, std_env_args, agent, num_products, num_organic_users_to_train, num_users_to_train, num_users_to_score, random_seed, agent_class, agent_configs, agent_name, with_cache)



def first_element(sc, name):
    sc['model'] = name
    sc['q0.025'] = sc['q0.025'][0]
    sc['q0.500'] = sc['q0.500'][0]
    sc['q0.975'] = sc['q0.975'][0]
    print(sc)
    return sc

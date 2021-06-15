from itertools import product
import os
import random
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from model import PGN
from utils import make_dir, save_experiment_result
from utils import initialize_proc, initialize_queue, initialize_logger


def select_action(policy, state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    probs = policy(torch.Tensor(state))
    multinomial = Categorical(probs)
    action = multinomial.sample()
    log_prob_action = multinomial.log_prob(action)
    return action, log_prob_action


def update_policy(log_prob_actions, rewards, optimizer, gamma):
    R = 0
    discounted_rewards = []

    for r in list(reversed(rewards)):
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.FloatTensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) \
        / (discounted_rewards.std() + 1e-10)

    log_prob_actions = torch.stack(log_prob_actions)
    loss = torch.sum(torch.mul(-log_prob_actions, discounted_rewards), -1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def worker(queue):
    done = queue.empty()
    while not done:
        args = queue.get()
        args_list, idx = args
        for args in args_list:
            try:
                main(*args)
            except Exception as e:
                print(e)
                continue
        done = queue.empty()
    return


def main(learning_rate, gamma, dropout, n_dim, seed=1):
    running_reward = 10
    episode = 0
    
    env = gym.make("CartPole-v1")
    env.seed(seed)
    torch.manual_seed(seed)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    policy = PGN(state_space, action_space, dropout, n_dim)
    optimizer = optim.Adam(list(policy.parameters()), lr=learning_rate)
    data_step = []
    data_reward = []
    fn = f"{learning_rate}-{gamma}-{dropout}-{n_dim}-{seed}"

    logger = initialize_logger(os.path.join("log", fn + ".log"))
    
    max_running_reward = -1
    while episode < N_EPISODES:
        frames = []
        state = env.reset()
        # disable_view_window()
        log_prob_actions = []
        rewards = []
        reward_per_episode = 0
        for step in range(N_STEPS):
            action, log_prob_action = select_action(policy, state)
            state, reward, done, _ = env.step(action.item())
            log_prob_actions.append(log_prob_action)
            rewards.append(reward)
            reward_per_episode += reward
            if done:
                break
                
        running_reward = 0.05 * reward_per_episode + (1 - 0.05) * running_reward
        update_policy(log_prob_actions, rewards, optimizer, gamma)
        data_step.append(step)
        data_reward.append(running_reward)
        if running_reward > max_running_reward:
            max_running_reward = running_reward
            torch.save(policy.state_dict(), os.path.join("save2", fn + ".pt"))
        
        if episode % 100 == 0:
            logger.info(f"Episode {episode}\tdone at step: {step:5d}\twith"
                    f"Average reward: {running_reward:.2f}")
        if running_reward > env.spec.reward_threshold:
            logger.info(f"Solved! Running reward is now {running_reward}"
                    f" and the {episode} episode runs to {step} steps!")
            torch.save(policy.state_dict(), os.path.join("save1", fn + ".pt"))
            # data = {
            #     "steps_per_episode": data_step,
            #     "rewards_per_episode": data_reward,
            # }
            # save_experiment_result(fn + ".pkl", data, "data1")
        episode += 1
    data = {
        "steps_per_episode": data_step,
        "rewards_per_episode": data_reward,
    }
    save_experiment_result(fn + ".pkl", data, "data2")


if __name__ == "__main__":
    N_EPISODES = 1000
    N_STEPS = 10000
    lrs = [5e-2, 1e-2, 5e-3]
    gms = [0.99, 0.90]
    dos = [0.0, 0.5]
    nds = [64, 128, 256]
    seeds = [0, 1, 2, 3, 4]

    make_dir("save1")
    make_dir("save2")
    make_dir("data1")
    make_dir("data2")
    make_dir("log")

    # NCPU = 20
    NCPU = int(sys.argv[1])
    data = list(product(lrs, gms, dos, nds, seeds))
    random.shuffle(data)
    queue = initialize_queue(data, 60)
    procs = initialize_proc(queue, worker, NCPU)
    for proc in procs:
        proc.join()

    # for args in product(lrs, gms, dos, nds, seeds):
    #     main(*args)

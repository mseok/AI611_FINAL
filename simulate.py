import matplotlib.pyplot as plt
from matplotlib import animation
import os

import gym
import numpy as np
import torch

from model import PGN
from train import select_action
from utils import read_data


def save_frames_as_gif(frames, path="./", filename="gym_animation.gif"):
    patch = plt.imshow(frames[0])
    plt.axis("off")
    def animate(i):
        patch.set_data(frames[i])
    if not os.path.exists(path):
        os.mkdir(path)
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=60)


def simulate(fn):
    learning_rate, gamma, dropout, n_dim, seed = fn.split(
        "/")[-1].split(".pt")[0].split("-")
    env = gym.make("CartPole-v1")
    env.seed(int(seed))
    torch.manual_seed(int(seed))
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    policy = PGN(state_space, action_space, float(dropout), int(n_dim))
    policy.load_state_dict(torch.load(fn))

    frames = []
    state = env.reset()
    for step in range(N_STEPS):
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        action, log_prob_action = select_action(policy, state)
        state, reward, done, _ = env.step(action.item())
        if done:
            break
    env.close()
    print(f"Simulation done")
    save_frames_as_gif(frames, path="./", filename=f"Best.gif")
    print(f"Save done")


if __name__ == "__main__":
    BEST = "0.01-0.9-0.0-256"
    N_STEPS = 10000
    SEEDS = [0, 1, 2, 3, 4]

    data = []
    for seed in SEEDS:
        data_fn = os.path.join("data2", BEST + "-" + str(seed) + ".pkl")
        max_data = max(read_data(data_fn)["rewards_per_episode"])
        data.append(max_data)
    idx = np.argmax(np.array(data))
    fn = os.path.join("save2", BEST + "-" + str(idx) + ".pt")
    simulate(fn)
        

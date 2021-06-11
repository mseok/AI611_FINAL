from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle

import numpy as np


def read_data(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


def plot(data):
    # fig = plt.figure(figsize=(16, 12), dpi=300)
    fig = plt.figure(figsize=(16, 12), dpi=100)
    subplots = []
    x = 4
    y = 4
    for i in range(1, 17):
        ax = fig.add_subplot(4, 4, i)
        ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
        subplots.append(ax)
    # rewards = data["rewards_per_episode"]
    # steps = data["steps_per_episode"]
    # x = np.arange(len(rewards))
    
    # plt.plot(x, rewards, color=COLORSCHEMES['123'], linewidth=LINEWIDTH)
    # plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
    plt.show()
    # plt.xlabel('# Episodes')
    # plt.ylabel('rewards')
    # plt.legend(["RoMA", "RoMA w/o adaptation", "Gaussian smoothing"], loc ="lower right", fontsize='small') 
    return


def plt_default_settings():
    plt.tight_layout()
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rcParams['xtick.major.width'] = 2.5
    plt.rcParams['xtick.minor.width'] = 2.5
    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['ytick.minor.width'] = 2.5
    plt.rcParams.update({'font.size': 14})
    mpl.rc('font', family = 'serif', serif = 'cmr10') 


def ensemble():
    total = {}
    for args in product(lrs, gms, dos, nds):
        total[args] = []
        for seed in SEEDS:
            seed_args = args + (seed, )
            seed_args = list(map(str, seed_args))
            fn = "-".join(seed_args) + ".pkl"
            fn = os.path.join("./data", fn)
            if os.path.exists(fn):
                total[args].append(seed)
    steps = {}
    for args, seeds in total.items():
        seed_steps = []
        if len(seeds) != 3:
            continue
        print(args)
        for seed in seeds:
            seed_args = args + (seed, )
            seed_args = list(map(str, seed_args))
            fn = "-".join(seed_args) + ".pkl"
            fn = os.path.join("./data", fn)
            seed_data = read_data(fn)
            seed_steps.append(len(seed_data["steps_per_episode"]))
            print(len(seed_data["rewards_per_episode"]))
            print(seed_data["rewards_per_episode"][-5:])
        steps[args] = round(sum(seed_steps) / len(seed_steps), 3)
    # print(steps)
    return


if __name__ == "__main__":
    plt_default_settings()
    COLORSCHEMES = {}
    COLORSCHEMES['123'] = (247/255, 112/255, 136/255)
    COLORSCHEMES['23'] = (51/255, 176/255, 122/255)
    COLORSCHEMES['3'] = (128/255, 150/255, 244/255)
    COLORSCHEMES['1'] = (255/255, 161/255, 0/255)
    LINEWIDTH = 2.
    # plot(None)
    
    lrs = [5e-2, 1e-2, 5e-3]
    gms = [0.99, 0.90]
    dos = [0.0, 0.5]
    nds = [64, 128, 256]
    SEEDS = [0, 1, 2]
    steps = []
    rewards = []
    ensemble()
    exit()
    for args in product(lrs, gms, dos, nds, SEEDS):
        args = list(map(str, args))
        fn = "-".join(args) + ".pkl"
        fn = os.path.join("./data", fn)
        print(fn)
        if os.path.exists(fn):
            data = read_data(fn)
            print(len(data["steps_per_episode"]))
            steps.append((fn, len(data["steps_per_episode"])))
            rewards.append((fn, max(data["steps_per_episode"])))
            # print(data["rewards_per_episode"])
        else:
            print("failed")
    steps = sorted(steps, key=lambda t: t[1])
    rewards = sorted(rewards, key=lambda t: t[1])
    print(steps)
    print(len(steps) // 3)
    print(rewards)

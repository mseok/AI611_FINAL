from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import numpy as np

from utils import read_data


def plot_avg(avg, std):
    fig, axes = plt.subplots(3, 2)
    fig.figsize = FIGSIZE
    fig.dpi = DPI
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False,
                    bottom=False, left=False, right=False)
    plt.xlabel("# EPISODES", fontsize=11)
    plt.ylabel("AVERAGE REWARDS", fontsize=11)
    keys = list(avg.keys())
    for lr_idx, lr_name in LRS.items():
        for gm_idx, gm_name in GMS.items():
            title = lr_name + "," + gm_name
            lr = float(lr_name.split(":")[-1])
            gm = float(gm_name.split(":")[-1])
            ax = axes[lr_idx][gm_idx]
            ax.set_title(title)
            ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)

            tmp_keys = [key for key in keys if lr in key and gm in key]
            tmp_dos = [key[-2] for key in tmp_keys]
            tmp_nds = [key[-1] for key in tmp_keys]
            do_names = [DOS[dos.index(do)] for do in tmp_dos]
            nd_names = [NDS[nds.index(nd)] for nd in tmp_nds]
            legends = [do_name + "," + nd_name for do_name,
                       nd_name in zip(do_names, nd_names)]
            max_values = []
            for idx, legend in enumerate(legends):
                do_name, nd_name = legend.split(",")
                do = float(do_name.split(":")[-1])
                # if do == 0.5:
                #     continue
                # if do == 0.0:
                #     continue
                nd = int(nd_name.split(":")[-1])
                avg_y = avg[(lr, gm, do, nd)]
                std_y = std[(lr, gm, do, nd)]
                max_values.append(max(avg_y))
                x = np.arange(len(avg_y))
                ax.plot(x, avg_y, color=COLORSCHEMES[idx], linewidth=LINEWIDTH)
                ax.fill_between(x, avg_y - std_y, avg_y + std_y,
                                alpha=0.1, color=COLORSCHEMES[idx])
            ax.set_xlim(xmin=0, xmax=1000)
            ax.set_ylim(ymin=0, ymax=550)
            ax.plot(x, np.array([475 for _ in range(len(x))]),
                    linestyle=':', linewidth=LINEWIDTH,
                    color=COLORSCHEMES["base"])
            ticks = np.linspace(0, int(max(max_values) + 1), 3)
            ticks = [int(tick) for tick in ticks] + [550]
            ax.set_yticks(ticks)
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # extra space for legends
    plt.subplots_adjust(hspace=0.5)  # margin between subplots
    fig.legend(legends, loc="center right", fontsize='small')
    # fig.legend(legends[:3], loc="center right", fontsize='small')
    # fig.legend(legends[3:], loc="center right", fontsize='small')
    plt.savefig("average_rewards.pdf")
    # plt.savefig("average_rewards_do00.pdf")
    # plt.savefig("average_rewards_do05.pdf")
    # plt.show()
    return


def plt_default_settings():
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams.update({'font.size': 7})
    mpl.rc('font', family='serif', serif='cmr10')


def get_average_rewards():
    avg = {}
    std = {}
    high_count = {}
    first_step = {}
    for args in product(lrs, gms, dos, nds):
        args_rewards = []
        for seed in SEEDS:
            seed_args = args + (seed, )
            seed_args = list(map(str, seed_args))
            fn = "-".join(seed_args) + ".pkl"
            fn = os.path.join("./data2", fn)
            data = read_data(fn)
            seed_rewards = np.array(data["rewards_per_episode"])
            args_rewards.append(seed_rewards)
        args_rewards = np.stack(args_rewards)
        avg[args] = np.average(args_rewards, axis=0)
        std[args] = np.std(args_rewards, axis=0)
        high_count[args] = np.stack(
            np.where(avg[args][STEP_THRESHOLD2:]>REWARD_THRESHOLD)).shape[-1]
        if args_rewards.shape[0] != 5:
            continue
        steps = np.where(args_rewards > REWARD_THRESHOLD, 1, 0)
        steps = np.argmax(steps, axis=1)
        first_step[args] = round(np.average(steps)) + 1
    return avg, std, high_count, first_step


def write_result(high_count, first_step, fn):
    variables = ["Learning rate", "Gamma", "Dropout",
                 "Hidden dim", "First episode", "N episodes"]
    variables = [variable.rjust(13) for variable in variables]
    with open(fn, "w") as w:
        w.write("\t".join(variables) + "\n")
        for args, count in high_count.items():
            if count == 0:
                continue
            if first_step[args] > STEP_THRESHOLD1:
                continue
            args += (first_step[args], count)
            args = list(map(str, args))
            args = [arg.rjust(13) for arg in args]
            w.write("\t".join(args) + "\n")


if __name__ == "__main__":
    """
    'dreamer': '#2f9e44', # green8
    'dreamer + re3': '#5c940d',
    'drq': '#f76707', # orange7
    'rad': '#495057', # gray7
    'rad + re3': '#fa5252', # red6
    'rad + imagenet': '#da77f2', # grape4
    'rad + rnd': '#0c8599', # Dark cyan
    'rad + icm': '#5c940d', # lime
    'rad + contrastive': '#339af0', # blue5
    'rad + inverse': '#82c91e', # lime 6
    'rad + contrastive (pre)': '#364fc7', # indigo
    'rad + inverse (pre)': '#087f5b', # teal
    'rad + re3 (pre)': '#e03131', # red8
    'rad + atc': '#5f3dc4' # violet 9
    """
    COLORSCHEMES = {}
    COLORSCHEMES[0] = "#5f3dc4"  # violet9
    COLORSCHEMES[1] = "#364fc7"  # indigo
    COLORSCHEMES[2] = "#e03131"  # red8
    COLORSCHEMES[3] = "#f76707"  # orange7
    COLORSCHEMES[4] = "#82c91e"  # lime6
    COLORSCHEMES[5] = "#087f5b"  # teal
    COLORSCHEMES["base"] = "#495057"  # gray7
    FIGSIZE = (8, 6)
    DPI = 200
    LINEWIDTH = 1.5

    STEP_THRESHOLD1 = 300
    STEP_THRESHOLD2 = 600
    REWARD_THRESHOLD = 475

    LRS = {0: "LR:5e-2", 1: "LR:1e-2", 2: "LR:5e-3"}
    GMS = {0: "GAMMA:0.99", 1: "GAMMA:0.90"}
    DOS = {0: "DO:0.0", 1: "DO:0.5"}
    NDS = {0: "DIM:64", 1: "DIM:128", 2: "DIM:256"}
    lrs = [5e-2, 1e-2, 5e-3]
    gms = [0.99, 0.90]
    dos = [0.0, 0.5]
    nds = [64, 128, 256]
    SEEDS = [0, 1, 2, 3, 4]

    avg, std, high_count, first_step = get_average_rewards()

    plt_default_settings()
    plot_avg(avg, std)
    write_result(high_count, first_step, "result.txt")

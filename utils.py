import logging
from logging.handlers import RotatingFileHandler
import os
import pickle
import time

from torch.multiprocessing import Process, Queue


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def save_experiment_result(fn, data, dir_name="data"):
    make_dir(dir_name)
    os.chdir(dir_name)
    with open(fn, "wb") as w:
        pickle.dump(data, w)
    os.chdir("..")


def initialize_queue(data, NTASKS, shared_dict=None):
    queue = Queue()
    length = len(data)
    for i in range(NTASKS):
        length_per_proc = length // NTASKS
        st = i * length_per_proc
        end = (i + 1) * length_per_proc if i != NTASKS - 1 else length
        data_per_proc = data[st:end]
        if shared_dict is not None:
            args = (data_per_proc, i, shared_dict)
        else:
            args = (data_per_proc, i)
        queue.put(args)
    return queue


def initialize_proc(queue, fn, NCPU=4):
    procs = []
    for _ in range(NCPU):
        proc = Process(target=fn, args=(queue,))
        procs.append(proc)
        proc.start()
        time.sleep(0.5)
    return procs


def initialize_logger(log_file=None, log_file_level=logging.NOTSET, rotate=False):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != "":
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

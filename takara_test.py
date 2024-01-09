
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from phc.utils.flags import flags

import numpy as np
import copy
import torch
import wandb
from phc.env.tasks.humanoid import Humanoid



if __name__ == '__main__':

    # print("Hello, world!")
    env = Humanoid()


    # env.reset()
    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample())
    # env.close()


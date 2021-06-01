## Dataloader definitions and other util functions for neural network training/evaluation
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTITASK_ENDPOINTS = {"energy": 0, "homo": 1, "lumo": 2, "gap": 3, "dipole": 4}


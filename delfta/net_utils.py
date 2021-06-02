## Dataloader definitions and other util functions for neural network training/evaluation
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTITASK_ENDPOINTS = {"energy", "homo", "lumo", "gap", "dipole"}


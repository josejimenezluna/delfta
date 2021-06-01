## Scripts to download training h5 files and trained models from polybox or other service.
import os

import h5py
import requests
import torch

from delfta.net_utils import DEVICE
from delfta.utils import DATA_PATH, MODEL_PATH

DATASET_PATH = {"qmugs": os.path.join(DATA_PATH, "qmugs.h5")}

DATASET_REMOTE = {
    "qmugs": "polybox.ethz.ch/...",
}


MODELS = {
    "multitask": os.path.join(MODEL_PATH, "multitask.pt"),
    "single_energy": os.path.join(MODEL_PATH, "single_energy.pt"),
    "charges": os.path.join(MODEL_PATH, "charges.pt"),
}

MODELS_REMOTE = {
    "multitask": "polybox.ethz.ch/...",
    "single_energy": "polybox.ethz.ch/...",
    "charges": "polybox.ethz.ch/...",
}



def download(name=""):
    # Return True upon successful download, False otherwise
    # 
    # requests.get(...)
    pass


def get_dataset(name="qmugs"):
    if name not in DATASET_PATH:
        raise ValueError("Dataset not supported")
    else:
        downloaded_flag = True
        if not os.path.exists(DATASET_PATH[name]):
            os.makedirs(DATA_PATH, exist_ok=True)
            downloaded_flag = download(name)

        if downloaded_flag:
            h5 = h5py.File(DATASET_PATH[name], "r")
            return h5
        else:
            raise ValueError("Failed at downloading dataset!")


def get_model_weights(name="multitask"):
    if name not in MODELS:
        raise ValueError("Model not supported")
    else:
        downloaded_flag = True
        if not os.path.exists(MODELS[name]):
            os.makedirs(MODEL_PATH, exist_ok=True)
            downloaded_flag = download(name, dict_lookup=MODEL_PATH, dict_remote=MODELS_REMOTE)

        if downloaded_flag:
            weights = torch.load(MODEL_PATH[name], map_location=DEVICE)
            return weights
        else:
            raise ValueError("Failed at downloading model!")

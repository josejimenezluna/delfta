import os
import tarfile

import h5py
import requests
import torch
from tqdm import tqdm

from delfta.net_utils import DEVICE
from delfta.utils import DATA_PATH, MODEL_PATH, XTB_PATH

DATASETS = {"qmugs": os.path.join(DATA_PATH, "qmugs.h5")}

# Load 100k datasets (train: 100k, eval: 20k, test: 20k). Final sets to be added in the end. 
DATASET_REMOTE = {
    "qmugs_train": "https://polybox.ethz.ch/index.php/s/NucWaxLPFDGc0DH/download",
    "qmugs_eval": "https://polybox.ethz.ch/index.php/s/tLvYetVUSfsuwM5/download",
    "qmugs_test": "https://polybox.ethz.ch/index.php/s/feOh2Unmwfyeq5N/download",
}

MODELS = {
    "multitask": os.path.join(MODEL_PATH, "multitask.pt"),
    "single_energy": os.path.join(MODEL_PATH, "single_energy.pt"),
    "charges": os.path.join(MODEL_PATH, "charges.pt"),
}

# Load models trained on 100k. Final sets to be added in the end. 
MODELS_REMOTE = {
    "multitask_delta": "https://polybox.ethz.ch/index.php/s/YcKUyHnUXup9vin/download",
    "single_energy_delta": "https://polybox.ethz.ch/index.php/s/2nIjp7xUJejiYhh/download",
    "charges_delta": "https://polybox.ethz.ch/index.php/s/5K1Q5Rx0zBphIHW/download",
    "multitask_direct": "https://polybox.ethz.ch/index.php/s/YUUfc4wo0GdSdsu/download",
    "single_energy_direct": "https://polybox.ethz.ch/index.php/s/51RBM0Bm4FvycPE/download",
    "charges_direct": "https://polybox.ethz.ch/index.php/s/65c9FZQ8V7Egnnd/download",
}

XTB_REMOTE = (
    "https://github.com/grimme-lab/xtb/releases/download/v6.3.1/xtb-200615.tar.xz"
)

TESTS_REMOTE = "https://polybox.ethz.ch/index.php/s/Lyn7OOnh9F7NIIc/download"


def download(src, dest):
    """Simple requests.get with a progress bar

    Parameters
    ----------
    src : str
        Remote path to be downloaded
    dest : str
        Local path for the download

    Returns
    -------
    None
    """
    r = requests.get(src, stream=True)
    tsize = int(r.headers.get('content-length', 0))
    progress = tqdm(total=tsize, unit='iB', unit_scale=True)

    with open(dest, "wb") as handle:
        for chunk in r.iter_content(chunk_size=256):
            progress.update(len(chunk))
            handle.write(chunk)


def get_dataset(name="qmugs"):
    """Returns a h5py dataset with a specific `name`. These are
    checked in the `DATASETS` global variable.

    Parameters
    ----------
    name : str, optional
        Name of the h5py dataset to be returned, by default "qmugs"

    Returns
    -------
    [h5py.File]
        h5py file handle of the requested dataset
    """
    if name not in DATASETS:
        raise ValueError("Dataset not supported")
    else:
        downloaded_flag = True
        if not os.path.exists(DATASETS[name]):
            os.makedirs(DATA_PATH, exist_ok=True)
            downloaded_flag = download(name)

        if downloaded_flag:
            h5 = h5py.File(DATASETS[name], "r")
            return h5
        else:
            raise ValueError("Failed at downloading dataset!")


def get_model_weights(name="multitask"):
    """Returns a torch.load handle for a model with a specific `name`.
    These are checked in the `MODELS` global variable.

    Parameters
    ----------
    name : str, optional
        Name of the model weights to be returned, by default "multitask"

    Returns
    -------
    [torch.weights]
        Trained weights for the requested model
    """
    if name not in MODELS:
        raise ValueError("Model not supported")
    else:
        downloaded_flag = True
        if not os.path.exists(MODELS[name]):
            os.makedirs(MODEL_PATH, exist_ok=True)
            downloaded_flag = download(
                name, dict_lookup=MODEL_PATH, dict_remote=MODELS_REMOTE
            )

        if downloaded_flag:
            weights = torch.load(MODEL_PATH[name], map_location=DEVICE)
            return weights
        else:
            raise ValueError("Failed at downloading model!")


if __name__ == "__main__":
    # Trained models
    os.makedirs(MODEL_PATH, exist_ok=True)
    for model_name, model_path in MODELS:
        download(MODELS_REMOTE[model_name], model_path)

    # Training data
    os.makedirs(DATA_PATH, exist_ok=True)
    for data_name, data_path in DATASETS:
        download(DATASET_REMOTE[model_name], data_path)

    # xtb binary
    os.makedirs(XTB_PATH, exist_ok=True)
    xtb_tar = os.path.join(XTB_PATH, "xtb.tar.xz")
    download(XTB_REMOTE, xtb_tar)

    with tarfile.open(xtb_tar) as handle:
        handle.extractall(XTB_PATH)

    # tests
    tests_tar = os.path.join(DATA_PATH, "tests.tar.gz")
    download(TESTS_REMOTE, tests_tar)

    with tarfile.open(tests_tar) as handle:
        handle.extractall(DATA_PATH)
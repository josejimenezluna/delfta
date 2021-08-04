import argparse
import os
import tarfile

import h5py
import requests
import torch
from tqdm import tqdm

from delfta.net_utils import DEVICE
from delfta.utils import DATA_PATH, LOGGER, MODEL_PATH, ROOT_PATH

DATASETS = {
    "qmugs_train": os.path.join(DATA_PATH, "qmugs", "qmugs_train.h5"),
    "qmugs_eval": os.path.join(DATA_PATH, "qmugs", "qmugs_eval.h5"),
    "qmugs_test": os.path.join(DATA_PATH, "qmugs", "qmugs_test.h5"),
}  # TODO update dataset structure

# TODO Load 100k datasets (train: 100k, eval: 20k, test: 20k). Final sets to be added in the end.
DATASET_REMOTE = "https://polybox.ethz.ch/index.php/s/BpkfVEgJWjoRRnN/download"

MODELS = {
    "multitask_delta": os.path.join(MODEL_PATH, "multitask_delta.pt"),
    "single_energy_delta": os.path.join(MODEL_PATH, "single_energy_delta.pt"),
    "charges_delta": os.path.join(MODEL_PATH, "charges_delta.pt"),
    "multitask_direct": os.path.join(MODEL_PATH, "multitask_direct.pt"),
    "single_energy_direct": os.path.join(MODEL_PATH, "single_energy_direct.pt"),
    "charges_direct": os.path.join(MODEL_PATH, "charges_direct.pt"),
}

# TODO Load models trained on 100k. Final sets to be added in the end.
MODELS_REMOTE = "https://polybox.ethz.ch/index.php/s/sJyP4lpSZJKOTaa/download"

UTILS_REMOTE = "https://polybox.ethz.ch/index.php/s/fNAsmn1JBnNyCUY/download"

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
    tsize = int(r.headers.get("content-length", 0))
    progress = tqdm(total=tsize, unit="iB", unit_scale=True, position=0, leave=False)

    with open(dest, "wb") as handle:
        progress.set_description(os.path.basename(dest))
        for chunk in r.iter_content(chunk_size=1024):
            handle.write(chunk)
            progress.update(len(chunk))


def get_dataset(name):
    """Returns a h5py dataset with a specific `name`. These are
    checked in the `DATASETS` global variable.

    Parameters
    ----------
    name : str, optional
        Name of the h5py dataset to be returned.

    Returns
    -------
    h5py.File
        h5py file handle of the requested dataset
    """
    if name not in DATASETS:
        raise ValueError("Dataset not supported")
    else:
        h5 = h5py.File(DATASETS[name], "r")
        return h5


def get_model_weights(name):
    """Returns a torch.load handle for a model with a specific `name`.
    These are checked in the `MODELS` global variable.

    Parameters
    ----------
    name : str, optional
        Name of the model weights to be returned

    Returns
    -------
    torch.weights
        Trained weights for the requested model
    """
    weights = torch.load(MODELS[name], map_location=DEVICE)
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General download script")
    parser.add_argument(
        "--training", dest="training", action="store_true", default=False
    )
    args = parser.parse_args()

    os.makedirs(DATA_PATH, exist_ok=True)

    # Trained models
    LOGGER.info("Now downloading trained models...")
    models_tar = os.path.join(ROOT_PATH, "models.tar.gz")
    download(MODELS_REMOTE, models_tar)

    with tarfile.open(models_tar) as handle:
        handle.extractall(ROOT_PATH)

    if args.training:
        # Training data
        LOGGER.info("Now downloading training data...")

        download(DATASET_REMOTE, os.path.join(DATA_PATH, "qmugs.tar.gz"))

        with tarfile.open(os.path.join(DATA_PATH, "qmugs.tar.gz")) as handle:
            handle.extractall(DATA_PATH)

    # tests
    LOGGER.info("Downloading tests...")
    tests_tar = os.path.join(DATA_PATH, "test_data.tar.gz")
    download(TESTS_REMOTE, tests_tar)

    with tarfile.open(tests_tar) as handle:
        handle.extractall(DATA_PATH)

    # utils
    LOGGER.info("Downloading utils...")
    utils_tar = os.path.join(ROOT_PATH, "utils.tar.gz")
    download(UTILS_REMOTE, utils_tar)

    with tarfile.open(utils_tar) as handle:
        handle.extractall(ROOT_PATH)

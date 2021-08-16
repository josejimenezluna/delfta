import os

import requests
from delfta.download import (
    DATASET_REMOTE,
    MODELS,
    MODELS_REMOTE,
    TESTS_REMOTE,
    UTILS_REMOTE,
    get_model_weights,
)
from delfta.utils import MODEL_PATH

_remotes = [TESTS_REMOTE, UTILS_REMOTE, MODELS_REMOTE, DATASET_REMOTE]


def test_islinkup():
    for remote in _remotes:
        r = requests.head(remote)
        assert r.ok


def test_get_model_weights():
    for name in MODELS:
        _ = get_model_weights(os.path.join(MODEL_PATH, f"{name}.pt"))


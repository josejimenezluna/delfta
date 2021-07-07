import requests
from delfta.download import (
    DATASETS,
    DATASET_REMOTE,
    MODELS,
    MODELS_REMOTE,
    TESTS_REMOTE,
    UTILS_REMOTE,
    get_dataset,
    get_model_weights,
)

_remotes = [TESTS_REMOTE, UTILS_REMOTE, MODELS_REMOTE, DATASET_REMOTE]


def test_islinkup():
    for remote in _remotes:
        r = requests.head(remote)
        assert r.ok


def test_get_model_weights():
    for name in MODELS.keys():
        _ = get_model_weights(name)


def test_get_dataset():
    for name in DATASETS.keys():
        _ = get_dataset(name)

import requests
from delfta.download import TESTS_REMOTE, UTILS_REMOTE, MODELS_REMOTE, DATASET_REMOTE

_remotes = [TESTS_REMOTE, UTILS_REMOTE, MODELS_REMOTE, DATASET_REMOTE]

def test_islinkup():
    for remote in _remotes:
        r = requests.head(remote)
        assert r.ok

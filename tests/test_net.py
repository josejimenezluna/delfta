from delfta.net import EGNN
from delfta.net_utils import MODEL_HPARAMS, DelftaDataset
from openbabel.pybel import readstring
from torch_geometric.data.dataloader import DataLoader


def test_net_outshape():
    mol = readstring("smi", "CCO")
    data = DelftaDataset([mol])
    loader = DataLoader(data, batch_size=1, shuffle=False)
    batch = next(iter(loader))

    for param in MODEL_HPARAMS.values():
        model = EGNN(n_outputs=param.n_outputs, global_prop=param.global_prop).eval()
        out = model(batch)
        if param.global_prop:
            assert out.shape[1] == param.n_outputs
        else:
            assert len(out) == len(mol.atoms)

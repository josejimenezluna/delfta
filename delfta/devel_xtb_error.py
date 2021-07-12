import numpy as np
from delfta.calculator import DelftaCalculator
from openbabel.pybel import readfile, readstring
from openbabel.pybel import Molecule
# needs some more rigorous testing; and remove TODO s in calculator.xtb
import random 

smiles = ["C", "C", "CCC", "B", "CCC", "CCC", "C", "B"]
for _ in range(10): 
    random.shuffle(smiles)
    print(smiles)
    # "C" works, "CCC" and "B" do not (dummy debug)
    nan_idx_expected = [i for i, smi in enumerate(smiles) if smi!="C"]
    mols = [readstring("smi", smi) for smi in smiles] 
    calc = DelftaCalculator(tasks="all", delta=True, force3d=True, addh=True)
    preds_delta = calc.predict(mols, batch_size=1)
    nan_idx_seen = np.where(np.isnan(preds_delta["E_form"]))[0].tolist()
    assert nan_idx_expected == nan_idx_seen
a = 2
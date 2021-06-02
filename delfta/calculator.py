## classes and functions for the main predictive capabilities
import numpy as np
from torch._C import dtype
from delfta.net_utils import MULTITASK_ENDPOINTS
from delfta.download import MODELS, get_model_weights
from delfta.xtb import get_xtb_props


class DelftaCalculator:
    def __init__(self, tasks, accurate_energy=False) -> None:
        self.tasks = tasks
        self.models = []

        if accurate_energy:
            self.models.append("single_energy")

        for task in tasks:
            if task in MULTITASK_ENDPOINTS:
                self.models.append("multitask")

            if task == "charges":
                self.models.append("charges")

        if self.models:
            all_models = [MULTITASK_ENDPOINTS] + ["charges", "single_energy"]
            raise ValueError(
                f"None of the provided tasks were recognized, please provide one or several from {all_models}"
            )

        pass
    
    def _preprocess(self, mols):
        # make sure that pybel mols contain all the information that is needed
        # (i.e. 3D coordinates and whatnot). Otherwise compute it.
        return mols

    def _get_preds(loader, model):
        y_hats = []

        for batch in loader:
            y_hats.append(model(batch).numpy())
        return y_hats
            
    def predict(self, mols, batch_size=32):
        xtb_props = []

        # Data featurization code for the network goes here
        # data = DeltaData(mols)
        # loader = DataLoader(data, batch_size=batch_size)
        # ...

        for mol in mols:
            xtb_out = get_xtb_props(mol)
            xtb_props.append([xtb_out[task] for task in self.tasks])

        preds = []

        for idx_model, model_name in enumerate(self.models):
            if model_name == "multitask":
                # model = MultitaskArchitecture(...)
                pass
            elif model_name == "charges":
                # model = NodeLevelArchitecture(...)
                pass
            else:
                # model = SingleTaskArchitecture(...)
                pass

            weights = get_model_weights(model_name)
            model.load_state_dict(weights)
            delta_y = self._get_preds(loader, model)

            if model_name != "charges":
                delta_y = np.vstack(delta_y)
                pred = np.array([xtb[idx_model] for xtb in xtb_props], dtype=np.float32) + delta_y
                #TODO: check whether this difference direction is correct 
                preds.append(pred)
            else:
                #TODO: handle differently-sized arrays. Probably has to be done manually.
                # pred = 
                preds.append(pred)
                pass 
        return preds


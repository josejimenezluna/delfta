## classes and functions for the main predictive capabilities
from delfta.net_utils import MULTITASK_ENDPOINTS
from delfta.download import MODELS


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

    def predict(mols):
        pass


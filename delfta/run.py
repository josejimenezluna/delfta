import argparse
from delfta.utils import preds_to_lists
import os
import json
from delfta.utils import preds_to_lists, column_order
import pandas as pd


from openbabel.pybel import readfile

from delfta.calculator import DelftaCalculator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol_file")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--json", type=bool, default=True)
    parser.add_argument("--csv", type=bool, default=False)
    parser.add_argument("--outfile", type=str, default="_default_")
    parser.add_argument("--delta", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force3d", type=bool, default=True)
    parser.add_argument("--addh", type=bool, default=True)
    parser.add_argument("--xtbopt", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--progress", type=bool, default=True)
    parser.add_argument("--sanity_checks", type=bool, default=True)
    args = parser.parse_args()

    """Bash-API for run the DelftaCalculator.

        Parameters
        ----------
        mol_file : str
            Path to molecule file, readable with Openbabel.
        --tasks:  list, optional
            A list of tasks to predict. Available tasks include `[E_form, E_homo, E_lumo, E_gap, dipole, charges]`, by default ["all"].
        --json: bool, optional
            Whether to write the results as json, by default True
        --csv: bool, optional
            Whether to write the results as csv, by default False
        --outfile: str, optional
            Path to output filename. The file extension will be generated according to --json/--csv option choice. 
            Defaults to `delfta_predictions.json`/`delfta_predictions.csv` in the directory of `mol_file`.
        --delta : bool, optional
            Whether to use delta-learning models, by default True
        --force3d : bool, optional
            Whether to assign 3D coordinates to molecules lacking them, by default False
        --addh : bool, optional
            Whether to add hydrogens to molecules lacking them, by default False
        --xtbopt : bool, optional
            Whether to perform GFN2-xTB geometry optimization, by default False
        --verbose : bool, optional
            Enables/disables stdout logger, by default True
        --progress : bool, optional
            Enables/disables progress bars in prediction, by default True
        --sanity_checks : bool, optional
            Enables/disables sanity checks before prediction, including
            atom type validation, charge neutrality, hydrogen addition and
            3D conformer generation, by default True
        """

    outfile = (
        os.path.join(os.path.dirname(args.mol_file), "delfta_predictions")
        if args.outfile == "_default_"
        else args.outfile
    )
    ext = os.path.splitext(args.mol_file)[1].lstrip(".")
    reader = readfile(ext, args.mol_file)
    calc = DelftaCalculator(
        tasks=args.tasks,
        delta=args.delta,
        force3d=args.force3d,
        addh=args.addh,
        xtbopt=args.xtbopt,
        verbose=args.verbose,
        progress=args.progress,
        sanity_checks=args.sanity_checks,
    )
    preds = calc.predict(reader, batch_size=args.batch_size)

    preds_list = preds_to_lists(preds)
    if args.json:
        with open(outfile + ".json", "w", encoding="utf-8") as f:
            json.dump(preds_list, f, ensure_ascii=False, indent=4)
    if args.csv:
        df = pd.DataFrame(preds)
        df = df[sorted(df.columns.tolist(), key=lambda x: column_order[x])]
        df.to_csv(outfile + ".csv")

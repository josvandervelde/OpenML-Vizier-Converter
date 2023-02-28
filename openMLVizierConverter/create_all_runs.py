"""
As a test, create a study for every run and throw an error if the resulting study is not correct
"""


import logging
from pathlib import Path

import pandas as pd
import tqdm
import vizier.pyvizier as vz

from converter import create_study_from_openml

Study = vz.ProblemAndTrials  # Alias, similarly used in OptFormers repo

DIR_ROOT = Path(__file__).resolve().parents[1]
DIR_SUCCESS = DIR_ROOT / "result" / "success"
DIR_ERROR = DIR_ROOT / "result" / "error"


def write_study_or_error_to_file(id_):
    try:
        study = create_study_from_openml(id_)
        with open(DIR_SUCCESS / str(id_), "w") as f:
            # Successfully created a Study? Write the JSON to the success folder
            f.write(str(study))
    except Exception as e:
        logging.warning(f"Error while processing OpenML run with {id_=}")
        # On any exception, log the error message to the error folder
        with open(DIR_ERROR / str(id_), "w") as f:
            f.write(str(e))


def create_all():
    logging.basicConfig(level=logging.INFO)

    for dir_ in (DIR_SUCCESS, DIR_ERROR):
        dir_.mkdir(parents=True, exist_ok=True)

    df_ids = pd.read_csv(DIR_ROOT / "data" / "runids.csv")
    existing_success = {int(p.name) for p in DIR_SUCCESS.iterdir()}
    existing_error = {int(p.name) for p in DIR_ERROR.iterdir()}
    ids = set(df_ids["id"].unique()) - existing_error - existing_success

    logging.info(
        f"Will create studies for {len(ids)} / {len(df_ids)} runs.\n"
        f"{len(existing_success)} ignored because they are created already (in "
        f"{DIR_SUCCESS});\n{len(existing_error)} ignored because they were already tried "
        f"(see {DIR_ERROR})"
    )

    for id_ in tqdm.tqdm(ids):
        write_study_or_error_to_file(id_)
    logging.info("Done")


if __name__ == "__main__":
    create_all()

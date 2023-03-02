"""
As a test, create a study for every run and throw an error if the resulting study is not correct
"""
import json
import logging
import traceback
from pathlib import Path

import pandas as pd
import tqdm
import vizier.pyvizier as vz

from converter import create_study_from_openml

Study = vz.ProblemAndTrials  # Alias, similarly used in OptFormers repo

DIR_ROOT = Path(__file__).resolve().parents[1]
DIR_SUCCESS = DIR_ROOT / "result" / "success"
DIR_STATS = DIR_ROOT / "result" / "stats"
DIR_ERROR = DIR_ROOT / "result" / "error"


def write_study_or_error_to_file(id_):
    try:
        study = create_study_from_openml(id_)
        with open(DIR_SUCCESS / str(id_), "w") as f:
            # Successfully created a Study? Write the JSON to the success folder
            f.write(str(study))
        stats = get_statistics(study)
        with open(DIR_STATS / str(id_), "w") as f:
            json.dump(stats, f)
    except Exception as e:
        traceback.print_exc()
        logging.warning(f"Error while processing OpenML run with {id_=}")
        # On any exception, log the error message to the error folder
        with open(DIR_ERROR / str(id_), "w") as f:
            f.write(str(e))


def get_statistics(study: Study) -> dict:
    stats = {
        "trial_results": [t.final_measurement.metrics.get("objective").value for t in study.trials],
        "user": study.problem.metadata.get("uploader_id"),
        "dataset": study.problem.metadata.get("dataset_id"),
        "task": study.problem.metadata.get("task_id"),
        "flow": study.problem.metadata.get("flow_id"),
        "run": study.problem.metadata.get("run_id"),
        "parameters": [
            {
                "name": p.name,
                "type": p.type.name,
                "bounds": p.bounds if p.type != vz.ParameterType.CATEGORICAL else None,
                "scale_type": p.scale_type.name if p.scale_type is not None else None,
                "feasible_values": p.feasible_values
                if p.num_feasible_values != float("inf")
                else [],
            }
            for p in study.problem.search_space.parameters
        ],
    }

    return stats


def create_all():
    logging.basicConfig(level=logging.INFO)

    for dir_ in (DIR_SUCCESS, DIR_ERROR, DIR_STATS):
        dir_.mkdir(parents=True, exist_ok=True)

    df_ids = pd.read_csv(DIR_ROOT / "data" / "runids.csv")
    existing_success = {int(p.name) for p in DIR_SUCCESS.iterdir()}
    existing_error = {int(p.name) for p in DIR_ERROR.iterdir()}
    ids = sorted(set(df_ids["id"].unique()) - existing_error - existing_success)

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

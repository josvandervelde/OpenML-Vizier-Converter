"""
Convert an OpenML run into a Google Vizier study

This is the useful stuff for Google.
"""


import openml
from retry import retry
from vizier import pyvizier as vz

from search_space import determine_search_space

Study = vz.ProblemAndTrials  # Alias, similarly used in OptFormers repo

OBJECTIVE_NAME = "objective"


def create_study_from_openml(run_id: int = 10437679) -> Study:
    """
    This is the part of the code that should be used by Google.
    Run this for every run_id (see data/runids.csv) and you get a study.
    For 44 runs, an error is thrown. These runs can be ignored.
    """
    run = _get_openml_run(run_id)
    run_trace = _get_openml_traces(run_id)
    if len(run_trace.trace_iterations) < 2:
        raise ValueError("Expected at least 2 optimization traces")
    for trace in run_trace.trace_iterations.values():
        try:
            params = trace.get_parameters()
        except:
            raise ValueError("OpenML error on trace.get_parameters()")
        if len(params) == 0:
            raise ValueError("Expected at least 1 parameter")
        if trace.evaluation <= 0.0:
            raise ValueError("Expected field 'evaluation' to be present, larger than zero")

    goal = vz.ObjectiveMetricGoal.MAXIMIZE
    metric_information = vz.MetricsConfig([vz.MetricInformation(name=OBJECTIVE_NAME, goal=goal)])
    # Adding some metadata that might be interesting. "name" is the only field that is actually used
    # in the OptFormer
    metadata = vz.Metadata(
        {
            "name": f"Dataset {run.dataset_id}; Task {run.task_id} (type {run.task_type}); flow: "
            f"{run.flow_id}. Run {run.id}",
            "dataset_id": run.dataset_id,
            "task_id": run.task_id,
            "task_type": run.task_type,
            "flow_id": run.flow_id,
            "flow": run.flow_name,
            "run_id": run.id,
        }
    )

    search_space, parameter_convertors = determine_search_space(run, run_trace)

    problem = vz.ProblemStatement(
        search_space=search_space, metric_information=metric_information, metadata=metadata
    )
    trials = [
        vz.Trial(parameters=_convert_params(trace.get_parameters(), parameter_convertors)).complete(
            vz.Measurement({OBJECTIVE_NAME: trace.evaluation})
        )
        for trace in run_trace.trace_iterations.values()
    ]
    study = Study(problem=problem, trials=trials)
    _raise_error_if_inconsistent(study)
    return study


@retry(tries=2, delay=1)
def _get_openml_run(run_id):
    return openml.runs.get_run(run_id)


@retry(tries=2, delay=1)
def _get_openml_traces(run_id):
    return openml.runs.get_run_trace(run_id)


def _raise_error_if_inconsistent(study: Study):
    """
    Check if the study.problem.search_space is consistent with the study.trials
    """
    parameter_names = set(study.problem.search_space.parameter_names)
    feasible_values_dict = {
        p.name: _feasible_values_or_bounds(p) for p in study.problem.search_space.parameters
    }

    for trial in study.trials:
        parameter_names_trial = set(trial.parameters)
        if parameter_names != parameter_names_trial:
            only_in_search_space = parameter_names - parameter_names_trial
            only_in_trial = parameter_names_trial - parameter_names
            raise ValueError(
                f"Parameter names of search space and trial not the same. Only in "
                f"trial: {only_in_trial}. "
                f"Only in search space: {only_in_search_space}"
            )
        for parameter_name, feasible_values in feasible_values_dict.items():
            trial_value = trial.parameters[parameter_name].value
            if isinstance(feasible_values, tuple):
                # double parameter has no Feasible values set, but has bounds
                if trial_value < feasible_values[0] or trial_value > feasible_values[1]:
                    raise ValueError(
                        f"For {parameter_name=}, the {trial_value=} is not within "
                        f"the bounds {feasible_values[0]}, {feasible_values[1]}"
                    )
            else:
                if trial_value not in feasible_values:
                    raise ValueError(
                        f"For {parameter_name=}, the {trial_value=} "
                        f"is not in the {feasible_values=}"
                    )


def _feasible_values_or_bounds(parameter_config: vz.ParameterConfig):
    if parameter_config.type in (vz.ParameterType.INTEGER, vz.ParameterType.DOUBLE):
        return parameter_config.bounds
    if all(v in {"True", "False"} for v in parameter_config.feasible_values):
        return {True, False}
    return parameter_config.feasible_values


def _convert_params(params: dict, param_convertors: dict) -> dict:
    """For each [param_name, function] in the param_convertors, apply it to the parameters"""
    if param_convertors is None or len(param_convertors) == 0:
        return params
    return {
        name: value if name not in param_convertors else param_convertors[name](value)
        for name, value in params.items()
    }

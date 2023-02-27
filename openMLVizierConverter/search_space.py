"""
Find the Vizier Search Space given an OpenML run.
"""


import json
import numpy as np
import shlex

import pandas as pd
from openml import OpenMLRun
from openml.runs import OpenMLRunTrace
from scipy.stats import stats
from vizier import pyvizier as vz


def determine_search_space(run: OpenMLRun, run_trace: OpenMLRunTrace) -> tuple[vz.SearchSpace, dict]:
    """Find the Vizier Search Space given the OpenML run"""
    if run.setup_string is not None and run.setup_string.startswith("weka"):
        return determine_search_space_weka(run)
    else:
        return determine_search_space_sklearn(run, run_trace), {}


def determine_search_space_sklearn(run: OpenMLRun, run_trace: OpenMLRunTrace) -> vz.SearchSpace:
    """Create Vizier Search Space given an OpenML Sklearn run"""
    param_settings = sklearn_possible_param_values(run, run_trace)

    search_space = vz.SearchSpace()
    root = search_space.root
    for param_name, param_possible_values in param_settings.items():
        types = set(type(v) for v in param_possible_values)
        if types == {int, float}:
            types = {float}
        elif len(types) > 1:
            raise ValueError(f"Multiple types in a single parameter: {types}")
        type_, = types

        if type_ in (int, np.int64, np.int32, float, np.float64, np.float32):
            add_number_param(root, param_possible_values, param_name)
        elif type_ == bool:
            root.add_bool_param(name=param_name)
        elif type_ == str:
            root.add_categorical_param(name=param_name, feasible_values=param_possible_values)
        else:
            raise ValueError(f"Unknown type: {type_}")
    return search_space


def sklearn_possible_param_values(run, run_trace) -> dict[str, set]:
    """Return a dictionary containing for each parameter name a set of possible values"""
    if run.parameter_settings is None or len(run.parameter_settings) == 0:
        raise ValueError("Parameter settings not found")
    else:
        if any(ps['oml:name'] == 'param_grid' for ps in run.parameter_settings):
            param_settings, = [json.loads(ps['oml:value'].replace("'", "\""))
                               for ps in run.parameter_settings if ps['oml:name'] == 'param_grid']
        elif any(ps['oml:name'] == 'param_distributions' for ps in run.parameter_settings):
            param_settings, = [ps['oml:value'] for ps in run.parameter_settings if
                               ps['oml:name'] == 'param_distributions']
            try:
                param_settings = json.loads(param_settings)
            except:
                param_settings = {}
                # param_distribution is incorrect json (probably only first 1024 "
                # characters of the string, which makes for incorrect json)")
        else:
            raise ValueError(f"parameters not found for sklearn.")

    df_actual = pd.DataFrame([t.get_parameters() for t in run_trace.trace_iterations.values()])

    for param_name in df_actual:
        if param_name not in param_settings \
                or 'serialized_object' in str(param_settings[param_name]):
            # Fallback for when parameters could not be determined, or the parameters were for
            # one reason or the other not part of the parameter description.
            # Solution: check what parameters were actually tried in the run.
            param_settings[param_name] = set(df_actual[param_name].unique())

    return param_settings


def determine_scale_type(possible_values: set) -> vz.ScaleType:
    """
    Check if the possible values best fit a linear or a logarithmic scale.
    """
    values_sorted = sorted(possible_values)
    r_lin = stats.linregress(values_sorted, range(0, len(values_sorted))).rvalue
    r_log = stats.linregress(values_sorted,
                             np.logspace(1, len(values_sorted), num=len(values_sorted))).rvalue
    r_reverse_log = stats.linregress(values_sorted, sorted(
        -np.logspace(1, len(values_sorted), num=len(values_sorted)))).rvalue
    if r_lin > r_log and r_lin > r_reverse_log:
        return vz.ScaleType.LINEAR
    if r_log > r_reverse_log:
        return vz.ScaleType.LOG
    return vz.ScaleType.REVERSE_LOG


def determine_search_space_weka(run: OpenMLRun) -> tuple[vz.SearchSpace, dict]:
    """Given an OpenML WEKA run, return the Vizier Search Space."""
    weka_string, = [ps['oml:value'] for ps in run.parameter_settings if ps['oml:name'] == 'search']
    return search_space_from_weka_string(weka_string)


def search_space_from_weka_string(weka_string) -> tuple[vz.SearchSpace, dict]:
    """
    Given a WEKA description, return the Vizier Search Space

    :param weka_string: The string describing all parameters. Example:
        [-property log_param_name -min 1 -max 5 -base 10 -expression pow(BASE,I)
         -property lin_param_name -min 2 -max 4
         -property list_param_name -list "1 2 3"]
    :return:
        Vizier Search Space
        parameter_convertors: a possibly empty dictionary that contain for a parameter name (the
        dict key) a function (the dict value) to convert each parameter value with.
    """

    property_descriptions = []  # a list with a dict per parameter, containing all settings
    # Parse the weka_string
    weka_string = weka_string[1:-1]  # remove the "[" and "]"

    if weka_string[0] == "\"" and weka_string[-1] == "\"":
        # Sometimes each property is enclosed in quotes. We remove this extra nesting here,
        # so that the string is always formatted the same way.
        weka_string = weka_string.replace("\",\"", " ")[1:-1]
    parts = shlex.split(weka_string)  # split by spaces, respecting the quotes

    i = 0
    while i < len(parts) - 1:
        part_cur = parts[i]
        part_nxt = parts[i + 1]
        if part_nxt.endswith(","):
            part_nxt = part_nxt[:-1]

        if part_cur == "-property":
            property_descriptions.append({"-property": part_nxt})
            i += 2
        elif len(property_descriptions) == 0:
            i += 1
        elif part_cur.startswith("-"):
            property_descriptions[-1][part_cur] = part_nxt
            i += 2
        else:
            i += 1

    search_space = vz.SearchSpace()
    root = search_space.root
    parameter_convertors = {}
    for param in property_descriptions:
        if "-list" in param:
            options = [try_convert_string_to_float(v) for v in param['-list'].split()]

            if all(isinstance(v, str) for v in options):
                root.add_categorical_param(name=param['-property'],
                                           feasible_values=options)
            else:
                add_number_param(root, options, param['-property'])
        else:
            min_ = float(param['-min'])
            max_ = float(param['-max'])
            if "-expression" in param:
                expression = param['-expression']
                if expression == "pow(BASE,I)":
                    scale_type = vz.ScaleType.LOG
                    base_ = float(param['-base'])
                    options = {base_ ** v for v in (min_, max_)}
                    parameter_convertors[param['-property']] = exp_func(base_)
                elif expression == "I":
                    scale_type = vz.ScaleType.LINEAR
                    options = {min_, max_}
                else:
                    raise ValueError(f"Expression unknown: {expression}")
            else:
                scale_type = vz.ScaleType.LINEAR
                options = {min_, max_}
            add_number_param(root, options, param['-property'], scale_type=scale_type)
    return search_space, parameter_convertors


def exp_func(base):
    return lambda v: base ** v


def add_number_param(root, options, param_name, scale_type=None):
    if scale_type is None:
        scale_type = determine_scale_type(options)
    # TODO: should we use root.add_discrete_param instead? Because we know the possible values?
    kwargs = {
        "name": param_name,
        "min_value": min(options),
        "max_value": max(options),
        "scale_type": scale_type
    }
    int_type = all(isinstance(v, int) or v.is_integer() for v in options)
    add_param = root.add_int_param if int_type else root.add_float_param
    add_param(**kwargs)


def try_convert_string_to_float(string: str):
    if string.startswith("\"") and string.endswith("\""):
        string = string[1:-1]
    try:
        return float(string)
    except ValueError:
        return string

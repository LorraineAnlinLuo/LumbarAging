""" general utility functions"""

import importlib
import json
import logging
import os
import random
import re
import shutil
import sys
import typing
import argparse
import torch
import numpy as np
from argparse import ArgumentParser
from collections.abc import MutableMapping
from box import Box

logger = logging.getLogger()


def listorstr(inp):
    if len(inp) == 1:
        return try_cast(inp[0])

    for i, val in enumerate(inp):
        inp[i] = try_cast(val)
    return inp


def try_cast(text):
    """ try to cast to int or float if possible, else return the text itself"""
    result = try_int(text, None)
    if result is not None:
        return result

    result = try_float(text, None)
    if result is not None:
        return result

    return text


def try_float(text, default: typing.Optional[int] = 0.0):
    result = default
    try:
        result = float(text)
    except Exception as _:
        pass
    return result


def try_int(text, default: typing.Optional[int] = 0):
    result = default
    try:
        result = int(text)
    except Exception as _:
        pass
    return result



def boxify_dict(args):
    """
  this takes a flat dictionary and break it into sub-dictionaries based on "." seperation
    a = {"model.a":  1, "model.b" : 2,  "alpha" : 3} will return Box({"model" : {"a" :1,
    "b" : 2}, alpha:3})
    a = {"model.a":  1, "model.b" : 2,  "model" : 3} will throw error
  """
    new_args = {}
    # iterate over keys and split on "."
    for key in args:
        if "." in key:
            temp_args = new_args
            for k in key.split(".")[:-1]:
                # create non-existent keys as dictionary recursively
                if temp_args.get(k) is None:
                    temp_args[k] = {}
                elif not isinstance(temp_args.get(k), dict):
                    raise TypeError(f"Key '{k}' has values as well as child")
                temp_args = temp_args[k]
            temp_args[key.split(".")[-1]] = args[key]
        else:
            if new_args.get(key) is None:
                new_args[key] = args[key]
            else:
                raise TypeError(f"Key '{key}' has values as well as child")

    return Box(new_args)


# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return Box(dict(items))


def str2bool(v: typing.Union[bool, str, int]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", 1):
        return True
    if v.lower() in ("no", "false", "f", "n", "0", 0):
        return False
    raise TypeError("Boolean value expected.")


def safe_isdir(dir_name):
    return os.path.exists(dir_name) and os.path.isdir(dir_name)


def safe_makedirs(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        print(e)


def jsonize(x: object) -> typing.Union[str, dict]:
    try:
        temp = json.dumps(x)
        return temp
    except Exception as e:
        return {}


def copy_code(folder_to_copy, out_folder, replace=False):
    logger.info(f"copying {folder_to_copy} to {out_folder}")

    if os.path.exists(out_folder):
        if not os.path.isdir(out_folder):
            logger.error(f"{out_folder} is not a directory")
            sys.exit()
        else:
            logger.info(f"Not deleting existing result folder: {out_folder}")
    else:
        os.makedirs(out_folder)

    # replace / with _
    folder_name = f'{out_folder}/{re.sub("/", "_", folder_to_copy)}'

    # create a new copy if something already exists
    if not replace:
        i = 1
        temp = folder_name
        while os.path.exists(temp):
            temp = f"{folder_name}_{i}"
            i += 1
        folder_name = temp
    else:
        if os.path.exists(folder_name):
            if os.path.isdir(folder_name):
                shutil.rmtree(folder_name)
            else:
                raise FileExistsError("There is a file with same name as folder")

    logger.info(f"Copying {folder_to_copy} to {folder_name}")
    shutil.copytree(folder_to_copy, folder_name)


def get_state_params(run_id, result_folder):
    """This searches for model and run id in result folder
        The logic is as follows

    if we are not given run_id there are four cases:
        - we want to restart the wandb run but too lazy to look up run-id or/and statefile
        - we want a new wandb fresh run
        - we are not using wandb at all and need to restart
        - we are not using wandb and need a fresh run

    Case 1/3:
        - If we want to restart the run, we expect the result_folder name to end with
        /run_<numeric>.
        - In this case, if we are using wandb then we need to go inside wandb folder, list all
        directory and pick up run id and (or) statefile
        - If we are not using wandb we just look for model inside the run_<numeric> folder and
        return statefile, run id as none

    case 2/4:
        if not 1/3, it is case 2/4

    This is expected to be a fail safe script. i.e any of run_id or statefile may not be specified
    and relies on whims of the user _-_
    """
    # if not resume get run number and create result_folder/run_{run_num}
    # if someone is resuming we expect them to give the exact folder name upto run num.

    # this part of code searches for run_id i.e will work only if we are using wandb
    if run_id is None:
        # if result folder if of type folder/run_<num>, then search for current checkpoint and
        # run-id else we will just create a new run with run_<num+1>

        regex = r"^.*/?run_[0-9]+/?$"
        if re.match(regex, result_folder):
            logger.info(f"not using wandb")
            if os.path.exists(f"{result_folder}/current_model.pt"):
                statefile = f"{result_folder}/current_model.pt"
                logger.info(f"using statefile {statefile}")
            else:
                # just start a new run
                pass
        else:
            # trailing is not run_<num>; that means user wants a new fresh run
            # so we give a fresh run and create a new folder
            # case 2/4
            last_run_num = max(
                [0] + [try_int(i[-4:]) for i in os.listdir(result_folder)]) + 1
            result_folder = f"{result_folder}/run_{last_run_num:04d}"
            logger.info(f"Creating new run with {result_folder}")
            safe_makedirs(result_folder)

    # search for last checkpoint in case --statefile is none and we are resuming

    return result_folder


if __name__ == "__main__":
    # test boxify_dict
    a = {"model.a": 1, "m   odel.b": 2, "alpha": 3}
    print(boxify_dict(a))

    try:
        a = {"model.a": 1, "model.b": 2, "model": 3}
        print(boxify_dict(a))
    except Exception as e:
        print(e)

    try:
        a = {"model": 4, "model.a": 1, "model.b": 2, "model": 3}
        print(boxify_dict(a))
    except Exception as e:
        print(e)

    try:
        a = {"model.a": 1, "model": 4, "model.b": 2, "model": 3}
        print(boxify_dict(a))
    except Exception as e:
        print(e)

    try:
        a = {"model": {"attr1": 1, "attr2": {"attr_attr_3": 3}}, "train": 10}
        print(flatten(a))
    except Exception as e:
        print(e)


def set_seed(seed):
    if isinstance(seed, list):
        torch_seed, numpy_seed, random_seed = seed
    else:
        torch_seed, numpy_seed, random_seed = seed, seed, seed

    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
    random.seed(random_seed)

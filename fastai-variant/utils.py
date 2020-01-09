import numpy as np
import os
import json
import audioutils
import torch
class Params():
    

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__
def load_params(params_path):
    json_path = os.path.join(params_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    return Params(json_path)


import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "gtav": GTA5DataSet,
        "syn": SynthiaDataSet
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '../data/city/'
    if name == 'gtav' or name == 'gtaUniform':
        return '../data/gtav/'
    if name == 'syn':
        return '../data/syn/SYNTHIA_RAND_CITYSCPAES/'

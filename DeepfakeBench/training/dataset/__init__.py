import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from .abstract_dataset import DeepfakeAbstractBaseDataset

_OPTIONAL_IMPORT_ERRORS = {}

def _optional_import(module_name, class_name):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
        globals()[class_name] = getattr(module, class_name)
    except Exception as e:
        globals()[class_name] = None
        _OPTIONAL_IMPORT_ERRORS[class_name] = e

_optional = [
    ("I2G_dataset", "I2GDataset"),
    ("iid_dataset", "IIDDataset"),
    ("ff_blend", "FFBlendDataset"),
    ("fwa_blend", "FWABlendDataset"),
    ("lrl_dataset", "LRLDataset"),
    ("pair_dataset", "pairDataset"),
    ("sbi_dataset", "SBIDataset"),
    ("lsda_dataset", "LSDADataset"),
    ("tall_dataset", "TALLDataset"),
]

for mod, cls in _optional:
    _optional_import(mod, cls)
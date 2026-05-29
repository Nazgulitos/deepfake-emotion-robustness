import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR

_OPTIONAL_IMPORT_ERRORS = {}

def _optional_import(module_name, class_name):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
        globals()[class_name] = getattr(module, class_name)
    except Exception as e:
        globals()[class_name] = None
        _OPTIONAL_IMPORT_ERRORS[class_name] = e

# optional slowfast utils
try:
    from .utils import slowfast
except Exception as e:
    slowfast = None
    _OPTIONAL_IMPORT_ERRORS["slowfast"] = e

_optional = [
    ("facexray_detector", "FaceXrayDetector"),
    ("xception_detector", "XceptionDetector"),
    ("efficientnetb4_detector", "EfficientDetector"),
    ("resnet34_detector", "ResnetDetector"),
    ("f3net_detector", "F3netDetector"),
    ("meso4_detector", "Meso4Detector"),
    ("meso4Inception_detector", "Meso4InceptionDetector"),
    ("spsl_detector", "SpslDetector"),
    ("core_detector", "CoreDetector"),
    ("capsule_net_detector", "CapsuleNetDetector"),
    ("srm_detector", "SRMDetector"),
    ("ucf_detector", "UCFDetector"),
    ("recce_detector", "RecceDetector"),
    ("fwa_detector", "FWADetector"),
    ("ffd_detector", "FFDDetector"),
    ("videomae_detector", "VideoMAEDetector"),
    ("clip_detector", "CLIPDetector"),
    ("timesformer_detector", "TimeSformerDetector"),
    ("xclip_detector", "XCLIPDetector"),
    ("sbi_detector", "SBIDetector"),
    ("ftcn_detector", "FTCNDetector"),
    ("i3d_detector", "I3DDetector"),
    ("altfreezing_detector", "AltFreezingDetector"),
    ("stil_detector", "STILDetector"),
    ("lsda_detector", "LSDADetector"),
    ("sladd_detector", "SLADDXceptionDetector"),
    ("pcl_xception_detector", "PCLXceptionDetector"),
    ("iid_detector", "IIDDetector"),
    ("lrl_detector", "LRLDetector"),
    ("rfm_detector", "RFMDetector"),
    ("uia_vit_detector", "UIAViTDetector"),
    ("multi_attention_detector", "MultiAttentionDetector"),
    ("sia_detector", "SIADetector"),
    ("tall_detector", "TALLDetector"),
    ("effort_detector", "EffortDetector"),
]

for mod, cls in _optional:
    _optional_import(mod, cls)
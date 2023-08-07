import ikomia
import os
import yaml
import cv2
import detectron2
from ikomia.utils.tests import run_for_test
from detectron2 import model_zoo


def test(t, data_dict):
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[::-1]
    input_img = t.getInput(0)
    input_img.setImage(img)

    config_paths = os.path.dirname(detectron2.__file__) + "/model_zoo"

    for root, dirs, files in os.walk(config_paths, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            possible_cfg = os.path.join(*file_path.split('/')[-2:])
            if "Keypoints" in possible_cfg and possible_cfg.endswith('.yaml') and "Base" not in possible_cfg:
                try:
                    model_zoo.get_checkpoint_url(possible_cfg)
                except RuntimeError:
                    continue
                params = t.get_parameters()
                params["model_name"] = possible_cfg.replace('.yaml', '')
                # without update = 1, model is not updated between 2 test
                params["update"] = "1"
                t.set_parameters(params)
                yield run_for_test(t)

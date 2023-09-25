# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from infer_detectron2_keypoints import update_path
from ikomia import utils, core, dataprocess
import copy
import os
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
import torch


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDetectron2KeypointsParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x"
        self.conf_det_thres = 0.5
        self.conf_kp_thres = 0.05
        self.cuda = True if torch.cuda.is_available() else False
        self.update = False
        self.use_custom_model = False
        self.config_file = ""
        self.model_weight_file = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.conf_det_thres = float(param_map["conf_det_thres"])
        self.conf_kp_thres = float(param_map["conf_kp_thres"])
        self.cuda = eval(param_map["cuda"])
        self.use_custom_model = eval(param_map["use_custom_model"])
        self.config_file = param_map["config_file"]
        self.model_weight_file = param_map["model_weight_file"]
        self.update = utils.strtobool(param_map["update"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
                    "model_name": self.model_name,
                    "conf_det_thres": str(self.conf_det_thres),
                    "conf_kp_thres": str(self.conf_kp_thres),
                    "cuda": str(self.cuda),
                    "use_custom_model": str(self.use_custom_model),
                    "config_file": self.config_file,
                    "model_weight_file": self.model_weight_file,
                    "update": str(self.update)}
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDetectron2Keypoints(dataprocess.CKeypointDetectionTask):

    def __init__(self, name, param):
        dataprocess.CKeypointDetectionTask.__init__(self, name)
        self.predictor = None
        self.cfg = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferDetectron2KeypointsParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Set cache dir in the algorithm folder to simplify deployment
        os.environ["FVCORE_CACHE"] = os.path.join(os.path.dirname(__file__), "models")

        if self.predictor is None or param.update:
            if param.model_weight_file != "":
                if os.path.isfile(param.model_weight_file):
                    param.use_custom_model = True

            if param.use_custom_model:
                with open(param.config_file, 'r') as file:
                    cfg_data = file.read()
                    self.cfg = CfgNode.load_cfg(cfg_data)
                connections = self.cfg.KEYPOINT_CONNECTION_RULES
                self.cfg.MODEL.WEIGHTS = param.model_weight_file
                name_to_index = {k: v for v, k in enumerate(self.cfg.KEYPOINT_NAMES)}
                keypoint_names = self.cfg.KEYPOINT_NAMES
            else:
                self.cfg = get_cfg()
                dataset_name, config_name = param.model_name.replace(os.path.sep, '/').split('/')
                config_path = os.path.join(os.path.dirname(detectron2.__file__), "model_zoo", "configs",
                                           dataset_name, config_name + '.yaml')
                self.cfg.merge_from_file(config_path)
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url((param.model_name + '.yaml').replace('\\', '/'))
                name_to_index = {k: v for v, k in
                                 enumerate(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_names"))}
                keypoint_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_names")
                connections = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_connection_rules")

            self.cfg.MODEL.DEVICE = 'cuda' if param.cuda and torch.cuda.is_available() else 'cpu'
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_det_thres
            self.kp_thres = param.conf_kp_thres
            self.predictor = DefaultPredictor(self.cfg)
            self.set_object_names(["person"])
            self.set_keypoint_names(list(keypoint_names))

            # Compute keypoint links
            keypoint_links = []
            for name1, name2, color in connections:
                link = dataprocess.CKeypointLink()
                link.start_point_index = name_to_index[name1]
                link.end_point_index = name_to_index[name2]
                link.label = f"{name1} - {name2}"
                link.color = list(color)
                keypoint_links.append(link)

            self.set_keypoint_links(keypoint_links)
            param.update = False
            print("Inference will run on " + self.cfg.MODEL.DEVICE)

        # Get input :
        img_input = self.get_input(0)
        self.emit_step_progress()

        if img_input.is_data_available():
            img = img_input.get_image()
            self.infer(img)

        # Step progress bar:
        self.emit_step_progress()
        # Call endTaskRun to finalize process
        self.end_task_run()

    def infer(self, img):
        outputs = self.predictor(img)
        obj_id = 0

        if "instances" in outputs.keys():
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes
            scores = instances.scores
            classes = instances.pred_classes
            pred_keypoints = instances.pred_keypoints

            for box, score, cls, pred_kp in zip(boxes, scores, classes, pred_keypoints):
                score = float(score)
                if score >= self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                    box_x1, box_y1, box_x2, box_y2 = box.numpy()
                    w = float(box_x2 - box_x1)
                    h = float(box_y2 - box_y1)
                    keypts = []
                    kept_kp_id = []

                    for link in self.get_keypoint_links():
                        kp1, kp2 = pred_kp[link.start_point_index], pred_kp[link.end_point_index]
                        x1, y1, conf1 = kp1
                        x2, y2, conf2 = kp2
                        cond1 = conf1 >= self.kp_thres
                        cond2 = conf2 >= self.kp_thres

                        if cond1 and cond2:
                            if link.start_point_index not in kept_kp_id and cond1:
                                kept_kp_id.append(link.start_point_index)
                                keypts.append((link.start_point_index, dataprocess.CPointF(float(x1), float(y1))))

                            if link.end_point_index not in kept_kp_id and cond2:
                                kept_kp_id.append(link.end_point_index)
                                keypts.append((link.end_point_index, dataprocess.CPointF(float(x2), float(y2))))

                    self.add_object(obj_id, 0, score, float(box_x1), float(box_y1), w, h, keypts)
                    obj_id += 1


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDetectron2KeypointsFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_keypoints"
        self.info.short_description = "Inference for Detectron2 keypoint models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Pose"
        self.info.version = "1.0.1"
        self.info.icon_path = "icons/detectron2.png"
        self.info.authors = "Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick"
        self.info.article = "Detectron2"
        self.info.journal = ""
        self.info.year = 2019
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://detectron2.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "infer, detectron2, keypoint, pose"

    def create(self, param=None):
        # Create process object
        return InferDetectron2Keypoints(self.info.name, param)

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
from ikomia.core import CPointF


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
        self.custom = False
        self.cfg_file = ""
        self.weights = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.conf_det_thres = float(param_map["conf_det_thres"])
        self.conf_kp_thres = float(param_map["conf_kp_thres"])
        self.cuda = eval(param_map["cuda"])
        self.custom = eval(param_map["custom"])
        self.cfg_file = param_map["cfg_file"]
        self.weights = param_map["weights"]
        self.update = utils.strtobool(param_map["update"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["model_name"] = self.model_name
        param_map["conf_det_thres"] = str(self.conf_det_thres)
        param_map["conf_kp_thres"] = str(self.conf_kp_thres)
        param_map["cuda"] = str(self.cuda)
        param_map["custom"] = str(self.custom)
        param_map["cfg_file"] = self.cfg_file
        param_map["weights"] = self.weights
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDetectron2Keypoints(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.predictor = None
        self.cfg = None
        self.name2id = None
        self.connections = None
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CBlobMeasureIO())
        self.addOutput(dataprocess.CDataStringIO())
        # Create parameters class
        if param is None:
            self.setParam(InferDetectron2KeypointsParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Get parameters :
        param = self.getParam()
        if self.predictor is None or param.update:
            if param.custom:
                with open(param.cfg_file, 'r') as file:
                    cfg_data = file.read()
                    self.cfg = CfgNode.load_cfg(cfg_data)
                self.connections = self.cfg.KEYPOINT_CONNECTION_RULES
                self.cfg.MODEL.WEIGHTS = param.weights
                self.name2id = {k: v for v, k in enumerate(self.cfg.KEYPOINT_NAMES)}
            else:
                self.cfg = get_cfg()
                config_path = os.path.join(os.path.dirname(detectron2.__file__), "model_zoo", "configs",
                                           param.model_name + '.yaml')
                self.cfg.merge_from_file(config_path)
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url((param.model_name + '.yaml').replace('\\', '/'))
                self.cfg.MODEL.DEVICE = 'cuda' if param.cuda else 'cpu'
                self.name2id = {k: v for v, k in
                                enumerate(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_names"))}
                self.connections = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("keypoint_connection_rules")
                self.predictor = DefaultPredictor(self.cfg)

            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_det_thres
            self.kp_thres = param.conf_kp_thres
            self.predictor = DefaultPredictor(self.cfg)

            param.update = False
            print("Inference will run on " + ('cuda' if param.cuda else 'cpu'))

            connection_rules_output = self.getOutput(3)
            starting_kp = ["%s:%d" %(name1,self.name2id[name1]) for name1, name2, color in self.connections]
            ending_kp = ["%s:%d" %(name2,self.name2id[name2]) for name1, name2, color in self.connections]
            connection_rules_output.addValueList(starting_kp, "Starting point")
            connection_rules_output.addValueList(ending_kp, "Ending point")

        # Get input :
        input = self.getInput(0)

        # Get output :
        graphics_output = self.getOutput(1)
        numeric_output = self.getOutput(2)

        if input.isDataAvailable():
            graphics_output.setNewLayer("Detectron2_Detection")
            graphics_output.setImageIndex(0)
            img = input.getImage()
            numeric_output.clearData()
            self.infer(img, graphics_output, numeric_output)
        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def infer(self, img, graphics_output, numeric_output):
        outputs = self.predictor(img)
        if "instances" in outputs.keys():
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes
            scores = instances.scores
            classes = instances.pred_classes
            pred_keypoints = instances.pred_keypoints

            for box, score, cls, pred_kp in zip(boxes, scores, classes, pred_keypoints):
                score = float(score)
                if score >= self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                    x1, y1, x2, y2 = box.numpy()
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    prop_rect = core.GraphicsRectProperty()
                    prop_rect.pen_color = [255, 0, 0]
                    graphics_box = graphics_output.addRectangle(float(x1), float(y1), w, h, prop_rect)
                    # Object results
                    results = []
                    confidence_data = dataprocess.CObjectMeasure(
                        dataprocess.CMeasure(core.MeasureId.CUSTOM, "Confidence"),
                        score,
                        graphics_box.getId(),
                        "person")
                    box_data = dataprocess.CObjectMeasure(
                        dataprocess.CMeasure(core.MeasureId.BBOX),
                        graphics_box.getBoundingRect(),
                        graphics_box.getId(),
                        "person")

                    results.append(confidence_data)
                    results.append(box_data)

                    kept_kp_id = []
                    kept_kp_pos = []
                    connections_idx = []
                    for connection_to_do in self.connections:
                        name_kp1, name_kp2, color = connection_to_do
                        id_kp1, id_kp2 = self.name2id[name_kp1], self.name2id[name_kp2]
                        kp1, kp2 = pred_kp[id_kp1], pred_kp[id_kp2]
                        x1, y1, conf1 = kp1
                        x2, y2, conf2 = kp2
                        cond1 = conf1 >= self.kp_thres
                        cond2 = conf2 >= self.kp_thres
                        if cond1 and cond2:
                            if id_kp1 not in kept_kp_id and cond1:
                                kept_kp_id.append(id_kp1)
                                kept_kp_pos.append([float(x1), float(y1)])
                                graphics_output.addPoint(CPointF(float(x1), float(y1)))
                            if id_kp2 not in kept_kp_id and cond2:
                                kept_kp_id.append(id_kp2)
                                kept_kp_pos.append([float(x2), float(y2)])
                                graphics_output.addPoint(CPointF(float(x2), float(y2)))
                            connections_idx.append(id_kp1)
                            connections_idx.append(id_kp2)

                            pts = [CPointF(float(x1), float(y1)), CPointF(float(x2), float(y2))]
                            properties_line = core.GraphicsPolylineProperty()
                            properties_line.pen_color = [int(c) for c in color]
                            graphics_output.addPolyline(pts, properties_line)
                    kp_id_pos = [[idx] + pos for idx, pos in zip(kept_kp_id, kept_kp_pos)]

                    keypoints_data1 = dataprocess.CObjectMeasure(
                        dataprocess.CMeasure(core.MeasureId.CUSTOM, "Keypoints id and pos"),
                        [item for sublist in kp_id_pos for item in sublist],
                        graphics_box.getId(),
                        "person")

                    keypoints_data2 = dataprocess.CObjectMeasure(
                        dataprocess.CMeasure(core.MeasureId.CUSTOM, "Id of linked keypoints"),
                        connections_idx,
                        graphics_box.getId(),
                        "person")
                    results.append(keypoints_data1)
                    results.append(keypoints_data2)
                    numeric_output.addObjectMeasures(results)


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDetectron2KeypointsFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_keypoints"
        self.info.shortDescription = "Inference for Detectron2 keypoint models"
        self.info.description = "Inference for Detectron2 keypoint models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Pose"
        self.info.version = "1.0.1"
        self.info.iconPath = "icons/detectron2.png"
        self.info.authors = "Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick"
        self.info.article = "Detectron2"
        self.info.journal = ""
        self.info.year = 2019
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentationLink = "https://detectron2.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "infer, detectron2, keypoint, pose"

    def create(self, param=None):
        # Create process object
        return InferDetectron2Keypoints(self.info.name, param)

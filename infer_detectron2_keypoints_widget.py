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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_detectron2_keypoints.infer_detectron2_keypoints_process import InferDetectron2KeypointsParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
import detectron2
import os


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferDetectron2KeypointsWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferDetectron2KeypointsParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        config_paths = os.path.dirname(detectron2.__file__) + "/model_zoo"

        available_cfg = []
        for root, dirs, files in os.walk(config_paths, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                possible_cfg = os.path.join(*file_path.split('/')[-2:])
                if "Keypoints" in possible_cfg and possible_cfg.endswith('.yaml'):
                    available_cfg.append(possible_cfg.replace('.yaml', ''))
        self.combo_model = pyqtutils.append_combo(self.gridLayout, "Model Name")
        for model_name in available_cfg:
            self.combo_model.addItem(model_name)
        self.combo_model.setCurrentText(self.parameters.model_name)

        self.double_spin_det_thres = pyqtutils.append_double_spin(self.gridLayout, "Detection confidence threshold",
                                                                  self.parameters.conf_det_thres, min=0., max=1.,
                                                                  step=1e-2, decimals=2)
        self.double_spin_kp_thres = pyqtutils.append_double_spin(self.gridLayout, "Keypoint confidence threshold",
                                                                 self.parameters.conf_kp_thres, min=0., max=1.,
                                                                 step=1e-2, decimals=2)
        self.check_cuda = pyqtutils.append_check(self.gridLayout, "Cuda", self.parameters.cuda)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def onApply(self):
        # Apply button clicked slot

        self.parameters.conf_det_thres = self.double_spin_det_thres.value()
        self.parameters.conf_kp_thres = self.double_spin_kp_thres.value()
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.update = True

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferDetectron2KeypointsWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_detectron2_keypoints"

    def create(self, param):
        # Create widget object
        return InferDetectron2KeypointsWidget(param, None)
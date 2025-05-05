<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_detectron2_keypoints/main/icons/detectron2.png" alt="Algorithm icon">
  <h1 align="center">infer_detectron2_keypoints</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_detectron2_keypoints">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_detectron2_keypoints">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_detectron2_keypoints/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_detectron2_keypoints.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run keypoints detection models from Detectron2 framework.

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_detectron2_keypoints/main/images/rugby-result.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add keypoints detection algorithm
keypts_detector = wf.add_task(name="infer_detectron2_keypoints", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_detectron2_keypoints/main/images/rugby.jpg")

# Display result
display(keypts_detector.get_image_with_graphics(), title="Detectron2 keypoints")

```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_detectron2_keypoints", auto_connect=True)

algo.set_parameters({
    "model_name": "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x",
    "conf_det_thres": "0.5",
    "conf_kp_thres": "0.05",
    "cuda": "True",
    "use_custom_model": "False",
    "config_file": "",
    "model_weight_file": "",
})

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_detectron2_keypoints/main/images/rugby.jpg")
```

- **model_name** (str, default="COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x"): pre-trained model name. Choose one on the list below:
    - COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x
    - COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x
    - COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x
    - COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x
- **conf_det_thres** (float, default=0.5): object detection confidence.
- **conf_kp_thres** (float, default=0.05): keypoints detection confidence.
- **cuda** (bool, default=True): CUDA acceleration if True, run on CPU otherwise.
- **use_custom_model** (bool, default=False): flag to enable the custom train model choice.
- **config_file** (str, default=""): path to model config file (.yaml). Only for custom model.
- **model_weight_file** (str, default=""): path to model weights file (.pt). Only for custom model.

***Note***: parameter key and value should be in **string format** when added to the dictionary.

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add keypoints detection algorithm
keypts_detector = wf.add_task(name="infer_detectron2_keypoints", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_detectron2_keypoints/main/images/rugby.jpg")

# Iterate over outputs
for output in keypts_detector.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

Detectron2 keypoints detection algorithm generates 2 outputs:

1. Forwaded original image (CImageIO)
2. Keypoints detection output (CKeypointsIO)
# CHCNet: Cross-scale Hierarchical Context-aware Network for Small Object Detection in UAV Images

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation of the paper **"Cross-scale Hierarchical Context-aware Network for Small Object Detection in UAV Images"**.

## 📖 Introduction
Detecting small objects in Unmanned Aerial Vehicle (UAV) images remains a major bottleneck due to the inadequate preservation of fine-grained details and limited cross-layer semantic integration. To address these challenges, we propose **CHCNet**. 

### ✨ Highlights
* **CSPP (Cross Stage Partial Pyramid)**: Decouples shallow details and deep semantics to strengthen multi-scale feature modeling.
* **AFCF (Adaptive Feature Calibration Fusion)**: Adaptively calibrates cross-resolution features to suppress noise and enhance complementarity bidirectionally.
* **CCDH (Cascaded Cross-level Detection Head)**: Balances local details and global semantic context via a cascaded cross-scale attention mechanism.
* Achieves **State-of-the-Art (SOTA)** results on mainstream UAV datasets: **VisDrone** and **UAVDT**.

---

## 📁 Code Structure: Where are the core modules?

Our proposed innovative modules have been modularized for easy integration. You can find the core implementation of **CSPP**, **AFCF**, and **CCDH** in the following directory:

```bash
# Core modules location:
ultralytics/nn/our_modules/
├── cspp.py         # Implementation of CSP-Pyramid (CSPP)
├── afcf.py         # Implementation of Adaptive Feature Calibration Fusion (AFCF)
├── ccdh.py         # Implementation of Cascaded Cross-level Detection Head (CCDH)
└── __init__.py
```

*Note: The overall network architecture config files (`.yaml`) integrating these modules are located in `ultralytics/cfg/models/`.*

---

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yue-2002/CHCNet.git
cd CHCNet
```

2. Create a virtual environment and install the required dependencies:
```bash
conda create -n chcyolo python=3.8 -y
conda activate chcyolo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

---

## 🚀 How to Run (Usage)

### 1. Data Preparation
Please download the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) or [UAVDT](https://sites.google.com/site/daviddo0323/projects/uavdt) datasets and place them in the `datasets/` folder. Ensure the data format follows the standard YOLO format.

### 2. Training the Model
We provide a unified training script `train.py`. To train CHCNet on the VisDrone dataset, simply run the following command in your terminal:

```bash
# Standard training command
python train.py --model ultralytics/cfg/models/CHCNet.yaml --data datasets/VisDrone.yaml --epochs 300 --batch-size 2 --device 0
```
* **`--model`**: Path to the CHCNet architecture config file.
* **`--data`**: Path to the dataset config file.
* **`--batch-size`**: Set to 2 (as described in the paper implementation details).
* **`--device`**: GPU ID (e.g., `0` for single GPU, `0,1` for multi-GPU).

### 3. Evaluation & Inference
Once you have a trained model weight file (e.g., `your_trained_weight.pt`), you can evaluate it on the validation set:
```bash
python val.py --model your_trained_weight.pt --data datasets/VisDrone.yaml
```

To run inference on your own UAV images or videos:
```bash
python predict.py --model your_trained_weight.pt --source your_image_folder/
```

---

## 📊 Results

Quantitative comparisons on the VisDrone dataset:

| Model | mAP | AP50 | AP75 | APS (Small) | APM (Medium) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (PANet) | 34.1 | 54.4 | 35.5 | 25.3 | 46.1 |
| **CHC-Net (Ours)** | **38.5** | **60.3** | **41.1** | **29.9** | **50.0** |

---

## 📝 Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{chcnet2026,
  title={Cross-scale Hierarchical Context-aware Net for Small Object Detection in UAV Images},
  author={Yang, Feng and Wang, Ziqian and Chen, Tao and Qin, Anyong and Liu, Yin and Zhao, Yue and Song, Tiecheng and Luo, Fulin},
  journal={ESWA},
  year={2026}
}
```

## 📧 Contact
If you have any questions about the code or paper, please feel free to open an issue or contact `qian85822137@163.com` / `luoflyn@163.com`.

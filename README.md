# SggNet: A Lightweight Semantic- and Graph-Guided Network for Advanced Optical Remote Sensing Image Salient Object Detection (Under Review)

Thank you for your interest in **SggNet**, a novel lightweight network designed for salient object detection (SOD) in optical remote sensing images (ORSIs). This repository provides part of the code for SggNet. The full implementation and key components will be made available after the paper completes the peer-review process.

---

## 🚨 Notice

The paper presenting SggNet is currently **under review**. To preserve the integrity of the review process, only partial code is being shared at this stage. The full implementation, including essential model details, will be released upon acceptance. Stay tuned for updates!

---

## 📝 Overview

SggNet is a lightweight and efficient network for ORSI-SOD, achieving:
- **Parameters**: 2.70M
- **FLOPs**: 1.38G
- **Inference Speed**: 108 FPS

It demonstrates superior performance compared to state-of-the-art lightweight ORSI-SOD methods, delivering accurate saliency detection, sharper boundaries, and clearer activation maps.

---

## 📊 Results

Below are sample results showcasing the effectiveness of SggNet:

### Example Outputs
| Input                  | Ground Truth          | Predicted Saliency Map |
|------------------------|-----------------------|-------------------------|
| ![Input](example1.jpg) | ![GT](ground_truth1.jpg) | ![Prediction](prediction1.jpg) |

### Key Metrics on [EORSSD and ORSSD Datasets](https://github.com/LittleGrey-hjp/SggNet)
| Method       | $S_m$  | $F_\beta^{max}$ | $E_\phi^{mean}$ | MAE    | FLOPs | Parameters |
|--------------|---------|-----------------|-----------------|--------|-------|------------|
| **SggNet**   | 0.9279  | 0.8770          | 0.9689          | 0.0068 | 1.38G | 2.70M      |

---

## 🔧 Current Release

### Included:
- Partial implementation and Scripts of SggNet.
- Visualization scripts for saliency map outputs.

### Coming Soon:
- Full model implementation.
- Training and evaluation scripts.
- Additional experiments and insights.

---

## 📥 Installation and Usage

### Clone the Repository
```bash
git clone https://github.com/LittleGrey-hjp/SggNet
cd SggNet
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Training Configuration
- The pretrained model(MobileNetv2) is stored in [Google Drive]() and [Baidu Drive]() (). After downloading, please change the file path in the corresponding code.
- Run `train.sh` to train.

### Testing Configuration

- Our well-trained model is stored in [Google Drive]() and [Baidu Drive]() (). After downloading, please change the file path in the corresponding code.
- Run `test.sh` to train.

### Evaluation

- Evaluate SggNet: After configuring the test dataset path, run `eval.sh` in the `srun` folder for evaluation.
- PR-Curves: We provide the code for obtaining PR-Curves through detection results. Please refer to 'PR_Curve.py'.

---

## 📬 Contact

For questions or feedback, feel free to open an issue on GitHub or contact us via email at [darrellduncan313@gmail.com](darrellduncan313@gmail.com).

---


### 💡 Stay Updated

We appreciate your interest and patience! The full implementation and additional resources will be made available after the review process is complete. 🎉

--- 

Let me know if you’d like additional customization!

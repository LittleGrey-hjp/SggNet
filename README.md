# SggNet: A Lightweight Semantic- and Graph-Guided Network for Advanced Optical Remote Sensing Image Salient Object Detection
---

## 🚨 Notice

-> The paper presenting SggNet is currently **under review**. To preserve the integrity of the review process, only partial code is being shared at this stage. The full implementation, including essential model details, will be released upon acceptance. Stay tuned for updates!

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
- In Figure 1, we visualize saliency maps generated by SggNet compared to other state-of-the-art methods in challenging scenarios:
![Qualitative Results](https://github.com/LittleGrey-hjp/SggNet/blob/main/visual-compare.png)

### Key Metrics on [EORSSD and ORSSD Datasets](https://github.com/LittleGrey-hjp/SggNet)
| Dataset   | $S_m \uparrow$ | $F^{max}_{\beta} \uparrow$ | $F^{mean}_{\beta} \uparrow$ | $F^{adp}_{\beta} \uparrow$ | $E^{max}_{\phi} \uparrow$ | $E^{mean}_{\phi} \uparrow$ | $E^{adp}_{\phi} \uparrow$ | $\mathcal{M} \downarrow$ |
|-----------|----------------|----------------------------|-----------------------------|----------------------------|---------------------------|----------------------------|---------------------------|--------------------------|
| EORSSD    | 0.9279         |  0.8770                    |   0.8596                    |  0.8386                    |  0.9762                   |   0.9689                   |  0.9678                   |  0.0068                  |
| ORSSD     | 0.9342         |  0.9032                    |   0.8896                    |  0.8884                    |  0.9759                   |   0.9695                   |  0.9720                   |  0.0111                  |

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
The pretrained model(MobileNetv2) is stored in [Google Drive](https://drive.google.com/file/d/1rhBFs-P3M7zQyLv8IDXTqa-pmioy2qLy/view?usp=drive_link) and [Baidu Drive](https://pan.baidu.com/s/1xVT_ebonD5bK6U39oSodFQ) (zskr). After downloading, please change the file path in the corresponding code.
```bash
Run `train.sh` to train.
```

### Testing Configuration
Our well-trained model is stored in [Google Drive](https://drive.google.com/file/d/14mHtCHAZrLik2ZVH3CRqX2_d7L6lu6NO/view?usp=drive_link) and [Baidu Drive](https://pan.baidu.com/s/1dQU5eXDyeSVIHMm2BjvwWA) (3knk). After downloading, please change the file path in the corresponding code.
```bash
Run `test.sh` to train.
```

### Detection Maps
Our Detection Maps are stored in [Google Drive](https://drive.google.com/drive/folders/1dYkE5saknjTFtLbMQHa37us59Vt6XK4v?usp=drive_link). Please check.
```bash
Run `test.sh` to train.

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

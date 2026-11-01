# UNet Semantic Segmentation for Geometric Shapes

This repository contains a TensorFlow/Keras implementation of the **U-Net architecture** designed for multi-class semantic segmentation on 2D synthetic geometric shapes. The project structure is organized to facilitate local prototyping as well as GPU-accelerated model training on platforms like Kaggle.

---

## 📌 Project Overview

* **Architecture:** U-Net (Encoder-Decoder with Skip Connections)
* **Framework:** TensorFlow / Keras
* **Task:** Multi-class Semantic Segmentation (4 classes: Background, Circles, Squares, Triangles)
* **Dataset:** Synthetic geometric shapes with corresponding segmentation masks

---

## 📁 Repository Structure

```text
.
├── data/                       # Dataset directory (Excluded from Git tracking)
│   ├── train/                  # Training set (200 images & masks)
│   │   ├── images/
│   │   └── masks/
│   └── test/                   # Evaluation set (40 images)
│       └── images/
├── models/                     # Saved model artifacts & weights (Excluded from Git)
│   └── unet_trained_model/
│       ├── config.json
│       ├── metadata.json
│       └── model.weights.h5
├── notebooks/                  # Jupyter notebooks & visual assets
│   ├── Unet.png                # Architecture diagrams / results
│   └── unet-segmentation-train.ipynb
├── .gitignore                  # Git tracking rules
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/u-net-project2.git
   cd u-net-project2
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install tensorflow numpy matplotlib pillow opencv-python
   ```

---

## 🚀 Running on Kaggle

Due to repository size limits on GitHub, large datasets (`data/`) and heavy model weight files (`*.h5`) are excluded via `.gitignore`. 

To train this model on Kaggle using a GPU:

1. **Upload Dataset:** Compress the `data/` directory into a `.zip` archive and upload it as a private/public **Kaggle Dataset** (e.g., `u-net-dataset.zip`).
2. **Import Notebook:** Upload `notebooks/unet-segmentation-train.ipynb` into a new Kaggle Notebook.
3. **Attach Dataset:** Link the uploaded dataset to your Kaggle Notebook via the **Add Data** sidebar menu.
4. **Update File Paths:** Adjust the dataset path to reference `/kaggle/input/u-net-dataset/data/`.
5. **Train with GPU:** Turn on the `GPU T4` or `GPU T4 x2` accelerator in notebook settings and run all cells.

---

## 📊 Model & Training Details

- **Input Shape:** `(256, 256, 3)` (or configured image resolution)
- **Output Shape:** `(256, 256, 4)` with `softmax` activation for 4 distinct segmentation classes.
- **Loss Function:** `sparse_categorical_crossentropy` / `categorical_crossentropy`
- **Metrics:** Accuracy & Mean Intersection over Union (mIoU)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
# GAN-CIFAR10-Image-Generation

A deep learning project implementing a Generative Adversarial Network (GAN) using Keras and TensorFlow to generate synthetic images from the CIFAR-10 dataset (filtered to class 8).

## 🚀 Project Overview
This project demonstrates how GANs can be trained to generate images from a specific image class using adversarial learning.

## 🧠 Key Components
- Generator & Discriminator built with Keras Sequential API
- Dataset: CIFAR-10 (single class)
- Training loop over 15,000 epochs with visualization every 2500 steps

## 📊 Results
The Generator evolved from producing random noise to structured—but still unrealistic—images. Mode collapse was observed by epoch 15,000.

## 📁 Project Structure
```
GAN-CIFAR10-Image-Generation/
├── gan_cifar10.py               # Python code
├── README.md                    # This file
├── analysis/
│   └── GAN_CIFAR10_Project_Report.pdf
├── results/
│   ├── epoch_0000.png
│   ├── epoch_2500.png
│   └── epoch_15000.png
└── requirements.txt             # Optional
```

## ✅ How to Run
```bash
pip install -r requirements.txt
python gan_cifar10.py
```

## 📈 Future Enhancements
- Add Conditional GAN (CGAN)
- Experiment with WGAN-GP for stability
- Evaluate outputs using FID score

## 🧑‍💻 Author
Joanna Ciesielski  
[GitHub](https://github.com/joanna-ciesielski) | [LinkedIn](https://linkedin.com/in/joanna-ciesielski)

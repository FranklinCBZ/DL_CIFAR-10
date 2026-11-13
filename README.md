# CIFAR-10 Image Classification Using Pre-Activation SE-ResNet

**Deep Learning Project 1 — Fall 2026**

**Team:** Shallow Learning  
**Team Members:** Franklin Chen, Mengzhe Wang

This project explores the design and optimization of Residual Neural Networks (ResNets) for CIFAR-10 image classification under a strict constraint of **5 million trainable parameters**.

Our final model combines **Pre-Activation Residual Blocks, Squeeze-and-Excitation (SE) attention, dropout, and alternating horizontal-flip augmentation**.

All models were trained from scratch using PyTorch, without externally pretrained weights.

## Final Results

| Metric | Result |
|---|---|
| Best architecture | Pre-Activation Deep SE-ResNet (V13) |
| **Kaggle public accuracy** | **87.00%** |
| Trainable parameters | 4,763,631 |
| Parameter limit | 5,000,000 |
| Residual blocks | 11 |
| Training images | 50,000 |
| Initial training | 250 epochs |
| Additional fine-tuning | 30 epochs |
| Total training | 280 epochs |
| Inference | Horizontal-flip TTA |

**Our final model achieved a Kaggle public leaderboard score of 0.870 while remaining below the parameter limit.**

The reported Kaggle accuracy is a public leaderboard result, not a locally calculated accuracy on hidden test labels.

## 1. Project Overview

The objective was to maximize classification accuracy on a held-out CIFAR-10 competition test set while designing a modified ResNet containing no more than five million trainable parameters.

We explored multiple architectures and training strategies, including:

- Standard ResNet and deeper residual networks
- Squeeze-and-Excitation (SE) attention
- Convolutional Block Attention Module (CBAM)
- Dropout and label smoothing
- MixUp and Random Erasing
- Random and alternating horizontal flips
- Cosine learning-rate scheduling
- Pre-Activation Residual Blocks
- Low-learning-rate fine-tuning
- Test-Time Augmentation (TTA)

Our final architecture was selected after comparing the results of multiple experiments.

## 2. Final Model Architecture

The final V13 model is a **Pre-Activation Deep SE-ResNet**, containing 11 residual blocks across four stages.

### Architecture Configuration

| Component | Configuration |
|---|---|
| Input | 3 × 32 × 32 |
| Stem | 3 × 3 convolution, 39 channels |
| Stage 1 | 3 PreAct SE blocks, 39 channels |
| Stage 2 | 3 PreAct SE blocks, 78 channels |
| Stage 3 | 3 PreAct SE blocks, 156 channels |
| Stage 4 | 2 PreAct SE blocks, 312 channels |
| Pooling | Global Average Pooling |
| Dropout | 0.15 |
| Classifier | Fully connected, 10 outputs |

### Pre-Activation Residual Blocks

Unlike conventional ResNet blocks, which apply ReLU after residual addition, our pre-activation blocks apply BatchNorm and ReLU before each convolution.

Each residual block follows:

1. BatchNorm → ReLU → 3 × 3 Convolution
2. BatchNorm → ReLU → 3 × 3 Convolution
3. Squeeze-and-Excitation attention
4. Addition with the identity or projection shortcut

The residual operation is:

\[
y=S(x)+SE(F_{\mathrm{pre}}(x))
\]

where \(F_{\mathrm{pre}}(x)\) represents the pre-activation convolutional transformation.

This design preserves a direct shortcut path while allowing SE attention to adaptively emphasize informative feature channels.

## 3. Dataset and Preprocessing

The competition dataset contains:

| Dataset | Images |
|---|---:|
| Labeled training set | 50,000 |
| Unlabeled test set | 2,000 |
| Image resolution | 32 × 32 |
| Classes | 10 |

During development, models were evaluated using a stratified split of 45,000 training images and 5,000 validation images.

For the final V13 model, all 50,000 labeled images were used for training.

### Data Augmentation

Our final training pipeline uses:

- Random cropping with four-pixel padding
- Alternating horizontal flips between epochs
- Random Erasing with probability 0.5
- Random data shuffling
- Channel-wise normalization

Normalization constants:

```python
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
```

Alternating horizontal flipping systematically changes image orientation between consecutive epochs, rather than applying independent random flips to individual images.

## 4. Training Configuration

We used SGD with Nesterov momentum, label smoothing, and cosine learning-rate scheduling.

| Hyperparameter | Initial Training | Fine-Tuning |
|---|---|---|
| Epochs | 250 | 30 |
| Learning rate | 0.1 | 0.001 |
| Batch size | 128 | 128 |
| Momentum | 0.9 | 0.9 |
| Weight decay | 0.0005 | 0.0005 |
| Label smoothing | 0.1 | 0.1 |
| Scheduler | Cosine Annealing | New cosine schedule |
| Augmentation | Alternating flips | Alternating flips |

After 250 epochs, the model reached approximately 99.60% training accuracy. We then continued from the epoch-250 weights for another 30 epochs using a lower learning rate.

During inference, we used horizontal-flip TTA by averaging predicted probabilities from original and flipped images.

## 5. Environment and Installation

The project was developed using Python and PyTorch with CUDA acceleration.

**Development hardware:** NVIDIA GeForce RTX 5080 (16 GB VRAM)

**Software:** Python, PyTorch, torchvision, NumPy, pandas, scikit-learn, Matplotlib, and Pillow.

### Installation

Clone the repository:

```bash
git clone https://github.com/FranklinCBZ/DL_CIFAR-10.git
cd DL_CIFAR-10
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-compatible PyTorch installation is recommended for training.

### Dataset Setup

Organize the Kaggle competition files as follows:

```text
DL_CIFAR-10/
├── data/
│   ├── train_images/
│   │   ├── train_00000.png
│   │   └── ...
│   └── test_images/
│       ├── test_0000.png
│       └── ...
├── train_labels.csv
├── test.csv
└── sample_submission.csv
```

The dataset must be obtained separately from the course's Kaggle competition.

## 6. Reproducing the Final Model

Run all commands from the project root.

### Step 1 — Verify the architecture

```bash
python model_v13_preact.py
```

The V13 architecture should have:

```text
Trainable parameters: 4,763,631
Under 5M limit: True
Output shape: torch.Size([2, 10])
```

### Step 2 — Train V13 from scratch

```bash
python train_v13_preact.py
```

This runs the initial 250-epoch training stage using the alternating-flip dataset loader.

The trained model is saved under:

```text
experiments/kaggle_v13_preact_alt_full50k/
```

The epoch-250 checkpoint is:

```text
model_epoch_250.pth
```

### Step 3 — Fine-tune for 30 additional epochs

```bash
python finetune_v13.py
```

This loads the epoch-250 model weights and performs another 30 training epochs using a fresh optimizer and cosine learning-rate schedule with an initial learning rate of 0.001.

The fine-tuned checkpoint is saved at:

```text
experiments/kaggle_v13_preact_finetuned_280/final_model.pth
```

### Step 4 — Generate Kaggle predictions

```bash
python predict_v13_280.py
```

The prediction script loads the fine-tuned V13 checkpoint and applies two-view horizontal-flip test-time augmentation.

It generates a submission file in the required Kaggle format:

```csv
id,label
0,3
1,7
...
```

The output filename is:

```text
submission_kaggle_v13_epoch280_tta.csv
```

The class labels above are illustrative; actual predictions are generated by the trained model.

### Step 5 — Submit to Kaggle

Upload the generated CSV to the competition's submission page.

Our recorded public leaderboard accuracy for the final V13 model was **0.870 (87.00%)**.

## 7. Experimental Results

We compared multiple architectures and training methods throughout development.

| Experiment | Description | Kaggle Public Accuracy |
|---|---|---:|
| V1 | Baseline ResNet | 81.70% |
| V3 | SE-ResNet + TTA | 85.10% |
| V5 | CBAM-ResNet + TTA | 83.80% |
| V6 | Deeper SE-ResNet + TTA | 85.40% |
| V7 | Deep SE-ResNet + Dropout | 86.00% |
| V8 | GitHub-derived CIFAR ResNet | 86.10% |
| V9 | Airbench-inspired residual CNN | 84.60% |
| V10 | Deep SE-ResNet trained on full data | 86.80% |
| V11 | Deep SE-ResNet + Alternating Flips | 86.90% |
| **V13** | **Pre-Activation Deep SE-ResNet + Fine-Tuning** | **87.00%** |

### Key Findings

**Depth versus width:** Increasing the number of residual blocks while controlling channel widths improved model performance without exceeding the parameter limit.

**Attention mechanisms:** SE attention improved our initial baseline, but the more complex CBAM architecture did not consistently perform better.

**Training duration:** Longer cosine training schedules generally produced stronger results in our later experiments.

**Full-data training:** Using all 50,000 labeled images improved our best public leaderboard results.

**Alternating flips:** Replacing random flips with alternating flips produced a small improvement in the full-data experiment.

**Pre-activation:** The final pre-activation model achieved our highest public score, although additional fine-tuning also contributed to the overall experimental change.

## 8. Visualizations

The repository includes scripts for plotting training accuracy, cross-entropy loss, data augmentation examples, and confusion matrices.

For example, training curves can be generated using:

```bash
python plot_resnet_curves.py \
  --name V13 \
  --history experiments/kaggle_v13_preact_alt_full50k/training_history.csv
```

The resulting plots show training accuracy approaching 99.60% and training loss decreasing toward approximately 0.51 during the first 250 epochs.

**Evaluation note:** V13 was trained using all 50,000 labeled images and therefore has no independent local validation set. Confusion matrices computed from its training images are training diagnostics, not held-out validation results. V12's independent validation results are reported separately.

## 9. Limitations

The final model's Kaggle public accuracy was 87.00%, but it did not have an independent local validation set because the complete labeled dataset was used for training.

Differences between validation accuracy and Kaggle accuracy in earlier experiments were not conclusively explained.

Additionally, the improvement from V11 to V13 was only 0.10 percentage points, and V13 also received 30 additional fine-tuning epochs. Therefore, the improvement cannot be attributed solely to the pre-activation architecture.

## 10. References

1. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. https://arxiv.org/abs/1512.03385
2. He et al., *Identity Mappings in Deep Residual Networks*, ECCV 2016. https://arxiv.org/abs/1603.05027
3. Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018. https://arxiv.org/abs/1709.01507
4. Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018. https://arxiv.org/abs/1807.06521
5. Kuang Liu, *pytorch-cifar*. https://github.com/kuangliu/pytorch-cifar
6. Keller Jordan, *94% on CIFAR-10 in 3.29 Seconds on a Single GPU*, 2024. https://arxiv.org/abs/2404.00498

## Acknowledgments

Developed as part of Deep Learning Project 1, Fall 2026.

The project draws on published work on residual learning, pre-activation ResNets, channel attention, and efficient CIFAR-10 training. All submitted models were trained from scratch without external pretrained weights.

# ViT-KNN: Semi-Supervised Pseudo-Labeling with Vision Transformers and KNN 

This repository contains the codebase developed by the **CUDA_Libre** team for the [Neural Wave Hackathon 2024](https://neuralwave.ch/#/2024/recap), where our solution earned **1<sup>st</sup> place**. The project automates the verification of steel bar alignment in a rolling mill using state-of-the-art Computer Vision models, combining semi-supervised Vision Transformers (ViT) and KNN-based pseudo-labeling. By enhancing operational efficiency and reducing human error, this system offers a scalable solution to modernize steel bar manufacturing processes.

 
## Problem Context

Fig. 1 depicts a sequence of steel bars moving towards a stopper on a rolling table. The goal is to assess whether the bars are properly aligned. Currently, this alignment check is performed manually by human operators who rely solely on visual inspection of real-time images. Determining alignment can be challenging due to uncertainties caused by various factors, including perspective distortions, vibrations, shadows, and inconsistent lighting conditions.

<figure style="text-align: center;">
  <img src="assets/sample_images.png" alt="Steel bar alignment process" width="1000" style="display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px;">Fig. 1 Sample images showing a sequence of aligned and not aligned bars on a rolling table approaching the stopper.</figcaption>
</figure>


## Methodology

### 1. DINOv2 KNN-based Pseudo-Labeling Workflow

Given the large, mostly unlabeled dataset of 15,630 images, we used an efficient approach to label the dataset, which combines human-labeling and pseudo-labeling.

- **Human Labeling**: We labeled manually an initial subset of 5,000 images, creating a foundation for reliable training and test data.

- **DINOv2 for Embeddings**: We used [DINOv2](https://arxiv.org/abs/2304.07193), a self-supervised vision transformer model, to generate high-dimensional embeddings of the images. These embeddings capture complex semantic features, without requiring any fine-tuning, that make it possible to measure image similarity effectively.

- **K-Nearest Neighbors (KNN) with FAISS**: We used [FAISS](https://github.com/facebookresearch/faiss) for fast, scalable similarity searches within the embedding space. For each unlabeled image, we identified its K-nearest neighbors and assigned a label based on a majority vote of their known labels, taken from the manually labeled dataset.

- **Cosine Similarity**: To ensure robust label assignment, we employ cosine similarity to compare image features and calculating "distances" in the KNN embedding space, with the following similarity function $m$:
```math
m(s, r) = \text{cosine-similarity} (f(s), f(r)) = \frac{f(s) \cdot f(r)}{\|f(s)\|_2 \|f(r)\|_2}
```
where $s$ and $r$ are a pair of images to compare and $f$ is the model generating features.
This method enabled us to expand the labeled dataset efficiently without manual effort for each image.
To run the Pseudo-Labeling check out the documentation: [DINOv2 KNN-based Pseudo-Labeling](dino/README.md).

### 2. Model Training, Inference and Results

The expanded dataset was used to train an [EfficientNet-B0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html) model, chosen for its balance of accuracy and computational efficiency.
We trained EfficientNet-B0 starting from the original weigths, adapting the classification layer for binary classification measuring of the alignment status. EfficientNet-B0 was also compared against the MobileNetV2 model.

- **Training Details**: The model was trained for 30 epochs, with the peak validation performance observed at epoch 10. Key performance metrics included:
  - **Accuracy**: 93.40%
  - **Precision**: 94.37%
  - **Recall**: 95.82%
  - **F1 Score**: 95.09%

- The model demonstrated reliable classification capabilities with a mean inference time on the test set of **0.0298 seconds per image**, meeting the real-time inference requirement of under 0.5 seconds per image.

<div align="center">

| **Inference Time Statistic** | **Time (seconds)** |
|------------------------------|---------------------|
| Mean Time                    | 0.0298              |
| 25th Percentile              | 0.0111              |
| Median (50th Percentile)     | 0.0117              |
| 75th Percentile              | 0.0128              |
</div>


## Run the Code

### Installation

Install the required packages with:
```bash
pip install -r requirements.txt
```

### Training

To perform the Pseudo-Labeling check out the documentation: [DINOv2 KNN-based Pseudo-Labeling](dino/README.md).
To train the EfficentNet-B0 model, run the training script:
```bash
python train.py \
    --data_config_path "dataset/augmented_split.json" \
    --batch_size 32 \
    --num_epochs 30 \
    --learning_rate 0.0001 \
    --checkpoint_path "checkpoints/efficient_net"
```

### Testing

Evaluate the model performance on the test set using:
```bash
python test.py \
    --data_config_path "dataset/split.json" \
    --batch_size 16 \
    --model_path "checkpoints/efficient_net/20241027_083453/model_epoch_10.pt"
```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

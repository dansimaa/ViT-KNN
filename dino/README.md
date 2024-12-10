## DINOv2 KNN-based Pseudo-Labeling

#### Build index to path and path to index mappings
```bash
DATA_DIR=/path/to/dataset

python dino/build_mappings.py \
    --data_dir $DATA_DIR \
    --output_dir "/dino/mappings"
```

#### Extract and save the DINOv2 embeddings and FAISS index:
```bash
python dino/extract_embedding.py \
    --data_config_path "dino/mappings/idx_to_imgpath.json" \
    --index_path "dino/indexes/vector.index"
```

#### KNN Pseudo-Labeling using the feature vectors extracted:
```bash
python dino/faiss_search.py \
    --dataset_config "dataset/split.json" \
    --k 3 \
    --output_path "dataset/pseudo_labeled_data.json"
```

#### Merge the pseudo labeled dataset to the manually labeled data:
```bash
python split_pseudo_labeled_data.py \
    --labeled_split_path "dataset/split.json" \
    --pseudo_labeled_data_path "dataset/pseudo_labeled_data.json" \ 
    --output_path "dataset/augmented_split.json"
```
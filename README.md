# Know What to Align: Learning Semantic Units for Cross-Modal Retrieval

## Code Files
We have provided the core code relevant to the methods.
- `casu_model.py`: ViT/BERT encoders, Describability Gating, Competitive Semantic Aggregation with a bias-free output projection, Local–Global Adaptive Matching, and the retrieval loss.
- `train.py`: Model configuration, training, validation, and checkpoint selection.
- `casu_data.py`: Image–text data loading, image preprocessing, BERT tokenization, and batch collation.

## Execution Environment

- Python 3.12
- PyTorch with CUDA, torchvision, Transformers, NumPy, and Pillow
- NVIDIA A800 GPU with 80 GB memory

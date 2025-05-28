# 🧬 Protein Sequence Classification with Deep Learning
This project explores protein function annotation using deep learning models—CNNs, LSTMs, and Transformers—on UniProt sequences. The task is framed as a multilabel classification problem over selected GO terms. Results show that CNN-based architectures outperform recurrent and transformer models in both training speed and accuracy.

### Highlights:

- Multilabel protein classification (4 GO terms)
- Custom sequence encoding pipeline
- Multiple architectures: CNN, Inception CNN, LSTM, Transformer
- Evaluation with AUC and accuracy
- Train/validation/test split with early stopping and performance tracking

### Best Performance (Test Set):
- Inception CNN: AUC = 0.808, Accuracy = 0.561, Loss = 0.462
- CNN: AUC = 0.784, Accuracy = 0.550, Loss = 0.498

Detailed results, figures, and discussion are available in `Report.pdf`.




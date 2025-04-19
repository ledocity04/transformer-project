import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

def plot_training_history(history, save_dir='data/figures'):
    """
    Vẽ biểu đồ lịch sử huấn luyện (loss và accuracy)
    
    Args:
        history: Đối tượng history từ quá trình huấn luyện
        save_dir: Thư mục lưu biểu đồ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Vẽ loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Vẽ accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    print(f"Đã lưu biểu đồ lịch sử huấn luyện vào {save_dir}/training_history.png")

def plot_confusion_matrix(y_true, y_pred, save_dir='data/figures'):
    """
    Vẽ confusion matrix
    
    Args:
        y_true: Nhãn thực tế
        y_pred: Nhãn dự đoán
        save_dir: Thư mục lưu biểu đồ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Đã lưu confusion matrix vào {save_dir}/confusion_matrix.png")
    
    # In classification report
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    print("\nClassification Report:")
    print(report)
    
    # Lưu classification report vào file
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

def plot_roc_curve(y_true, y_prob, save_dir='data/figures'):
    """
    Vẽ đường cong ROC
    
    Args:
        y_true: Nhãn thực tế
        y_prob: Xác suất dự đoán cho lớp dương
        save_dir: Thư mục lưu biểu đồ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Tính đường cong ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    print(f"Đã lưu đường cong ROC vào {save_dir}/roc_curve.png")
    print(f"AUC: {roc_auc:.4f}")

def plot_attention_weights(text, attention_weights, tokenizer, save_dir='data/figures', filename='attention_weights.png'):
    """
    Trực quan hóa attention weights
    
    Args:
        text: Văn bản đầu vào
        attention_weights: Ma trận attention weights
        tokenizer: Tokenizer đã sử dụng
        save_dir: Thư mục lưu biểu đồ
        filename: Tên file lưu biểu đồ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Tokenize văn bản
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    
    # Lấy attention weights trung bình từ tất cả các head
    if len(attention_weights.shape) > 2:
        # Nếu có nhiều head, lấy trung bình
        attention_weights = attention_weights.mean(axis=0)
    
    # Cắt ma trận attention weight theo số token thực tế
    attention_weights = attention_weights[:len(tokens), :len(tokens)]
    
    # Vẽ heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    
    print(f"Đã lưu biểu đồ attention weights vào {save_dir}/{filename}")
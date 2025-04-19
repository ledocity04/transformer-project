import os
import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.download_data import download_imdb_dataset
from src.data.preprocess import process_imdb_dataset
from src.data.dataset import load_and_prepare_data
from src.visualization.plots import plot_training_history, plot_confusion_matrix, plot_roc_curve

def train_model(train_loader, val_loader, model, optimizer, scheduler, device, num_epochs=4):
    """
    Huấn luyện mô hình Transformer
    
    Args:
        train_loader: DataLoader cho tập train
        val_loader: DataLoader cho tập validation
        model: Mô hình Transformer
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Thiết bị để huấn luyện
        num_epochs: Số epoch
        
    Returns:
        dict: Lịch sử huấn luyện
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Tính metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Cập nhật thanh tiến trình
            train_bar.set_postfix({'loss': f"{train_loss/len(train_bar):.4f}", 
                                  'acc': f"{train_correct/train_total:.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                # Tính metrics
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Cập nhật thanh tiến trình
                val_bar.set_postfix({'loss': f"{val_loss/len(val_bar):.4f}", 
                                    'acc': f"{val_correct/val_total:.4f}"})
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

def evaluate_model(test_loader, model, device):
    """
    Đánh giá mô hình trên tập test
    
    Args:
        test_loader: DataLoader cho tập test
        model: Mô hình Transformer
        device: Thiết bị để đánh giá
        
    Returns:
        tuple: (y_true, y_pred, y_prob) - Nhãn thực tế, nhãn dự đoán và xác suất
    """
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Lấy nhãn dự đoán và xác suất
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            # Thêm vào danh sách
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())  # Xác suất cho lớp positive
    
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def main(args):
    # Thiết lập các thư mục
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Tải dữ liệu
    if args.download:
        print("=== Bước 1: Tải dữ liệu IMDB ===")
        download_imdb_dataset()
    
    # Tiền xử lý dữ liệu
    if args.preprocess:
        print("\n=== Bước 2: Tiền xử lý dữ liệu ===")
        process_imdb_dataset('data/raw/IMDB Dataset.csv')
    
    # Huấn luyện mô hình
    if args.train:
        print("\n=== Bước 3: Huấn luyện mô hình ===")
        
        # Tải dữ liệu đã xử lý
        if not os.path.exists('data/processed/processed_data.pkl'):
            print("Không tìm thấy dữ liệu đã xử lý. Vui lòng chạy với tham số --preprocess trước.")
            return
        
        # Chuẩn bị dữ liệu
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        train_loader, val_loader, test_loader = load_and_prepare_data(
            'data/processed/processed_data.pkl',
            tokenizer_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # Thiết lập thiết bị
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {device}")
        
        # Tải mô hình
        model = BertForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=2
        ).to(device)
        
        # Thiết lập optimizer và scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
        
        # Huấn luyện mô hình
        history = train_model(
            train_loader, 
            val_loader, 
            model, 
            optimizer, 
            scheduler, 
            device, 
            num_epochs=args.num_epochs
        )
        
        # Lưu mô hình
        model_path = os.path.join('models', 'bert_sentiment_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Đã lưu mô hình vào {model_path}")
        
        # Lưu lịch sử huấn luyện
        with open('data/processed/training_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        # Vẽ biểu đồ lịch sử huấn luyện
        from src.visualization.plots import plot_training_history
        plot_training_history(history)
        
        # Đánh giá mô hình trên tập test
        print("\n=== Bước 4: Đánh giá mô hình ===")
        y_true, y_pred, y_prob = evaluate_model(test_loader, model, device)
        
        # Vẽ confusion matrix và ROC curve
        from src.visualization.plots import plot_confusion_matrix, plot_roc_curve
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_prob)
    
    # Dự đoán với mô hình đã huấn luyện
    if args.predict and args.text:
        print("\n=== Dự đoán văn bản ===")
        
        # Kiểm tra xem mô hình có tồn tại không
        model_path = os.path.join('models', 'bert_sentiment_model.pth')
        if not os.path.exists(model_path):
            print("Không tìm thấy mô hình đã huấn luyện. Vui lòng huấn luyện mô hình trước khi dự đoán.")
            return
        
        # Thiết lập thiết bị
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tải tokenizer và mô hình
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        model = BertForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=2
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Tokenize văn bản
        inputs = tokenizer(
            args.text,
            padding='max_length',
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt'
        ).to(device)
        
        # Dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
        
        # In kết quả
        sentiment = "Tích cực" if prediction == 1 else "Tiêu cực"
        confidence = probs[0][prediction].item()
        print(f"Văn bản: {args.text}")
        print(f"Dự đoán: {sentiment} (Độ tin cậy: {confidence:.4f})")
    
    print("\n=== Hoàn thành! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Classification với mô hình Transformer")
    
    # Các tham số chung
    parser.add_argument('--download', action='store_true', help='Tải dữ liệu IMDB')
    parser.add_argument('--preprocess', action='store_true', help='Tiền xử lý dữ liệu')
    parser.add_argument('--train', action='store_true', help='Huấn luyện mô hình')
    parser.add_argument('--predict', action='store_true', help='Dự đoán văn bản')
    
    # Các tham số cho huấn luyện
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Tên của pre-trained model')
    parser.add_argument('--batch_size', type=int, default=16, help='Kích thước batch')
    parser.add_argument('--max_length', type=int, default=256, help='Độ dài tối đa của văn bản')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Tốc độ học')
    parser.add_argument('--num_epochs', type=int, default=4, help='Số epoch')
    
    # Tham số cho dự đoán
    parser.add_argument('--text', type=str, help='Văn bản để dự đoán')
    
    args = parser.parse_args()
    
    # Nếu không có tham số nào được chỉ định, hiển thị help
    if not any(vars(args).values()):
        parser.print_help()
    else:
        main(args)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class IMDBDataset(Dataset):
    """
    Dataset cho bộ dữ liệu IMDB để sử dụng với mô hình Transformer
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Khởi tạo dataset
        
        Args:
            texts: Series chứa văn bản đã xử lý
            labels: Series chứa nhãn (0: negative, 1: positive)
            tokenizer: Tokenizer để chuyển đổi văn bản thành input IDs
            max_length: Độ dài tối đa của văn bản sau khi token hóa
        """
        self.texts = texts.values
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """Trả về số lượng mẫu trong dataset"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Trả về một mẫu dữ liệu đã được token hóa
        
        Args:
            idx: Chỉ số của mẫu cần lấy
            
        Returns:
            dict: Dictionary chứa input_ids, attention_mask và label
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Token hóa văn bản
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Loại bỏ chiều batch (batch_size=1)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(data_dict, tokenizer, batch_size=16, max_length=512):
    """
    Tạo DataLoader cho tập train, validation và test
    
    Args:
        data_dict: Dictionary chứa X_train, X_val, X_test, y_train, y_val, y_test
        tokenizer: Tokenizer để chuyển đổi văn bản
        batch_size: Kích thước batch
        max_length: Độ dài tối đa của văn bản
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Tạo các dataset
    train_dataset = IMDBDataset(data_dict['X_train'], data_dict['y_train'], tokenizer, max_length)
    val_dataset = IMDBDataset(data_dict['X_val'], data_dict['y_val'], tokenizer, max_length)
    test_dataset = IMDBDataset(data_dict['X_test'], data_dict['y_test'], tokenizer, max_length)
    
    # Tạo các dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def load_and_prepare_data(processed_data_path, tokenizer_name='bert-base-uncased', batch_size=16, max_length=256):
    """
    Đọc dữ liệu đã xử lý và chuẩn bị dataloader
    
    Args:
        processed_data_path: Đường dẫn đến file dữ liệu đã xử lý
        tokenizer_name: Tên tokenizer cần sử dụng
        batch_size: Kích thước batch
        max_length: Độ dài tối đa của văn bản
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    import pickle
    
    # Đọc dữ liệu đã xử lý
    with open(processed_data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Tải tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Tạo dataloader
    return create_data_loaders(data_dict, tokenizer, batch_size, max_length)

if __name__ == "__main__":
    # Kiểm tra nếu file được chạy trực tiếp
    print("Đang kiểm tra việc tải dữ liệu và tạo DataLoader...")
    
    try:
        # Đường dẫn đến dữ liệu đã xử lý
        processed_data_path = 'data/processed/processed_data.pkl'
        
        # Tải tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Thử tải dữ liệu và tạo DataLoader
        import os
        if os.path.exists(processed_data_path):
            train_loader, val_loader, test_loader = load_and_prepare_data(processed_data_path)
            
            # Kiểm tra một batch
            batch = next(iter(train_loader))
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Attention mask shape: {batch['attention_mask'].shape}")
            print(f"Labels shape: {batch['label'].shape}")
            print("DataLoader đã được tạo thành công!")
        else:
            print(f"Không tìm thấy file dữ liệu đã xử lý tại {processed_data_path}")
            print("Bạn cần chạy file preprocess.py trước để tạo dữ liệu đã xử lý.")
    except Exception as e:
        print(f"Lỗi khi kiểm tra DataLoader: {str(e)}")
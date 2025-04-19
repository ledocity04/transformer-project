import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def download_nltk_resources():
    """Tải xuống tài nguyên NLTK cần thiết"""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Không thể tải {resource}: {str(e)}")

def explore_data(df, text_column, label_column, save_dir='data/figures'):
    """Khám phá dữ liệu ban đầu và tạo các biểu đồ"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Kích thước dữ liệu: {df.shape}")
    print(f"\nCác cột trong dữ liệu: {df.columns.tolist()}")
    
    print(f"\nPhân bố nhãn:")
    print(df[label_column].value_counts())
    
    # Kiểm tra độ dài văn bản
    df['text_length'] = df[text_column].apply(len)
    
    print(f"\nThống kê độ dài văn bản:")
    print(df['text_length'].describe())
    
    # Vẽ phân bố nhãn
    plt.figure(figsize=(10, 6))
    sns.countplot(x=label_column, data=df)
    plt.title('Phân bố nhãn')
    plt.savefig(os.path.join(save_dir, 'label_distribution.png'))
    plt.close()
    
    # Vẽ phân bố độ dài văn bản
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', hue=label_column, bins=50, kde=True)
    plt.title('Phân bố độ dài văn bản')
    plt.xlabel('Độ dài (số ký tự)')
    plt.ylabel('Số lượng')
    plt.savefig(os.path.join(save_dir, 'text_length_distribution.png'))
    plt.close()
    
    # Vẽ boxplot theo nhãn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=label_column, y='text_length', data=df)
    plt.title('Độ dài văn bản theo nhãn')
    plt.savefig(os.path.join(save_dir, 'text_length_boxplot.png'))
    plt.close()
    
    # Xem vài mẫu dữ liệu
    print("\nVài mẫu dữ liệu:")
    print(df.head())

def clean_text(text):
    """Làm sạch văn bản"""
    # Chuyển đổi thành chữ thường
    text = text.lower()
    
    # Loại bỏ HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Loại bỏ các ký tự đặc biệt và số
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Loại bỏ khoảng trắng dư thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """Tiền xử lý văn bản"""
    # Làm sạch văn bản
    text = clean_text(text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Loại bỏ stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Ghép lại thành câu
    processed_text = ' '.join(tokens)
    
    return processed_text

def analyze_common_words(texts, top_n=20, save_dir='data/figures'):
    """Phân tích từ phổ biến"""
    # Gộp tất cả các từ
    all_words = ' '.join(texts).split()
    
    # Đếm tần suất từng từ
    word_freq = pd.Series(all_words).value_counts()
    
    print(f"\nTop {top_n} từ phổ biến nhất:")
    print(word_freq.head(top_n))
    
    # Vẽ biểu đồ
    plt.figure(figsize=(14, 8))
    word_freq.head(top_n).plot(kind='bar')
    plt.title(f'Top {top_n} từ phổ biến nhất')
    plt.xlabel('Từ')
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'top_words.png'))
    plt.close()
    
    return word_freq

def process_imdb_dataset(file_path, save_dir='data/processed', figure_dir='data/figures'):
    """Xử lý bộ dữ liệu IMDB và lưu kết quả"""
    # Tạo thư mục lưu kết quả
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    # Tải tài nguyên NLTK
    print("Đang tải tài nguyên NLTK...")
    download_nltk_resources()
    
    # Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df = pd.read_csv(file_path)
    
    # Khám phá dữ liệu
    print("\nĐang khám phá dữ liệu...")
    explore_data(df, 'review', 'sentiment', figure_dir)
    
    # Tiền xử lý văn bản
    print("\nĐang tiền xử lý văn bản...")
    tqdm.pandas(desc="Tiền xử lý")
    df['processed_review'] = df['review'].progress_apply(lambda x: preprocess_text(x))
    
    # Chuyển đổi nhãn thành số
    print("\nĐang chuyển đổi nhãn...")
    label_map = {'positive': 1, 'negative': 0}
    df['label'] = df['sentiment'].map(label_map)
    
    # Hiển thị vài mẫu đã xử lý
    print("\nDữ liệu sau khi xử lý:")
    print(df[['review', 'processed_review', 'sentiment', 'label']].head())
    
    # Phân tích từ phổ biến
    print("\nĐang phân tích từ phổ biến...")
    word_freq = analyze_common_words(df['processed_review'], top_n=20, save_dir=figure_dir)
    
    # Chia dữ liệu thành tập train, validation và test
    print("\nĐang chia dữ liệu...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df['processed_review'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=y_train_val
    )
    
    print(f"Số lượng mẫu train: {len(X_train)}")
    print(f"Số lượng mẫu validation: {len(X_val)}")
    print(f"Số lượng mẫu test: {len(X_test)}")
    
    # Lưu dữ liệu đã xử lý
    print("\nĐang lưu dữ liệu đã xử lý...")
    data_dict = {
        'X_train': X_train,
        'X_val': X_val, 
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    with open(os.path.join(save_dir, 'processed_data.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)
    
    # Lưu DataFrame đã xử lý
    df.to_csv(os.path.join(save_dir, 'processed_imdb.csv'), index=False)
    
    print(f"\nDữ liệu đã được xử lý và lưu vào {save_dir}")
    print(f"Biểu đồ phân tích đã được lưu vào {figure_dir}")
    
    return data_dict

if __name__ == "__main__":
    # Đường dẫn đến file dữ liệu
    file_path = 'data/raw/IMDB Dataset.csv'
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file dữ liệu tại {file_path}")
        print("Hãy chạy file download_data.py trước để tải dữ liệu.")
    else:
        # Xử lý dữ liệu
        process_imdb_dataset(file_path)
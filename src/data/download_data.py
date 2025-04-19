import os
import kaggle
import zipfile

def download_imdb_dataset(save_path='data/raw'):
    """Tải bộ dữ liệu IMDB từ Kaggle"""
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    os.makedirs(save_path, exist_ok=True)
    
    # Tên dataset trên Kaggle
    dataset_name = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    
    try:
        # Tải bộ dữ liệu
        print(f"Đang tải xuống bộ dữ liệu IMDB từ Kaggle...")
        kaggle.api.dataset_download_files(dataset_name, path=save_path)
        
        # Giải nén file zip
        zip_file_path = os.path.join(save_path, "imdb-dataset-of-50k-movie-reviews.zip")
        if os.path.exists(zip_file_path):
            print(f"Đang giải nén file...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(save_path)
            print(f"Đã giải nén thành công vào {save_path}")
            
            # Kiểm tra file CSV đã tồn tại
            csv_path = os.path.join(save_path, "IMDB Dataset.csv")
            if os.path.exists(csv_path):
                print(f"Bộ dữ liệu đã sẵn sàng tại: {csv_path}")
            else:
                print("Không tìm thấy file CSV sau khi giải nén!")
        else:
            print("Không tìm thấy file zip sau khi tải xuống!")
    
    except Exception as e:
        print(f"Lỗi khi tải xuống bộ dữ liệu: {str(e)}")

if __name__ == "__main__":
    # Tải bộ dữ liệu
    download_imdb_dataset()
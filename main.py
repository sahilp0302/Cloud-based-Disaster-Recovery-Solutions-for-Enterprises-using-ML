import os
import kagglehub

# Optional: Manually set kaggle.json path if not in default location
# os.environ["KAGGLE_CONFIG_DIR"] = "/absolute/path/to/.kaggle"

dataset_id = "katehighnam/beth-dataset"

def download_dataset():
    try:
        print("📦 Downloading dataset from KaggleHub...")
        path = kagglehub.dataset_download(dataset_id)
        print("✅ Dataset downloaded successfully!")
        print("📁 Path to dataset files:", path)
    except Exception as e:
        print("❌ Failed to download dataset.")
        print("Error:", e)

if __name__ == "__main__":
    download_dataset()

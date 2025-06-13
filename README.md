# Beth Dataset Downloader

This project uses `kagglehub` to download the BETH dataset from Kaggle.

## Steps

1. Clone this repo or download files
2. Place your `kaggle.json` API key inside the default `.kaggle` directory:
   - Windows: `C:\Users\<User>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

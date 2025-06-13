import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Step 0: Create output folder
# --------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Step 1: Auto-detect dataset file
# --------------------------
base_dir = r"C:\Users\Sahil Parab\.cache\kagglehub\datasets\katehighnam\beth-dataset\versions\3"

dataset_file = None
for file in os.listdir(base_dir):
    if file.endswith('.csv') or file.endswith('.xlsx'):
        dataset_file = os.path.join(base_dir, file)
        break

if not dataset_file:
    raise FileNotFoundError("❌ No CSV or Excel file found in the specified directory.")

print(f"📄 Dataset file found: {dataset_file}")

# --------------------------
# Step 2: Load dataset
# --------------------------
if dataset_file.endswith('.csv'):
    df = pd.read_csv(dataset_file)
elif dataset_file.endswith('.xlsx'):
    df = pd.read_excel(dataset_file)
else:
    raise ValueError("Unsupported file format.")

# --------------------------
# Step 3: Inspect Data and Save Logs
# --------------------------
# Save first few rows and info into a text log
log_file = os.path.join(output_dir, "data_overview.txt")
with open(log_file, "w", encoding="utf-8") as f:
    f.write("🔍 First 5 rows:\n")
    f.write(df.head().to_string())
    f.write("\n\n🧠 Dataset Info:\n")
    df.info(buf=f)
    f.write("\n\n🧾 Columns:\n")
    f.write(str(df.columns.tolist()))

print(f"📝 Dataset insights saved to {log_file}")

# --------------------------
# Step 4: Visualizations
# --------------------------
sns.set(style="whitegrid")

# 1. Distribution of 'sus' values
plt.figure(figsize=(6, 4))
sns.countplot(x='sus', data=df)
plt.title("Suspicious Traffic (sus)")
plt.xlabel("Suspicious (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sus_distribution.png"))
plt.close()

# 2. Distribution of 'evil' values
plt.figure(figsize=(6, 4))
sns.countplot(x='evil', data=df)
plt.title("Malicious Traffic (evil)")
plt.xlabel("Malicious (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evil_distribution.png"))
plt.close()

# 3. Top 10 DNS Queries
plt.figure(figsize=(10, 6))
top_queries = df['DnsQuery'].value_counts().nlargest(10)
sns.barplot(x=top_queries.values, y=top_queries.index, palette='viridis')
plt.title("Top 10 DNS Queries")
plt.xlabel("Count")
plt.ylabel("DNS Query")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_dns_queries.png"))
plt.close()

print(f"📊 Visualizations saved to folder: {output_dir}")

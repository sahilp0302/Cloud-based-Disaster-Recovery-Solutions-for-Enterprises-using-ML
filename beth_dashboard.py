import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.exceptions import UndefinedMetricWarning
from scipy import stats
from fpdf import FPDF
import warnings

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="DNS Analytics Dashboard", page_icon="🛡️", layout="wide")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --------------------------
# Constants
# --------------------------
DATA_PATH = r"C:\Users\Sahil Parab\.cache\kagglehub\datasets\katehighnam\beth-dataset\versions\3\labelled_2021may-ip-10-100-1-105-dns.csv"
PLOT_DIR = "outputs"
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------------
# Load Dataset
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# --------------------------
# Logo and Title
# --------------------------
with st.container():
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Security_icon.svg/1024px-Security_icon.svg.png", width=100)
    st.markdown("""
        <style>
            .main-title {
                font-size: 40px;
                font-weight: bold;
                color: #ffffff;
                text-align: center;
                background: linear-gradient(90deg, #2193b0, #6dd5ed);
                padding: 15px;
                border-radius: 12px;
            }
        </style>
        <div class="main-title">🛡️ DNS Traffic Analysis & Threat Detection Dashboard</div>
    """, unsafe_allow_html=True)

# --------------------------
# Sidebar Navigation (No page_link used to avoid error)
# --------------------------
st.sidebar.title("Navigation")
st.sidebar.info("Use the sections below to explore the dashboard features.")

# --------------------------
# Dataset Overview
# --------------------------
with st.expander("📄 Dataset Overview", expanded=False):
    st.dataframe(df.head(), height=200)
    st.markdown(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")

# --------------------------
# Correlation Heatmap
# --------------------------
st.subheader("🔗 Correlation Heatmap")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
correlation = df[numeric_cols].corr()
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# --------------------------
# Normalize Data
# --------------------------
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# --------------------------
# Comparison of Data Before and After Normalization
# --------------------------
st.subheader("📊 Feature Distribution Before & After Normalization")
feature_to_plot = st.selectbox("Select a numeric feature to compare", numeric_cols)
fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df[feature_to_plot], kde=True, ax=ax1, color="skyblue")
ax1.set_title(f"Original: {feature_to_plot}")
sns.histplot(df_normalized[feature_to_plot], kde=True, ax=ax2, color="salmon")
ax2.set_title(f"Normalized: {feature_to_plot}")
st.pyplot(fig_compare)

# --------------------------
# Train/Test Split
# --------------------------
st.subheader("🎯 Choose Prediction Target")
target = st.radio("Choose target", ['sus', 'evil'], horizontal=True)
X = df_normalized[numeric_cols].drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --------------------------
# Train ML Model
# --------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# --------------------------
# Feature Importance
# --------------------------
st.subheader("🌟 Feature Importance")
importances = clf.feature_importances_
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax_imp, palette='mako')
st.pyplot(fig_imp)

# --------------------------
# Confusion Matrix
# --------------------------
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# --------------------------
# ROC Curve
# --------------------------
st.subheader("📉 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % roc_auc)
ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic')
ax_roc.legend(loc='lower right')
st.pyplot(fig_roc)

# --------------------------
# Classification Report
# --------------------------
st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))

# --------------------------
# Hypothesis Testing
# --------------------------
st.subheader("📌 Hypothesis Testing")
st.markdown("**Hypothesis 1:** Suspicious and malicious DNS queries have different TTL values.")
st.markdown("**Hypothesis 2:** The number of answers is significantly different for suspicious vs non-suspicious queries.")

h1_group1 = pd.to_numeric(df[df['sus'] == 1]['DnsAnswerTTL'], errors='coerce')
h1_group2 = pd.to_numeric(df[df['sus'] == 0]['DnsAnswerTTL'], errors='coerce')
ttest1 = stats.ttest_ind(h1_group1.dropna(), h1_group2.dropna(), nan_policy='omit')
st.markdown(f"**Hypothesis 1 Result:** p-value = {ttest1.pvalue:.5f} {'(Significant)' if ttest1.pvalue < 0.05 else '(Not Significant)'}")

h2_group1 = pd.to_numeric(df[df['sus'] == 1]['NumberOfAnswers'], errors='coerce')
h2_group2 = pd.to_numeric(df[df['sus'] == 0]['NumberOfAnswers'], errors='coerce')
ttest2 = stats.ttest_ind(h2_group1.dropna(), h2_group2.dropna(), nan_policy='omit')
st.markdown(f"**Hypothesis 2 Result:** p-value = {ttest2.pvalue:.5f} {'(Significant)' if ttest2.pvalue < 0.05 else '(Not Significant)'}")

better_hypo = "Hypothesis 1" if ttest1.pvalue < 0.05 and (ttest1.pvalue < ttest2.pvalue or ttest2.pvalue >= 0.05) else "Hypothesis 2"
st.success(f"📌 Based on analysis, the better hypothesis is: **{better_hypo}**")

# --------------------------
# Cost Analysis
# --------------------------
st.subheader("💰 Cost Analysis")
cost_per_false_negative = 1000
cost_per_false_positive = 100
fn = cm[1, 0] if cm.shape[0] > 1 else 0
fp = cm[0, 1] if cm.shape[1] > 1 else 0
total_cost = fn * cost_per_false_negative + fp * cost_per_false_positive
st.markdown(f"- **False Negatives (FN):** {fn} × Rs.{cost_per_false_negative} = Rs.{fn * cost_per_false_negative}")
st.markdown(f"- **False Positives (FP):** {fp} × Rs.{cost_per_false_positive} = Rs.{fp * cost_per_false_positive}")
st.markdown(f"### ✅ Estimated Total Risk Cost: Rs.{total_cost}")

# --------------------------
# Top Source IPs
# --------------------------
st.markdown("""
    <style>
        .insight-card {
            background-color: #f0f2f6;
            padding: 25px 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .insight-title {
            font-size: 22px;
            font-weight: bold;
            color: #0e1117;
            margin-bottom: 20px;
        }
    </style>
    <div class="insight-card">
        <div class="insight-title">🌐 Top Source IPs by DNS Activity</div>
""", unsafe_allow_html=True)

metric_option = st.selectbox("Select Metric", ['DnsQuery', 'NumberOfAnswers'])
top_n = st.slider("Select Top N IPs", min_value=5, max_value=30, value=10)

domain_plot = df.groupby('SourceIP')[metric_option].count().sort_values(ascending=False).head(top_n)
fig_domain, ax_domain = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("crest", len(domain_plot))
domain_plot.plot(kind='barh', ax=ax_domain, color=colors[::-1])
for index, value in enumerate(domain_plot):
    ax_domain.text(value + 0.5, index, str(value), va='center', fontsize=9)
ax_domain.set_xlabel("Count")
ax_domain.set_ylabel("Source IP")
ax_domain.set_title(f"Top {top_n} Source IPs by {metric_option}", fontsize=13)
plt.tight_layout()
st.pyplot(fig_domain)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Save Outputs
# --------------------------
for filename, data in zip(["cleaned_data.csv", "feature_importance.csv"], [df, feature_df]):
    filepath = os.path.join(PLOT_DIR, filename)
    if os.path.exists(filepath):
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(os.path.join(PLOT_DIR, f"{base}_{counter}{ext}")):
            counter += 1
        filepath = os.path.join(PLOT_DIR, f"{base}_{counter}{ext}")
    data.to_csv(filepath, index=False)

# --------------------------
# Export Report to PDF
# --------------------------
st.subheader("📄 Export PDF Report")
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(200, 10, "DNS Analytics Report", ln=True, align='C')
    def chapter_title(self, title):
        self.set_font("Arial", 'B', 10)
        self.cell(0, 10, title, ln=True, align='L')
    def chapter_body(self, body):
        self.set_font("Arial", '', 9)
        self.multi_cell(0, 8, body.replace('\u20b9', 'Rs.'))

pdf = PDF()
pdf.add_page()
pdf.chapter_title("Model Performance")
pdf.chapter_body(classification_report(y_test, y_pred))
pdf.chapter_title("Hypothesis Testing")
pdf.chapter_body(f"Hypothesis 1 p-value: {ttest1.pvalue:.5f}\nHypothesis 2 p-value: {ttest2.pvalue:.5f}\nSelected: {better_hypo}")
pdf.chapter_title("Cost Analysis")
pdf.chapter_body(f"FN: {fn} × Rs.{cost_per_false_negative} = Rs.{fn * cost_per_false_negative}\nFP: {fp} × Rs.{cost_per_false_positive} = Rs.{fp * cost_per_false_positive}\nTotal: Rs.{total_cost}")

pdf_path = os.path.join(PLOT_DIR, "dns_analytics_report.pdf")
pdf.output(pdf_path)
st.success(f"📄 PDF report saved to: `{pdf_path}`")
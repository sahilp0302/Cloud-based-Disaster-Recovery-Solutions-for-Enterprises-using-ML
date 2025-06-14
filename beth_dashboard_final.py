# ──────────────────── IMPORTS & CONFIG ──────────────────── #
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
import warnings
from PIL import Image

st.set_page_config(page_title="DNS Analytics Dashboard", page_icon="🛡️", layout="wide")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ──────────────────── LOAD DATA ──────────────────── #
DATA_PATH = r"C:\\Users\\Sahil Parab\\.cache\\kagglehub\\datasets\\katehighnam\\beth-dataset\\versions\\3\\labelled_2021may-ip-10-100-1-105-dns.csv"
image_path = "C:/Users/Sahil Parab/Cloud-based Disaster Recovery Solutions for Enterprises/beth-dataset-downloader/logo.jpg"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# ──────────────────── CUSTOM STYLES ──────────────────── #
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: rgba(30, 30, 30, 0.55) !important;
    backdrop-filter: blur(12px) saturate(180%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    padding: 2rem 1.5rem 1rem 1.5rem;
    border-radius: 0 25px 25px 0;
}

.sidebar-logo {
    display: block;
    margin: 0 auto 20px auto;
    width: 80px;
    border-radius: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
}
.sidebar-logo:hover {
    transform: scale(1.05);
}
.sidebar-title {
    font-size: 18px;
    text-align: center;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 20px;
    text-shadow: 0 2px 6px rgba(255, 255, 255, 0.2);
}
div[data-testid="stSidebar"] button {
    width: 100%;
    text-align: left;
    padding: 10px 16px;
    margin-bottom: 10px;
    border-radius: 12px;
    background-color: transparent;
    color: #e5e7eb;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
    border: none;
}
div[data-testid="stSidebar"] button:hover {
    background-color: rgba(255, 255, 255, 0.08);
    color: #38bdf8;
}
div[data-testid="stSidebar"] button:focus {
    color: #38bdf8 !important;
    background-color: rgba(255, 255, 255, 0.08);
}
</style>
""", unsafe_allow_html=True)

# ──────────────────── SIDEBAR NAVIGATION ──────────────────── #
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            logo = Image.open(image_path)
            st.image(logo, width=80)
        except:
            st.warning("Logo not found.")
    st.markdown("<div class='sidebar-title'>📁 Navigation</div>", unsafe_allow_html=True)
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Dataset Overview"
    nav_items = {
        "Dataset Overview": "📊",
        "Column Summary": "🔢",
        "Correlation Heatmap": "🔎",
        "Feature Normalization": "📈",
        "Model Training": "🎯",
        "Evaluation": "🧪",
        "Hypothesis Testing": "📊",
        "Cost Analysis": "💸",
        "Source IP Insights": "🌐"
    }
    for label, emoji in nav_items.items():
        if st.button(f"{emoji} {label}", key=label):
            st.session_state["current_page"] = label

# ──────────────────── HEADER ──────────────────── #
st.markdown("""
<div style="text-align:center; padding: 20px; background: linear-gradient(90deg, #00c6ff, #0072ff); color: white; border-radius: 18px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); animation: fadeInDown 1.2s ease-out;">
    <h1 style="margin:0;"> 🛡️ Cloud-based Disaster Recovery Solutions for Enterprises Dashboard </h1>
</div>
<style>
@keyframes fadeInDown {
    0% {opacity: 0; transform: translateY(-30px);}
    100% {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ──────────────────── PAGE LOGIC ──────────────────── #
selected_page = st.session_state["current_page"]

if selected_page == "Dataset Overview":
    st.subheader("📊 Dataset Overview")
    st.write("Display your dataset here.")

elif selected_page == "Column Summary":
    st.subheader("🔢 Column Summary")
    st.write("Summary statistics for each column.")

# ──────────────────── COLUMN SUMMARY INSIGHTS ──────────────────── #
st.markdown("---")
st.markdown("## 🧾 Column Summary Insights")
summary_tab = st.expander("Click to View Column-Wise Summary & Visualizations", expanded=False)

with summary_tab:
    st.markdown("### 📋 Dataset Description")
    st.dataframe(df.describe(include='all').transpose(), height=400)

    st.markdown("### 🧠 Feature Summaries")
    cols = df.columns.tolist()
    num_cols = 2
    for i in range(0, len(cols), num_cols):
        subcols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j >= len(cols): break
            col_name = cols[i + j]
            with subcols[j]:
                st.markdown(f"**🔹 {col_name}**")
                chart_type = st.selectbox(f"Chart Type for {col_name}", ("Auto", "Histogram", "Pie Chart", "Bar Chart"), key=f"chart_{col_name}")
                unique_vals = df[col_name].dropna().unique()

                if chart_type == "Auto":
                    if df[col_name].dtype in ['int64', 'float64']:
                        chart_type = "Histogram"
                    elif df[col_name].nunique() <= 10:
                        chart_type = "Pie Chart"
                    elif df[col_name].dtype == 'object' or df[col_name].nunique() < 20:
                        chart_type = "Bar Chart"
                    else:
                        st.info("Too many unique values to visualize.")
                        continue

                if chart_type == "Histogram":
                    fig, ax = plt.subplots()
                    sns.histplot(df[col_name].dropna(), kde=True, ax=ax, color='steelblue')
                    ax.set_title(f"{col_name} Distribution")
                    st.pyplot(fig)

                elif chart_type == "Pie Chart" and df[col_name].nunique() <= 10:
                    fig, ax = plt.subplots()
                    df[col_name].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                    ax.set_ylabel('')
                    ax.set_title(f"{col_name} Distribution")
                    st.pyplot(fig)

                elif chart_type == "Bar Chart" and (df[col_name].dtype == 'object' or df[col_name].nunique() < 20):
                    fig, ax = plt.subplots()
                    sns.barplot(x=df[col_name].value_counts().values[:10], y=df[col_name].value_counts().index[:10], ax=ax, palette='Blues_d')
                    ax.set_title(f"{col_name} Top Categories")
                    st.pyplot(fig)
                else:
                    st.info("Incompatible chart type.")

    st.markdown("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    st.markdown("### 📈 Scatter Plot")
    numeric_columns = numeric_df.columns.tolist()
    if len(numeric_columns) >= 2:
        x_axis = st.selectbox("X-axis:", numeric_columns, key="scatter_x")
        y_axis = st.selectbox("Y-axis:", numeric_columns, index=1, key="scatter_y")
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis], alpha=0.5, color='seagreen')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{x_axis} vs {y_axis}")
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns.")

# Keep your remaining sections (tabs, model, hypothesis testing, cost, etc.) as they are.



# TABS Layout
tabs = st.tabs(["📄 Dataset", "📈 Analytics", "🎯 Model", "📊 Evaluation", "🧪 Hypothesis", "💰 Cost", "🌐 Source IPs"])

# Dataset Tab
with tabs[0]:
    with st.expander("Dataset Overview", expanded=True):
        st.dataframe(df.head(), height=200)
        st.markdown(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")

# Analytics Tab
with tabs[1]:
    st.markdown("## 🔗 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation = df[numeric_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    st.markdown("## 📊 Feature Normalization")
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    feature_to_plot = st.selectbox("Select a numeric feature to compare", numeric_cols, help="Compare distribution before and after normalization")
    fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[feature_to_plot], kde=True, ax=ax1, color="skyblue")
    ax1.set_title(f"Original: {feature_to_plot}")
    sns.histplot(df_normalized[feature_to_plot], kde=True, ax=ax2, color="salmon")
    ax2.set_title(f"Normalized: {feature_to_plot}")
    st.pyplot(fig_compare)

# Model Tab
with tabs[2]:
    st.markdown("## 🎯 Model Training")
    target = st.radio("Choose prediction target", ['sus', 'evil'], horizontal=True)
    X = df_normalized[numeric_cols].drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    st.markdown("### 🌟 Feature Importance")
    importances = clf.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax_imp, palette='mako')
    st.pyplot(fig_imp)

# Evaluation Tab
with tabs[3]:
    st.markdown("## 🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    st.markdown("## 📉 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % roc_auc)
    ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)

    st.markdown("## 📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

# Hypothesis Testing
with tabs[4]:
    st.markdown("## 🧪 Hypothesis Testing")
    st.markdown("**H1:** Suspicious vs. non-suspicious TTL values differ")
    st.markdown("**H2:** Suspicious vs. non-suspicious number of answers differ")
    h1_group1 = pd.to_numeric(df[df['sus'] == 1]['DnsAnswerTTL'], errors='coerce')
    h1_group2 = pd.to_numeric(df[df['sus'] == 0]['DnsAnswerTTL'], errors='coerce')
    ttest1 = stats.ttest_ind(h1_group1.dropna(), h1_group2.dropna(), nan_policy='omit')
    st.markdown(f"**H1 Result:** p = {ttest1.pvalue:.5f} {'(Significant)' if ttest1.pvalue < 0.05 else '(Not Significant)'}")
    h2_group1 = pd.to_numeric(df[df['sus'] == 1]['NumberOfAnswers'], errors='coerce')
    h2_group2 = pd.to_numeric(df[df['sus'] == 0]['NumberOfAnswers'], errors='coerce')
    ttest2 = stats.ttest_ind(h2_group1.dropna(), h2_group2.dropna(), nan_policy='omit')
    st.markdown(f"**H2 Result:** p = {ttest2.pvalue:.5f} {'(Significant)' if ttest2.pvalue < 0.05 else '(Not Significant)'}")
    better_hypo = "Hypothesis 1" if ttest1.pvalue < 0.05 and (ttest1.pvalue < ttest2.pvalue or ttest2.pvalue >= 0.05) else "Hypothesis 2"
    st.success(f"Better supported: {better_hypo}")

# Cost Tab
with tabs[5]:
    st.markdown("## 💰 Cost Analysis")
    cost_per_false_negative = 1000
    cost_per_false_positive = 100
    fn = cm[1, 0] if cm.shape[0] > 1 else 0
    fp = cm[0, 1] if cm.shape[1] > 1 else 0
    total_cost = fn * cost_per_false_negative + fp * cost_per_false_positive
    st.markdown(f"**False Negatives:** Rs.{fn * cost_per_false_negative}")
    st.markdown(f"**False Positives:** Rs.{fp * cost_per_false_positive}")
    st.markdown(f"### ✅ Total Estimated Risk Cost: Rs.{total_cost}")

# Top Source IPs
with tabs[6]:
    st.markdown("""
        <div class="insight-card">
            <div class="insight-title">🌐 Top Source IPs by DNS Activity</div>
    """, unsafe_allow_html=True)
    metric_option = st.selectbox("Metric", ['DnsQuery', 'NumberOfAnswers'])
    top_n = st.slider("Top N IPs", min_value=5, max_value=30, value=10)
    domain_plot = df.groupby('SourceIP')[metric_option].count().sort_values(ascending=False).head(top_n)
    fig_domain, ax_domain = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("crest", len(domain_plot))
    domain_plot.plot(kind='barh', ax=ax_domain, color=colors[::-1])
    for index, value in enumerate(domain_plot):
        ax_domain.text(value + 0.5, index, str(value), va='center', fontsize=9)
    ax_domain.set_xlabel("Count")
    ax_domain.set_ylabel("Source IP")
    ax_domain.set_title(f"Top {top_n} IPs by {metric_option}", fontsize=13)
    st.pyplot(fig_domain)
    st.markdown("</div>", unsafe_allow_html=True)
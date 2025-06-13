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
import base64 

# Set up Streamlit configuration
st.set_page_config(page_title="DNS Analytics Dashboard", page_icon="🛡️", layout="wide")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Load dataset
DATA_PATH = r"C:\\Users\\Sahil Parab\\.cache\\kagglehub\\datasets\\katehighnam\\beth-dataset\\versions\\3\\labelled_2021may-ip-10-100-1-105-dns.csv"
PLOT_DIR = "outputs"
os.makedirs(PLOT_DIR, exist_ok=True)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# CSS styles for neumorphism + modern sidebar
st.markdown("""
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.2);
        padding: 2rem 1rem 1rem 1rem;
    }
    
    .sidebar-logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        background: linear-gradient(90deg, #2193b0, #6dd5ed);
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        transition: all 0.5s ease-in-out;
        animation: fadeInDown 1s ease-out;
    }
    .main-title:hover {
        transform: scale(1.02);
    }
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
""", unsafe_allow_html=True)

# Add logo in sidebar
image_path = "C:/Users/Sahil Parab/Cloud-based Disaster Recovery Solutions for Enterprises/beth-dataset-downloader/logo.jpg"
if os.path.exists(image_path):
    with open(image_path, "rb") as f:
        logo_data = f.read()
    encoded_logo = base64.b64encode(logo_data).decode()
    st.sidebar.markdown(
        f"""
        <img src="data:image/jpeg;base64,{encoded_logo}" class="sidebar-logo"/>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.warning("Logo not found in sidebar.")

# Sidebar Navigation Info
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- Dataset Overview
- Column Summary
- Correlation Heatmap
- Feature Normalization
- Model Training
- Evaluation
- Hypothesis Testing
- Cost Analysis
- Source IP Insights
""")

# Title
st.markdown("<div class='main-title'>🛡️ Cloud-based Disaster Recovery Solutions for Enterprises Dashboard</div>", unsafe_allow_html=True)

# The rest of your app logic should continue here...
# Add tabs, visualizations, and interaction just as in your previous code

# This completes only the requested change. Let me know if you'd like me to restore the rest of the app under this structure.

# Additional Summary Tab
st.markdown("---")
st.markdown("## 🧾 Column Summary Insights")
summary_tab = st.expander("Click to View Column-Wise Summary & Visualizations")

with summary_tab:
    st.markdown("### 📋 Dataset Description")
    st.dataframe(df.describe(include='all').transpose(), height=400)

    st.markdown("### 📊 Graphical Summary of Features")
    for col in df.columns:
        st.markdown(f"#### 🔹 {col}")
        chart_type = st.selectbox(
            f"Choose chart type for {col}:",
            ("Auto", "Histogram", "Pie Chart", "Bar Chart"),
            key=f"chart_{col}"
        )

        unique_vals = df[col].dropna().unique()

        if chart_type == "Auto":
            if df[col].dtype in ['int64', 'float64']:
                chart_type = "Histogram"
            elif df[col].nunique() <= 10:
                chart_type = "Pie Chart"
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                chart_type = "Bar Chart"
            else:
                st.info("Too many unique values to visualize effectively.")
                continue

        if chart_type == "Histogram":
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='steelblue')
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

        elif chart_type == "Pie Chart" and df[col].nunique() <= 10:
            fig, ax = plt.subplots()
            df[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            ax.set_title(f"Pie Chart of {col}")
            st.pyplot(fig)

        elif chart_type == "Bar Chart" and (df[col].dtype == 'object' or df[col].nunique() < 20):
            top_vals = df[col].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_vals.values, y=top_vals.index, palette='Blues_d', ax=ax)
            ax.set_title(f"Top Categories in {col}")
            st.pyplot(fig)

        else:
            st.info("Selected chart type is not suitable for this column.")

    # Correlation Heatmap
    st.markdown("### 📌 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    # Scatter Plot Selection
    st.markdown("### 📈 Scatter Plot Viewer")
    numeric_columns = numeric_df.columns.tolist()
    if len(numeric_columns) >= 2:
        x_axis = st.selectbox("Select X-axis:", numeric_columns, key="scatter_x")
        y_axis = st.selectbox("Select Y-axis:", numeric_columns, index=1, key="scatter_y")
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis], alpha=0.5, color='seagreen')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns to plot scatter plot.")



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
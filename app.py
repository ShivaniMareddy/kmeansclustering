import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🟢",
    layout="wide"
)

# ------------------------------------------------
# LOAD CSS (same file)
# ------------------------------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Wholesale customers data.csv")
    return df

dataset = load_data()

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.markdown(
    """
    <div class="title-box">
        <h1>🟢 Customer Segmentation Dashboard</h1>
        <p>
            This system uses <b>K-Means Clustering</b> to group customers
            based on similarities in purchasing behavior.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Clustering Controls")

    numeric_cols = dataset.select_dtypes(include=np.number).columns.tolist()

    feature_1 = st.selectbox("Select Feature 1", numeric_cols)
    feature_2 = st.selectbox("Select Feature 2", numeric_cols, index=1)

    k = st.slider("Number of Clusters (K)", 2, 10, 3)

    random_state = st.number_input(
        "Random State (optional)",
        min_value=0,
        value=42
    )

    run_btn = st.button("🟦 Run Clustering")

# ------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------
if run_btn:

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    X = dataset[[feature_1, feature_2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # K-MEANS
    # -----------------------------
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state
    )

    clusters = kmeans.fit_predict(X_scaled)

    dataset["Cluster"] = clusters

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    st.markdown("## 📊 Customer Clusters Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=clusters,
        cmap="viridis",
        s=60
    )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=250,
        marker="X",
        label="Centroids"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("K-Means Customer Segmentation")

    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # CLUSTER SUMMARY
    # -----------------------------
    st.markdown("## 📋 Cluster Summary")

    summary = (
        dataset
        .groupby("Cluster")[[feature_1, feature_2]]
        .agg(["mean", "count"])
    )

    st.dataframe(summary)

    # -----------------------------
    # BUSINESS INTERPRETATION
    # -----------------------------
    st.markdown("## 💼 Business Interpretation")

    for i in range(k):
        st.info(
            f"Cluster {i}: Customers show similar purchasing behaviour "
            f"based on {feature_1} and {feature_2}. "
            f"This group can be targeted with customized offers."
        )

    # -----------------------------
    # USER GUIDANCE
    # -----------------------------
    st.markdown(
        """
        <div class="loan-section">
            <p>
            📌 Customers within the same cluster exhibit similar purchasing behaviour.
            These segments can be used for targeted marketing, inventory planning,
            and personalized business strategies.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("👈 Select features, choose K, and click **Run Clustering** to begin.")

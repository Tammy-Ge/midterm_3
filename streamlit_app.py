import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# 0. Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="College Major ROI Analytics", page_icon="🎓", layout="wide"
)

# Style
sns.set_theme(style="whitegrid")


# 1. load and process the data
@st.cache_data #streamlit dont need to reread the data when everytime it runs
def load_and_preprocess_data():
    df = pd.read_csv("college_major_roi.csv")
    # add a new column "roi_ratio"
    df["roi_ratio"] = np.where(
        df["net_cost_usd"] > 0, df["earnings_10yr_usd"] / df["net_cost_usd"], 0
    )
    return df


df = load_and_preprocess_data()

# ==============================================================================
# 1.Sidebar
# ==============================================================================
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    [
        "1. Society Case & Data Presentation",
        "2. Data Visualization & Insights",
        "3. Model Prediction & Decision Support",
    ],
)

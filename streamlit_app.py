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

# PAGE 1: Business Case & Data Presentation
if page == "1. Business Case & Data Presentation":
    st.title("💡 College Major ROI")
    st.subheader(
        "Understanding the Financial Return of Higher Education"
    )


    st.markdown("""
    ### Business Problem

    Choosing a college major is an important decision.
    Students need to consider future earnings, education costs,
    student debt, and other factors when evaluating their options.

    ### Project Goal

    Our goal is to explore which factors are associated with
    the financial return of a college major and use linear
    regression to predict a key financial outcome.

    ### Dataset Overview

    This dataset contains information about 30,000 graduates,
    including their majors, institution types, GPA, internship
    experience, education costs, debt, earnings, and ROI.
    """)

    st.markdown("---")
    st.subheader("📊 Key Metrics Overview")

    st.write(
        f"The dataset contains **{df.shape[0]:,}** graduate records across **{df.shape[1]}** features."
    )

    # Key Summary Statistics calculated using NumPy
    mean_earnings = np.mean(df["earnings_10yr_usd"])
    mean_roi = np.mean(df["net_roi_usd"])
    avg_cost = np.mean(df["net_cost_usd"])
    internship_rate = np.mean(df["had_internship"]) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg. 10-Year Earnings", f"${mean_earnings:,.0f}")
    c2.metric("Avg. 10-Year Net ROI", f"${mean_roi:,.0f}")
    c3.metric("Avg. Total Net Cost", f"${avg_cost:,.0f}")
    c4.metric("Internship Participation Rate", f"{internship_rate:.1f}%")

    st.markdown("##### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


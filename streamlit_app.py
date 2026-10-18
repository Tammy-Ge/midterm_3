## Step 00 - Import of the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="College Major ROI Explore 🎓",
    layout="centered",
    page_icon=" 🎓 ",
)


## Step 01 - Setup
st.sidebar.title("College Major ROI Explore")
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊","Prediction 📈"])

#st.video("video.mp4")

st.image("college.png")

st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("college_major_roi.csv")


## Step 02 - Load dataset
if page == "Introduction 📘":

    st.subheader("01 Introduction 📘")

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Visualization 📊":

    st.subheader("02 College Major ROI Data Viz")

    col_x = st.selectbox("Select X-axis variable", ["major", "major_category", "institution_tier", "region"])
    col_y = st.selectbox("Select Y-axis variable", ["net_roi_usd", "earnings_10yr_usd", "net_cost_usd", "debt_usd"])

    tab1, tab2, tab3 = st.tabs(["Bar Chart 📊", "Line Chart 📈", "Correlation Heatmap 🔥"])

    chart_data = df.groupby(col_x)[col_y].mean()

    with tab1:
        st.subheader("Bar Chart")
        st.bar_chart(chart_data, use_container_width=True)

    with tab2:
        st.subheader("Line Chart")
        st.line_chart(chart_data, use_container_width=True)

    with tab3:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)

        fig_corr, ax_corr = plt.subplots(figsize=(18, 14))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

elif page == "Prediction 📈":
    st.subheader("04 Prediction with Linear Regression")
    df2 = pd.read_csv("college_major_roi.csv")
    ## Data Preprocessing

    ### removing missing values 
    df2 = df2.dropna()

    ### Label Encoder to change text categories into number categories
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    df2["grad_id"] = le.fit_transform(df2["grad_id"])
    df2["major"] = le.fit_transform(df2["major"])
    df2["major_category"] = le.fit_transform(df2["major_category"])
    df2["institution_tier"] = le.fit_transform(df2["institution_tier"])
    df2["region"] = le.fit_transform(df2["region"])

    list_var = list(df2.columns)

    features_selection = st.sidebar.multiselect("Select Features (X)",list_var)
    target_selection  = st.sidebar.selectbox("Select target variable (Y))",list_var)
    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])

    ### i) X and y
    X = df2[features_selection]
    y = df2[target_selection]

    st.dataframe(X.head())
    st.dataframe(y.head())

    ### ii) train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    ## Model 

    ### i) Definition model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()

    ### ii) Training model
    model.fit(X_train,y_train)

    ### iii) Prediction
    predictions = model.predict(X_test)

    ### iv) Evaluation 
    from sklearn import metrics 
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")



    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
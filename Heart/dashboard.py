import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Heart Disease Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### Heart ]isease Insights Dashboard v1.0"
    }
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
    .stPlotlyChart {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    heart = pd.read_csv(r'C:\Users\YOUSSEF\Stroke-Detection\Heart\heart.csv')
    heart = heart.drop_duplicates()
    return heart

heart = load_data()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    # Age range filter
    age_range = st.slider("Select Age Range", 
                         min_value=int(heart['age'].min()), 
                         max_value=int(heart['age'].max()),
                         value=(29, 77),
                         help="Filter patients by age range")
    
    # target filter
    target_filter = st.radio("Heart Disease Status:",
                            options=["All", "Diseased (1)", "Not diseased (0)"],
                            index=0,
                            horizontal=True)
    
    # Gender filter
    gender_filter = st.radio("Gender:",
                            options=["All", "Female (0)", "Male (1)"],
                            index=0,
                            horizontal=True)
    
    # Additional filters
    with st.expander("Additional Filters"):
        chol_range = st.slider("Cholesterol Range (mg/dL):",
                              min_value=int(heart['cholestoral'].min()),
                              max_value=int(heart['cholestoral'].max()),
                              value=(126, 564))
        
        bp_range = st.slider("Resting Blood Pressure (mm Hg):",
                            min_value=int(heart['resting_bp'].min()),
                            max_value=int(heart['resting_bp'].max()),
                            value=(94, 200))
        
        heart_rate = st.slider("Max Heart Rate Achieved:",
                              min_value=int(heart['max_hr'].min()),
                              max_value=int(heart['max_hr'].max()),
                              value=(71, 202))

# Apply filters
filtered_heart = heart[
    (heart['age'] >= age_range[0]) & 
    (heart['age'] <= age_range[1]) &
    (heart['cholestoral'] >= chol_range[0]) &
    (heart['cholestoral'] <= chol_range[1]) &
    (heart['resting_bp'] >= bp_range[0]) &
    (heart['resting_bp'] <= bp_range[1]) &
    (heart['max_hr'] >= heart_rate[0]) &
    (heart['max_hr'] <= heart_rate[1])
]

if target_filter == "Diseased (1)":
    filtered_heart = filtered_heart[filtered_heart['target'] == 1]
elif target_filter == "Not diseased (0)":
    filtered_heart = filtered_heart[filtered_heart['target'] == 0]

if gender_filter == "Male (1)":
    filtered_heart = filtered_heart[filtered_heart['sex'] == 1]
elif gender_filter == "Female (0)":
    filtered_heart = filtered_heart[filtered_heart['sex'] == 0]

# ===== Main Dashboard =====
st.markdown("""
<h1 style='
    text-align: center; 
    font-weight: bold; 
    font-size: 4.2rem;
'>
    🫀 Heart Disease Analysis
</h1>
""", unsafe_allow_html=True)

# Real-time metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", len(filtered_heart), 
           delta=f"{len(filtered_heart)-len(heart)} vs total")
col2.metric("Average Age", 
           f"{np.mean(filtered_heart['age']):.1f} years",
           help="Average age of filtered patients")
col3.metric("Disease Prevalence", 
           f"{(len(filtered_heart[filtered_heart['target'] == 1])/len(filtered_heart)*100):.1f}%")
col4.metric("Avg Cholesterol", 
           f"{np.mean(filtered_heart['cholestoral']):.1f} mg/dL",
           delta=f"±{np.std(filtered_heart['cholestoral']):.1f} SD")

# tabs
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 30px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊 Distributions", 
    "💊 Relations",
    "🔗 Correlations"
])

attributes = [
    "age",
    "sex",
    "chest_pain_type", 
    "resting_bp", 
    "cholestoral", 
    "fasting_blood_sugar", 
    "restecg", 
    "max_hr", 
    "exang", 
    "oldpeak", 
    "slope", 
    "num_major_vessels", 
    "thal"
]

label = {
    'age': 'Age',
    'sex': 'Gender',
    'chest_pain_type': 'Chest Pain Type',
    'resting_bp': 'Resting Blood Pressure (mm Hg)',
    'cholestoral': 'Cholesterol (mg/dL)',
    'fasting_blood_sugar': 'Fasting Blood Sugar',
    'restecg': 'Resting Electrocardiographic Results',
    'max_hr': 'Max Heart Rate (bpm)',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'Old Peak',
    'slope': 'Slope of ST Segment',
    'num_major_vessels': 'Number of Major Vessels',
    'thal': 'Thalassemia',
    'target': 'Heart Disease Status'
}

def gender_color_discrete_sequence():
    if  gender_filter == "Male (1)":
        return ["#4169D0"] 
    elif  gender_filter == "Female (0)":
        return [ "#FF69c0"]
    else:
        return ["#4169D0", "#FF69c0"]
    
def target_color_discrete_sequence():
    if target_filter == "Diseased (1)":
        return ["#FFA7A7"]
    elif target_filter == "Not diseased (0)":
        return ["#A8E6CF"]
    else:
        return ["#FFA7A7", "#A8E6CF"]
    
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # age distribution
        st.subheader("Age Distribution Explorer")
        age_bins = st.slider("Number of Age Bins", 5, 50, 25)
        
        fig = px.histogram(
            filtered_heart, x="age", nbins=age_bins,
            color="sex", marginal="box",
            hover_data=filtered_heart.columns,
            barmode="overlay",
            opacity=0.8,
            color_discrete_sequence= gender_color_discrete_sequence(),
            labels={"sex": "Gender", "age": "Age"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # gender distribution
        st.subheader("Gender Distribution")
        show_pct = st.checkbox("Show percentages", value=True)
        fig = px.pie(filtered_heart, names='sex', hole=0.4,
                    color_discrete_sequence = gender_color_discrete_sequence())
        if show_pct:
            fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    #countplot for attributes with little unique values
    st.subheader("Count number of patients by attributes")
    attribute = ["sex", 
           "chest_pain_type",
           "fasting_blood_sugar",
           "restecg",
           "exang",
           "slope",
           "num_major_vessels",
           "thal"]
    
    hue = ["sex",
           "target",]

    col1, col2 = st.columns(2)
    with col1:
        col1.subheader("Select Feature")
        att = st.selectbox(
            "Select Attribute", attribute, index=0, key="attribute_select")

    with col2:    
        col2.subheader("Select Hue")
        h = st.selectbox(
            "Select Hue", hue, index=0, key="hue_select")

    fig = px.histogram(
            filtered_heart,
            x=att,
            color="target" if h == "target" else "sex",
            color_discrete_sequence=target_color_discrete_sequence() if h == "target" else gender_color_discrete_sequence(),
            title=f"{label[att]} Distribution",
            labels=label,
            barmode="group"
        )
    st.plotly_chart(fig, use_container_width=True)


    with st.expander("📌 More Distributions", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Cholesterol Distribution
            st.subheader("Cholesterol Distribution")
            fig = px.box(
                filtered_heart,
                x="target",
                y="cholestoral",
                color="sex",
                points="all",
                color_discrete_sequence=gender_color_discrete_sequence(),
                labels={'target': 'Heart Disease', 'cholestoral': 'Cholesterol (mg/dL)', 'sex':'Gender'},   
                title="Cholesterol Levels by Gender and Disease Status"
                )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Blood Pressure Distribution
            st.subheader("Blood Pressure Distribution")
            fig = px.box(
                filtered_heart,
                x="sex",
                y="resting_bp",
                color="sex",
                points="all",
                color_discrete_sequence=gender_color_discrete_sequence(),
                title="Blood Pressure by Gender",
                labels={'resting_bp': 'Blood Pressure (mm Hg)', 'sex': 'Gender'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Density Contour Plot
        st.subheader("Age vs Max Heart Rate Density")
        fig = px.density_contour(
            filtered_heart,
            x="age",
            y="max_hr",
            color="sex",
            marginal_x="rug",
            marginal_y="box",
            color_discrete_sequence=gender_color_discrete_sequence(),
            title="Age-Heart Rate Relationship",
            labels={'age': 'Age', 'max_hr': 'Max Heart Rate (bpm)', 'sex':'Gender'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Faceted Histograms
        st.subheader("Faceted Cholesterol Distribution")
        fig = px.histogram(
            filtered_heart,
            x="cholestoral",
            color="sex",
            facet_col="target",
            barmode="overlay",
            opacity=0.7,
            color_discrete_sequence=gender_color_discrete_sequence(),
            title="Cholesterol Distribution by Disease Status and Gender",
            labels={"target": "Heart Disease", "cholestoral": "Cholesterol (mg/dL)", 'sex': 'Gender'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:

    st.subheader('Medical Comparisons')

    compare_metric = st.selectbox(
        "Select  Attribute to Compare:",
        attributes
    )

    fig = px.box(
        filtered_heart, x="target", y=compare_metric,
        color="target", points="all",
        hover_data=["age", "sex"],
        title=f"{compare_metric.capitalize()} Distribution",
        labels=label,
        color_discrete_sequence=target_color_discrete_sequence()
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attributes Relationships")
    
    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox(
            "X-Axis", attributes, index=0, key="x_axis_select")
    with col2:
        y_axis = st.selectbox(
            "Y-Axis", attributes, index=1, key="y_axis_select")

    filtered_heart = filtered_heart.copy()
    filtered_heart['target'] = filtered_heart['target'].astype(str)

    fig = px.scatter(
        filtered_heart, 
        x=x_axis, 
        y=y_axis,
        color="target",
        size="age",
        hover_name="sex", 
        title=f"{label[x_axis]}  VS  {label[y_axis]}",
        color_discrete_sequence=target_color_discrete_sequence(), 
        labels=label
    )

    fig.update_layout(
        legend_title_text='Heart Disease Status',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.subheader("Attributes Correlation ")
    
    corr_matrix = filtered_heart.corr()
    features = st.multiselect(
        "Select Features to Correlate:",
        options=corr_matrix.columns,
        default=['target', 'age', 'cholestoral', 'max_hr']
    )
    
    fig = px.imshow(
        corr_matrix.loc[features, features],
        text_auto=True, aspect="auto",
        color_continuous_scale='reds',
        title="Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation with Target")

    correlation = heart.corr()["target"].drop("target").abs().sort_values(ascending=False)
    correlation_df = correlation.reset_index()
    correlation_df.columns = ["Feature", "Correlation"]

    features = st.multiselect(
        "Select Features to Correlate:",
        options=correlation_df["Feature"].tolist(),  
        default=['age', 'cholestoral', 'max_hr']
    )

    filtered_corr = correlation_df[correlation_df["Feature"].isin(features)]

    fig = px.bar(filtered_corr, 
                x="Feature", 
                y="Correlation",
                text=filtered_corr['Correlation'].apply(lambda x: f"{x:.3f}"),
                color="Correlation",
                color_continuous_scale="reds",
                title="Most Correlated Factors with Heart Disease")

    fig.update_layout(xaxis_title="Feature", yaxis_title="Correlation Strength", title_x=0.5, font_size=14)
    fig.update_traces(
        hovertemplate="Feature: %{x}<br>Correlation: %{y}",
        marker=dict(line=dict(color='#111', width=1))
    )
    st.plotly_chart(fig, use_container_width=True)


st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=filtered_heart.to_csv(index=False).encode('utf-8'),
    file_name='filtered_heart_data.csv',
    mime='text/csv'
)

st.sidebar.markdown("---")
theme = st.sidebar.selectbox("🎨 Color Theme", 
                            options=["Default", "Pastel", "Dark", "Vivid"])

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Stroke Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### Stroke Risk Insights Dashboard v1.0"
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


# # Load data
@st.cache_data
def load_data():
    stroke = pd.read_csv("Stroke/processed_data.csv")
    stroke = stroke.drop_duplicates()
    return stroke

stroke = load_data()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    # Age range filter
    age_range = st.slider("Select Age Range", 
                         min_value=int(stroke['age'].min()), 
                         max_value=int(stroke['age'].max()),
                         value =(8, 82))
    
    # Stroke filter
    stroke_filter = st.radio("Stroke Status:",
                            options=["All", "Stroke (1)", "No Stroke (0)"],
                            index=0,
                            horizontal=True)
    
    # Gender filter
    gender_filter = st.radio("Gender:",
                            options=["All", "Female (0)", "Male (1)"],
                            index=0,
                            horizontal=True)

    heart_filter = st.radio("Heart Disease:",
                            options=["All", "Diseased (1)", "Not diseased (0)"],
                            index=0,
                            horizontal=True)
    
    # Additional filters
    with st.expander("Advanced Filters"):
        glucose_range = st.slider("Glucose Level Range:",
                                min_value=int(stroke['avg_glucose_level'].min()),
                                max_value=int(stroke['avg_glucose_level'].max()),
                                value=(50, 300))
        
        bmi_range = st.slider("BMI Range:",
                            min_value=(int(stroke['bmi'].min())),
                            max_value=(int(stroke['bmi'].max())),
                            value=(10, 100))
        
# Apply filters
filtered_df = stroke[
    (stroke['age'] >= age_range[0]) & 
    (stroke['age'] <= age_range[1]) &
    (stroke['avg_glucose_level'] >= glucose_range[0]) &
    (stroke['avg_glucose_level'] <= glucose_range[1]) &
    (stroke['bmi'] >= bmi_range[0]) &
    (stroke['bmi'] <= bmi_range[1])
]

if stroke_filter == "Stroke (1)":
    filtered_df = filtered_df[filtered_df['stroke'] == 1]
elif stroke_filter == "No Stroke (0)":
    filtered_df = filtered_df[filtered_df['stroke'] == 0]

if gender_filter == "Male (1)":
    filtered_df = filtered_df[filtered_df['gender'] == 1]
elif gender_filter == "Female (0)":
    filtered_df = filtered_df[filtered_df['gender'] == 0]  

if heart_filter == "Diseased (1)":
    filtered_df = filtered_df[filtered_df['heart_disease'] == 1]
elif heart_filter == "Not diseased (0)":
    filtered_df = filtered_df[filtered_df['heart_disease'] == 0]
    



# ===== Main Dashboard =====

st.markdown("""
<h1 style='
    text-align: center; 
    font-weight: bold; 
    font-size: 4.2rem;
'>
    🧠 Stroke Analysis
</h1>
""", unsafe_allow_html=True)

# Real-time metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", len(filtered_df), 
           delta=f"{len(filtered_df)-len(stroke)} vs total")
col2.metric("Average Age", 
           f"{np.mean(filtered_df['age']):.1f} years")
col3.metric("Stroke Prevalence", 
           f"{(len(filtered_df[filtered_df['stroke'] == 1]))/len(filtered_df)*100 if len(filtered_df) > 0 else 0:.1f}%")
col4.metric("Avg Glucose Level", 
           f"{np.mean(filtered_df['avg_glucose_level']):.1f} mg/dL")


# Tabs
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
    "📈 Relationships",
    "🔗 Correlations"
])

attributes = [
    'bmi', 
    'gender', 
    'age', 
    'hypertension', 
    'heart_disease', 
    'ever_married', 
    'work_type', 
    'Residence_type',
    'avg_glucose_level',
    'smoking_status',
    'stroke',  
]

#labels
label = {
    'gender':'Gender', 
    'age':'Age', 
    'hypertension':'Hypertension', 
    'heart_disease':'Heart Disease', 
    'ever_married':'Married', 
    'work_type':'Work Type', 
    'Residence_type':'Residence Type',
    'avg_glucose_level':'Avg Glucose Level',
    'bmi':'Body Mass Index', 
    'smoking_status':'Smoking Status',
    'stroke':'Stroke',  
    }


def gender_color_discrete_sequence():
    if  gender_filter == "Male (1)":
        return ["#4169D0"] 
    elif  gender_filter == "Female (0)":
        return [ "#FF69c0"]
    else:
        return ["#4169D0", "#FF69c0"]
    
def heart_color_discrete_sequence():
    if heart_filter == "Diseased (1)":
        return ["#FFA7A7"]
    elif heart_filter == "Not diseased (0)":
        return ["#A8E6CF"]
    else:
        return ["#FFA7A7", "#A8E6CF"]

def stroke_color_discrete_sequence():
    if stroke_filter == "Stroke (1)":
        return ["#e5eb71"]
    elif stroke_filter == "No Stroke (0)":
        return ["#5681ba"]
    else:
        return ["#e5eb71", "#5681ba"]
    
with tab1:

    st.subheader("Age Distribution Explorer")
    age_bins = st.slider("Number of Age Bins", 5, 50, 25)
    
    fig = px.histogram(
        filtered_df, x="age", nbins=age_bins,
        color=filtered_df['gender'].map({0: 'Female', 1: 'Male'}),
        marginal="box",
        hover_data=filtered_df.columns,
        barmode="stack",
        opacity=0.8,
        color_discrete_sequence= gender_color_discrete_sequence(),
        labels=label,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        # gender distribution
        st.subheader("Gender Distribution")
        fig = px.pie(filtered_df, 
                    names=filtered_df['gender'].map({1: 'Male', 0: 'Female'}), 
                    hole=0.4,
                    color_discrete_sequence = ["#FF69c0", "#4169D0"] if gender_filter == 'All' else gender_color_discrete_sequence())
        st.plotly_chart(fig, use_container_width=True)        
    
    with col2:
        # stroke status distribution
        st.subheader("Stroke Status Distribution")
        fig = px.pie(filtered_df,
                     names=filtered_df['stroke'].map({1: 'Stroke', 0: 'NO Stroke'}),
                     hole = 0.4,
                     color_discrete_sequence= stroke_color_discrete_sequence())
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # heart disease distribution
        st.subheader('Heart Disease Distribution')
        fig = px.pie(filtered_df,
                     names=filtered_df['heart_disease'].map({1: 'Diseased', 0: 'Not Diseased'}),
                     hole= 0.4,
                     color_discrete_sequence= heart_color_discrete_sequence())
        st.plotly_chart(fig, use_container_width=True)
        
    #countplot for attributes with little unique values
    st.subheader("Count number of patients by attributes")
    attribute = [
        "gender", 
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",]
    
    hue = [
        "gender",
        "heart_disease",
        "stroke",]

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
            filtered_df,
            x=att,
            color="stroke" if h == "stroke" else ("gender" if h == 'gender' else "heart_disease"),
            color_discrete_sequence=stroke_color_discrete_sequence() if h == "stroke" else (gender_color_discrete_sequence() if h == 'gender' else heart_color_discrete_sequence()),
            title=f"{label[att]} Distribution",
            labels=label,
            barmode="group"
        )
    st.plotly_chart(fig, use_container_width=True)


    with st.expander("📌 More Distributions", expanded=True):

        col1, col2 = st.columns(2)
        
        with col1:
            # Ghucose Distribution
            st.subheader("Glucose Levels Distribution")
            fig = px.box(
                filtered_df,
                x="stroke",
                y="avg_glucose_level",
                color="gender",
                points="all",
                color_discrete_sequence=gender_color_discrete_sequence(),
                labels=label,   
                title="Average Glucose Levels by Gender and Stroke Status"
                )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # BMI Distribution
            st.subheader("Body Mass Index Distribution")
            fig = px.box(
                filtered_df,
                x="gender",
                y="bmi",
                color="gender",
                points="all",
                color_discrete_sequence=gender_color_discrete_sequence(),
                title="BMI by Gender",
                labels=label
                )
            st.plotly_chart(fig, use_container_width=True)

        # Density Contour Plot
        st.subheader("Age vs BMI Density")

        h2 = st.selectbox(
            "Select  Hue:",
            hue
        )
        
        fig = px.density_contour(
            filtered_df,
            x="age",
            y="bmi",
            color="stroke" if h2 == "stroke" else ("gender" if h2 == 'gender' else "heart_disease"),
            color_discrete_sequence=stroke_color_discrete_sequence() if h2 == "stroke" else (gender_color_discrete_sequence() if h2 == 'gender' else heart_color_discrete_sequence()),
            marginal_x="rug",
            marginal_y="box",
            title="Body Mass Index Relationship",
            labels=label
        )
        st.plotly_chart(fig, use_container_width=True)

        # Faceted Histograms
        st.subheader("Faceted Glucose Distribution")
        fig = px.histogram(
            filtered_df,
            x="avg_glucose_level",
            color="heart_disease",
            facet_col="stroke",
            barmode="stack",
            opacity=0.7,
            color_discrete_sequence=heart_color_discrete_sequence(),
            title="Glucose Distribution by Heart Disease Status and Stroke Status",
            labels=label)
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    
    st.subheader("Medical comparisons")

    compare = st.selectbox(
        "Select Attribute to Compare",
        attributes,
    )

    fig = px.box(
        filtered_df,
        x="stroke",
        y=compare,
        color="stroke",
        points="all",
        color_discrete_sequence=stroke_color_discrete_sequence(),
        title=f"{label[compare]} vs Stroke Status",
        labels=label
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Features Relationships')
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox(
            'Select X-Axis',
            attributes,
            index=0,
            key='x_axis_select'
        )  
    with col2:
        y_axis = st.selectbox(
            'Select Y-Axis',
            attributes,
            index=1,
            key='y_axis_select'
        )
    
    filtered_df = filtered_df.copy()
    filtered_df['stroke'] = filtered_df['stroke'].astype(str)

    fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color="stroke",
        size="age",
        hover_data=filtered_df.columns,
        color_discrete_sequence=stroke_color_discrete_sequence(),
        title=f"{label[x_axis]} VS {label[y_axis]}",
        labels=label,
    )
    fig.update_layout(
        legend_title_text="Stroke Status",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        )
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Correlation Analysis")
    
    # att = filtered_df[['age', 'avg_glucose_level', 'bmi', 'heart_disease', 'gender', 'hypertension']]
    corr_matrix = filtered_df.corr()
    features=st.multiselect(
        "Select Features to Include in Correlation Matrix",
        options=corr_matrix.columns,
        default=['age', 'avg_glucose_level', 'bmi'],
    )

    fig = px.imshow(
        corr_matrix.loc[features, features],
        text_auto=True,
        aspect="auto",
        color_continuous_scale='reds',
        title="Correlation Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Correlation with Stroke Status")

    if filtered_df['stroke'].nunique() < 2:
        st.warning('Cannot calculate correlation - stroke status is constant in current selection')
    else:
        correlation = filtered_df.corr()["stroke"].drop("stroke").abs().sort_values(ascending=False)


        correlation_df = correlation.reset_index()
        correlation_df.columns = ["Feature", "Correlation"]

        features = st.multiselect(
            "Select Features to Correlate:",
            options=correlation_df["Feature"].tolist(),  
            default=['age', 'bmi', 'avg_glucose_level', 'heart_disease']
        )

        filtered_corr = correlation_df[correlation_df["Feature"].isin(features)]

        fig = px.bar(filtered_corr, 
                    x="Feature", 
                    y="Correlation",
                    text=filtered_corr['Correlation'].apply(lambda x: f"{x:.3f}"),
                    color="Correlation",
                    color_continuous_scale="reds",
                    title="Most Correlated Factors with Heart Disease",
                    labels=label,
                    )

        fig.update_layout(xaxis_title="Feature", yaxis_title="Correlation Strength", title_x=0.5, font_size=14)
        fig.update_traces(
            hovertemplate="Feature: %{x}<br>Correlation: %{y}",
            marker=dict(line=dict(color='#111', width=1))
        )
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_stroke_data.csv',
    mime='text/csv'
)

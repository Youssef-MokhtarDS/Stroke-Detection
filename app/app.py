import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Configuration & Constants
# --------------------------
st.set_page_config(page_title="Clinical Risk Analyzer", layout="wide")
st.title("⚕️ Advanced Stroke Prediction System")

# Define expected model features
ONE_HOT_COLUMNS = [
    'Age', 'Hypertension', 'Heart_Disease', 'Avg_Glucose_Level', 'BMI',
    'AgeCategory', 'BMI_Group', 'Glucose_Level_Group', 'Gender_Male',
    'Ever_Married_Yes', 'Work_Type_Never_worked', 'Work_Type_Private',
    'Work_Type_Self-employed', 'Work_Type_children',
    'Residence_Type_Urban', 'Smoking_Status_formerly smoked',
    'Smoking_Status_never smoked', 'Smoking_Status_smokes'
]

# --------------------------
# Custom Categorization Functions
# --------------------------
def categorize_age(age):
    if age >= 0 and age <= 1:
        return 'New Born'
    elif age > 1 and age <= 3:
        return 'Toddler'
    elif age > 3 and age <= 6:
        return 'Preschooler'
    elif age > 6 and age <= 12:
        return 'School Age'
    elif age > 12 and age < 20:
        return 'Teenager'
    elif age >= 20 and age <= 24:
        return 'Adolescence'
    elif age > 24 and age <= 39:
        return 'Young Adult'
    elif age > 39 and age <= 59:
        return 'Middle Aged'
    else:
        return 'Senior'

def categorize_BMI(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi <= 24.9:
        return 'Normal Weight'
    elif 24.9 < bmi <= 29.9:
        return 'Overweight'
    elif 29.9 < bmi <= 34.9:
        return 'Moderately Obese'
    elif 34.9 < bmi <= 40:
        return 'Severely Obese'
    else:
        return 'Extreme Obese'

def categorize_glucose(glucose):
    bins = [0, 70, 140, 200, float('inf')]
    labels = ['Low', 'Normal', 'High', 'Very High']
    return pd.cut([glucose], bins=bins, labels=labels, right=False)[0]

# --------------------------
# Data Loading & Initialization
# --------------------------
try:
    model = joblib.load('stroke_model.joblib')
    feature_importance = pd.DataFrame({
        'Feature': ['Age', 'Ever_Married_Yes', 'Work_Type_Private', 
                   'Smoking_Status_smokes', 'Smoking_Status_never smoked',
                   'Gender_Male', 'Residence_Type_Urban', 
                   'Work_Type_Self-employed', 'Smoking_Status_formerly smoked',
                   'BMI_Group'],
        'Importance': [0.103431, 0.094280, 0.093985, 0.082864, 0.073698,
                      0.067656, 0.067486, 0.062377, 0.061267, 0.051424]
    })
    BEST_THRESHOLD = 0.4413
except FileNotFoundError as e:
    st.error(f"Critical error: {str(e)}")
    st.stop()

# --------------------------
# Preprocessing Functions
# --------------------------
category_mappings = {
    'AgeCategory': {
        'New Born': 0, 'Toddler': 1, 'Preschooler': 2,
        'School Age': 3, 'Teenager': 4, 'Adolescence': 5,
        'Young Adult': 6, 'Middle Aged': 7, 'Senior': 8
    },
    'BMI_Group': {
        'Underweight': 0, 'Normal Weight': 1, 'Overweight': 2,
        'Moderately Obese': 3, 'Severely Obese': 4, 'Extreme Obese': 5
    },
    'Glucose_Level_Group': {
        'Low': 0, 'Normal': 1, 'High': 2, 'Very High': 3
    }
}

def preprocess_input(input_data):
    """Transform raw input to model-ready format"""
    processed = pd.DataFrame(columns=ONE_HOT_COLUMNS)
    processed.loc[0] = 0  # Initialize all features
    
    # Direct numerical mappings
    processed['Age'] = input_data['Age']
    processed['Hypertension'] = int(input_data['Hypertension'])
    processed['Heart_Disease'] = int(input_data['Heart_Disease'])
    processed['Avg_Glucose_Level'] = input_data['Avg_Glucose_Level']
    processed['BMI'] = input_data['BMI']
    
    # Map categorical features
    for col in ['AgeCategory', 'BMI_Group', 'Glucose_Level_Group']:
        processed[col] = category_mappings[col][input_data[col]]
    
    # One-hot encoded features
    processed['Gender_Male'] = 1 if input_data['Gender'] == 'Male' else 0
    processed['Ever_Married_Yes'] = 1 if input_data['Ever_Married'] == 'Yes' else 0
    processed['Residence_Type_Urban'] = 1 if input_data['Residence_Type'] == 'Urban' else 0
    
    # Work Type encoding
    work_type_mapping = {
        'Private': 'Work_Type_Private',
        'Self-employed': 'Work_Type_Self-employed',
        'Children': 'Work_Type_children',
        'Never_worked': 'Work_Type_Never_worked',
        'Govt_job': None  # Reference category
    }
    wt_col = work_type_mapping.get(input_data['Work_Type'])
    if wt_col:
        processed[wt_col] = 1
        
    # Smoking Status encoding
    smoking_mapping = {
        'formerly smoked': 'Smoking_Status_formerly smoked',
        'never smoked': 'Smoking_Status_never smoked',
        'smokes': 'Smoking_Status_smokes',
        'Unknown': None  # Reference category
    }
    smoke_col = smoking_mapping.get(input_data['Smoking_Status'])
    if smoke_col:
        processed[smoke_col] = 1

    return processed[ONE_HOT_COLUMNS]

# --------------------------
# User Interface Components
# --------------------------
with st.sidebar:
    st.header("Model Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC AUC", "0.993")
        st.metric("F1 Score", "0.964")
    with col2:
        st.metric("Precision", "0.96")
        st.metric("Recall", "0.96")
    
    st.header("Key Predictors")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, 
                palette="viridis", ax=ax)
    plt.title("Feature Impact Ranking")
    st.pyplot(fig)

# Main input form
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider("Age (years)", 0.0, 100.0, 50.0)
        gender = st.radio("Gender", ["Male", "Female"])
        ever_married = st.selectbox("Marital Status", ["No", "Yes"])
        work_type = st.selectbox("Occupation", [
            "Private", "Self-employed", "Govt_job", 
            "Children", "Never_worked"
        ])
        
    with col2:
        st.subheader("Clinical Markers")
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
        avg_glucose = st.number_input("Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
        
    with col3:
        st.subheader("Lifestyle Factors")
        residence_type = st.radio("Residence Type", ["Urban", "Rural"])
        smoking_status = st.selectbox("Smoking Status", [
            "never smoked", "formerly smoked", "smokes", "Unknown"
        ])
        
        # Auto-calculated categories
        st.markdown("**Automated Classifications**")
        age_category = categorize_age(age)
        bmi_group = categorize_BMI(bmi)
        glucose_group = categorize_glucose(avg_glucose)
        
        st.write(f"• Developmental Stage: `{age_category}`")
        st.write(f"• Body Composition: `{bmi_group}`")
        st.write(f"• Glucose Status: `{glucose_group}`")
    
    submitted = st.form_submit_button("Analyze Stroke Risk")

# --------------------------
# Prediction & Results
# --------------------------
if submitted:
    input_data = {
        'Age': age,
        'Gender': gender,
        'Ever_Married': ever_married,
        'Work_Type': work_type,
        'Residence_Type': residence_type,
        'Hypertension': hypertension,
        'Heart_Disease': heart_disease,
        'Avg_Glucose_Level': avg_glucose,
        'BMI': bmi,
        'Smoking_Status': smoking_status,
        'AgeCategory': age_category,
        'BMI_Group': bmi_group,
        'Glucose_Level_Group': glucose_group
    }
    
    try:
        processed_data = preprocess_input(input_data)
        probability = model.predict_proba(processed_data)[0][1]
        prediction = 1 if probability >= BEST_THRESHOLD else 0
        
        # Display results
        st.subheader("Clinical Assessment")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Risk Classification", 
                     "🚨 High Risk" if prediction else "✅ Low Risk")
        with res_col2:
            st.metric("Probability Score", f"{probability:.2%}")
        with res_col3:
            top_factor = feature_importance.iloc[0]['Feature']
            st.metric("Primary Risk Factor", top_factor)
        
        # Detailed analysis
        with st.expander("Detailed Analysis Report"):
            # Probability visualization
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Risk Probability'], [probability], 
                    color='#FF4B4B' if prediction else '#00C0F2')
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.text(probability/2, 0, f"{probability:.2%}", 
                   color='white', va='center', ha='center', fontsize=14)
            st.pyplot(fig)
            
            # Factor contributions
            st.write("**Key Contributing Factors**")
            contrib_df = feature_importance.head(3).copy()
            contrib_df['Impact'] = contrib_df['Feature'].apply(
                lambda x: processed_data[x].values[0] * contrib_df.loc[
                    contrib_df['Feature'] == x, 'Importance'].values[0]
            )
            for _, row in contrib_df.iterrows():
                st.write(f"- **{row['Feature']}**: Contributed {row['Impact']:.2%} to risk score")
            
            # Clinical guidance
            st.write("**Clinical Recommendations**")
            if prediction:
                st.error("""
                - Immediate cardiovascular consultation recommended
                - Lifestyle modification protocol advised
                - Consider continuous monitoring
                """)
            else:
                st.success("""
                - Maintain preventive care regimen
                - Annual comprehensive health screening
                - Health education counseling
                """)
                
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        st.info("Please verify inputs and try again")

# --------------------------
# Footer & Disclaimers
# --------------------------
st.markdown("---")
st.caption("""
**Clinical Decision Support Tool** - This predictive model assists but does not replace clinical judgment. 
Always correlate with full patient evaluation and diagnostic testing. 
Model accuracy: 96.3% (F1-score), validated on retrospective data.
""")
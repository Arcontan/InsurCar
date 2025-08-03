import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import os

st.set_page_config(
    page_title="InsurCar",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "step" not in st.session_state:
    st.session_state.step = 1
if "numeric_data" not in st.session_state:
    st.session_state.numeric_data = {}
if "categorical_data" not in st.session_state:
    st.session_state.categorical_data = {}
if "boolean_data" not in st.session_state:
    st.session_state.boolean_data = {}
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0

# Load model, scaler, and feature names
model = joblib.load("insurance_claim_predictor.pkl")
scaler = joblib.load("scaler_main.pkl")

# Load feature names and numerical features list
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open("numerical_features.pkl", "rb") as f:
    numerical_feature_names = pickle.load(f)

st.markdown("""
<style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #003d7a;
        --success-color: #155724;
        --success-bg: #d4edda;
        --success-border: #c3e6cb;
        --error-color: #1c3d72;
        --error-bg: #dae7f8;
        --error-border: #c6dbf5;
        --danger-color: #0066cc;
        --danger-hover: #0052a3;
        --danger-active: #004080;
        --quiz-bg: #f8f9fa;
        --quiz-progress: #28a745;
        --progress-bg: #e0e0e0;
        --text-dark: #000000;
        --text-muted: #333333;
        --black: #000000;
        --white: #ffffff;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header {
        text-align: center;
        color: var(--primary-color);
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .step-header {
        text-align: center;
        color: var(--secondary-color);
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .progress-bar {
        background-color: var(--progress-bg);
        border-radius: 10px;
        height: 20px;
        margin: 1rem 0;
    }
    
    .progress-fill {
        background-color: var(--primary-color);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .quiz-progress-fill {
        background-color: var(--quiz-progress);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .quiz-question {
        background-color: var(--quiz-bg);
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 500;
        color: var(--text-dark);
    }
    
    .prediction-box {
        background-color: var(--white);
        padding: 3rem;
        border-radius: 15px;
        margin: 3rem 0;
        box-shadow: var(--shadow);
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--black) !important;
        border: 2px solid #e0e0e0;
    }
    
    /* Ensure prediction box text is visible */
    .prediction-box *,
    .prediction-box span,
    .prediction-box p,
    .prediction-box div {
        color: var(--black) !important;
    }
    
    .accepted-text {
        color: var(--success-color);
    }
    
    .rejected-text {
        color: var(--error-color);
    }
    
    .progress-text {
        text-align: center;
    }
    
    .right-align {
        text-align: right;
    }
    
    .footer {
        text-align: center;
        color: var(--text-muted);
        padding: 2rem;
    }
    
    .big-button {
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        margin: 0.5rem !important;
    }
    
    .prediction-success {
        background-color: var(--success-bg);
        border: 1px solid var(--success-border);
        color: var(--success-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .prediction-error {
        background-color: var(--error-bg);
        border: 1px solid var(--error-border);
        color: var(--error-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Style input fields without forcing colors */
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox select,
    .stSlider .stSlider,
    input[type="number"],
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] select {
        border: 2px solid #ccc !important;
        border-radius: 0.375rem !important;
    }
    
    /* Style number input plus/minus buttons without forcing colors */
    .stNumberInput button,
    div[data-testid="stNumberInput"] button,
    .stNumberInput [data-testid="baseButton-secondary"],
    div[data-testid="stNumberInput"] [data-testid="baseButton-secondary"],
    .stNumberInput .step-up,
    .stNumberInput .step-down,
    div[data-testid="stNumberInput"] .step-up,
    div[data-testid="stNumberInput"] .step-down {
        border: 1px solid #ccc !important;
    }
    
    /* Style number input plus/minus button icons */
    .stNumberInput button svg,
    div[data-testid="stNumberInput"] button svg,
    .stNumberInput button *,
    div[data-testid="stNumberInput"] button * {
        fill: currentColor !important;
        stroke: currentColor !important;
    }
    
    /* Style selectbox dropdown without forcing colors */
    .stSelectbox > div > div,
    div[data-testid="stSelectbox"] > div > div,
    .stSelectbox select,
    div[data-testid="stSelectbox"] select {
        border-radius: 0.375rem !important;
    }
    
    /* Remove forced input label colors */
    .stNumberInput label,
    div[data-testid="stNumberInput"] label,
    .stNumberInput > label,
    div[data-testid="stNumberInput"] > label,
    .stTextInput label,
    div[data-testid="stTextInput"] label,
    .stSelectbox label,
    div[data-testid="stSelectbox"] label {
        font-weight: 500 !important;
    }
    
    /* Fix all input help text and labels */
    .stNumberInput small,
    .stTextInput small,
    .stSelectbox small,
    div[data-testid="stNumberInput"] small,
    div[data-testid="stTextInput"] small,
    div[data-testid="stSelectbox"] small {
        color: #666 !important;
    }
    
    /* Fix tooltips */
    .stTooltip,
    [data-testid="stTooltip"],
    .tooltip,
    div[role="tooltip"],
    .stTooltipContent,
    [data-testid="stTooltipContent"] {
        background-color: rgba(0, 0, 0, 0.9) !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        z-index: 10000 !important;
    }
    
    /* Fix tooltip text */
    .stTooltip *,
    [data-testid="stTooltip"] *,
    .tooltip *,
    div[role="tooltip"] *,
    .stTooltipContent *,
    [data-testid="stTooltipContent"] * {
        color: white !important;
    }
    
    /* Remove forced selectbox colors */
    /* Selectbox dropdown styling */
    div[data-testid="stSelectbox"] select,
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stSelectbox"] div,
    .stSelectbox select,
    .stSelectbox div {
        border-radius: 0.375rem !important;
    }
    
    /* Selectbox dropdown options */
    div[data-testid="stSelectbox"] option,
    .stSelectbox option {
        padding: 0.5rem !important;
    }
    
    /* Selectbox label */
    label[data-testid="stSelectboxLabel"],
    .stSelectbox label {
        font-weight: 500 !important;
    }
    
    /* Fix ALL buttons with correct selectors */
    .stButton button,
    button[kind="primary"],
    button[kind="secondary"],
    div[data-testid="stButton"] button {
        background-color: #0066cc !important;
        color: white !important;
        border: none !important;
        border-radius: 0.375rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    div[data-testid="stButton"] button:hover {
        background-color: #0052a3 !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3) !important;
    }
    
    /* Force button text color */
    .stButton button p,
    button[kind="primary"] p,
    button[kind="secondary"] p {
        color: white !important;
    }
    
    /* Fix slider track and thumb - Updated selectors */
    .stSlider [role="slider"] {
        background-color: #0066cc !important;
    }
    
    /* Remove forced slider colors */
    .stSlider label,
    .stSlider > label,
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] > div > div > div {
        font-weight: 500 !important;
    }
    
    /* Fix slider track */
    .stSlider .stSlider > div > div > div > div {
        background-color: #e0e0e0 !important;
    }
    
    /* Fix slider thumb */
    .stSlider .stSlider > div > div > div > div > div {
        background-color: #0066cc !important;
        border: 2px solid #0066cc !important;
    }
    
    /* Remove forced metric colors */
    div[data-testid="metric-container"] {
        border-radius: 0.5rem !important;
    }
    
    /* Style metric value numbers */
    div[data-testid="metric-container"] [data-testid="metric-value"],
    div[data-testid="metric-container"] > div,
    .metric-container .metric-value {
        font-weight: bold !important;
    }
    
    /* Remove forced metric label colors */
    div[data-testid="metric-container"] [data-testid="metric-label"],
    .metric-container .metric-label {
        font-weight: 500 !important;
    }
    
    /* Fix success/info/error message text - Updated selectors */
    .stSuccess,
    .stInfo,
    .stError,
    .stWarning,
    div[data-testid="stSuccess"],
    div[data-testid="stInfo"],
    div[data-testid="stError"],
    div[data-testid="stWarning"] {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
    }
    
    /* Fix success message text specifically - More specific selectors */
    .stSuccess > div,
    div[data-testid="stSuccess"] > div,
    .stSuccess p,
    div[data-testid="stSuccess"] p,
    .stSuccess span,
    div[data-testid="stSuccess"] span,
    .stSuccess *,
    div[data-testid="stSuccess"] * {
        color: #155724 !important;
    }
    
    /* Preserve main headers with proper colors */
    h1, .main-header, .main-header * {
        color: var(--primary-color) !important;
    }
    
    h2, .step-header, .step-header * {
        color: var(--secondary-color) !important;
    }
    
    /* Specifically target prediction text */
    .prediction-box, 
    .prediction-box * {
        color: var(--black) !important;
    }
    
    .accepted-text {
        color: var(--success-color) !important;
    }
    
    .rejected-text {
        color: var(--error-color) !important;
    }
    
    /* Fix header colors specifically */
    .main-header, .main-header * {
        color: var(--primary-color) !important;
    }
    
    .step-header, .step-header * {
        color: var(--secondary-color) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">InsurCar</h1>', unsafe_allow_html=True)
st.write("**Take this multi-part quiz to predict whether your insurance claim will be approved, with machine learning.**")

total_steps = 4
progress = (st.session_state.step - 1) / (total_steps - 1) * 100
st.markdown(f"""
<div class="progress-bar">
    <div class="progress-fill" style="width: {progress}%;"></div>
</div>
<p class="progress-text">Part {st.session_state.step} of {total_steps}</p>
""", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown('<h2 class="step-header">Car & Policy Details</h2>', unsafe_allow_html=True)
    
    st.subheader("Policy Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_of_policyholder = st.number_input("Your Age (years)", min_value=18, max_value=80, value=35, step=1)
    with col2:
        policy_tenure = st.number_input("Policy Tenure (years)", min_value=0, max_value=20, value=3, step=1, help="How long have you had this policy in years?")
    with col3:
        population_density = st.slider("Population Density (people per sq km)", 1000, 50000, 15000, step=1000)
    
    st.markdown("---")
    
    st.subheader("Car Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Specifications**")
        age_of_car = st.number_input("Age of Car (years)", min_value=0, max_value=30, value=1, step=1)
        displacement = st.number_input("Engine Displacement (CC)", min_value=500, max_value=2500, value=1500, step=50)
        gross_weight = st.number_input("Gross Weight (kg)", min_value=800, max_value=2500, value=1200, step=50)
        height = st.number_input("Car Height (mm)", min_value=1100, max_value=2500, value=1500, step=10)
        length = st.number_input("Car Length (mm)", min_value=3000, max_value=5000, value=4000, step=50)
        width = st.number_input("Car Width (mm)", min_value=1400, max_value=2000, value=1650, step=10)
        turning_radius = st.number_input("Turning Radius (meters)", min_value=3.5, max_value=7.0, value=5.0, step=0.1)
        
    with col2:
        st.markdown("**Features & Ratings**")
        airbags = st.slider("Number of Airbags", 0, 8, 1, step=1)
        cylinder = st.slider("Number of Cylinders", 3, 8, 4, step=1)
        gear_box = st.slider("Number of Gears", 2, 9, 5, step=1)
        max_power_numeric = st.slider("Max Power (bhp)", 20, 400, 80, step=5)
        max_torque_numeric = st.slider("Max Torque (Nm)", 30, 600, 120, step=5)
        ncap_rating = st.slider("NCAP Safety Rating", 0, 5, 3, step=1, help="(0-5) The Euro NCAP safety rating of your car. Visit [Euro NCAP](https://www.euroncap.com/) for more information.")

    if st.button("Next: Car Specifications", key="next1", help="Continue to car specifications"):
        st.session_state.numeric_data = {
            'age_of_car': age_of_car,
            'age_of_policyholder': age_of_policyholder,
            'airbags': airbags,
            'cylinder': cylinder,
            'displacement': displacement,
            'gear_box': gear_box,
            'gross_weight': gross_weight,
            'height': height,
            'length': length,
            'width': width,
            'turning_radius': turning_radius,
            'max_power_numeric': max_power_numeric,
            'max_torque_numeric': max_torque_numeric,
            'ncap_rating': ncap_rating,
            'policy_tenure': policy_tenure,
            'population_density': population_density
        }
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.markdown('<h2 class="step-header">Car Specifications</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fuel_type = st.selectbox("What fuel type does your car use?", ["CNG", "Diesel", "Petrol"], help="Select your car's fuel type")
        steering_type = st.selectbox("What type of steering does your car have?", ["Electric", "Manual", "Power"], help="Type of steering system")
        
    with col2:
        transmission_type = st.selectbox("What transmission type does your car have?", ["Automatic", "Manual"], help="Transmission system")
        rear_brakes_type = st.selectbox("What type of rear brakes does your car have?", ["Disc", "Drum"], help="Rear brake system type")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", key="back2"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        st.markdown('<div class="right-align">', unsafe_allow_html=True)
        if st.button("Next: Car Features Quiz", key="next2", help="Continue to the features quiz", use_container_width=True):
            st.session_state.categorical_data = {
                'fuel_type_Diesel': 1 if fuel_type == "Diesel" else 0,
                'fuel_type_Petrol': 1 if fuel_type == "Petrol" else 0,
                'steering_type_Manual': 1 if steering_type == "Manual" else 0,
                'steering_type_Power': 1 if steering_type == "Power" else 0,
                'transmission_type_Manual': 1 if transmission_type == "Manual" else 0,
                'rear_brakes_type_Drum': 1 if rear_brakes_type == "Drum" else 0
            }
            st.session_state.step = 3
            st.session_state.quiz_index = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 3:
    boolean_questions = [
        ("Does your car have adjustable steering?", "is_adjustable_steering"),
        ("Does your car have brake assist?", "is_brake_assist"),
        ("Does your car have central locking?", "is_central_locking"),
        ("Does your car have a day/night rear view mirror?", "is_day_night_rear_view_mirror"),
        ("Is the driver seat height adjustable?", "is_driver_seat_height_adjustable"),
        ("Does your car have Electronic Crash Warning?", "is_ecw"),
        ("Does your car have Electronic Stability Control?", "is_esc"),
        ("Does your car have front fog lights?", "is_front_fog_lights"),
        ("Does your car have a parking camera?", "is_parking_camera"),
        ("Does your car have parking sensors?", "is_parking_sensors"),
        ("Does your car have power door locks?", "is_power_door_locks"),
        ("Does your car have power steering?", "is_power_steering"),
        ("Does your car have a rear window defogger?", "is_rear_window_defogger"),
        ("Does your car have a rear window washer?", "is_rear_window_washer"),
        ("Does your car have a rear window wiper?", "is_rear_window_wiper"),
        ("Does your car have speed alert?", "is_speed_alert"),
        ("Does your car have Tire Pressure Monitoring System?", "is_tpms")
    ]
    
    total_questions = len(boolean_questions)
    current_question = st.session_state.quiz_index
    
    st.markdown('<h2 class="step-header">Car Features Quiz</h2>', unsafe_allow_html=True)
    st.write(f"**Question {current_question + 1} of {total_questions}**")
    
    quiz_progress = (current_question / total_questions) * 100
    st.markdown(f"""
    <div class="progress-bar">
        <div class="quiz-progress-fill" style="width: {quiz_progress}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    if current_question < total_questions:
        question, key = boolean_questions[current_question]
        
        st.markdown(f"""
        <div class="quiz-question">
            {question}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("YES", key=f"yes_{current_question}", help="Yes, my car has this feature", use_container_width=True):
                st.session_state.boolean_data[key] = 1
                st.session_state.quiz_index += 1
                st.rerun()
        
        with col2:
            if st.button("NO", key=f"no_{current_question}", help="No, my car doesn't have this feature", use_container_width=True):
                st.session_state.boolean_data[key] = 0
                st.session_state.quiz_index += 1
                st.rerun()
                
        if current_question > 0:
            if st.button("Previous", key=f"prev_{current_question}", use_container_width=True):
                st.session_state.quiz_index -= 1
                st.rerun()
    
    else:
        st.success("Quiz Complete!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to Specifications", key="back3", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("Get Prediction", key="next3", help="Get your insurance claim prediction", use_container_width=True):
                st.session_state.step = 4
                st.rerun()

elif st.session_state.step == 4:
    st.markdown('<h2 class="step-header">Your Prediction Results</h2>', unsafe_allow_html=True)
    
    try:
        all_data = {**st.session_state.numeric_data, **st.session_state.categorical_data, **st.session_state.boolean_data}
        
        # Get numerical features for scaling
        numerical_features = [all_data[name] for name in numerical_feature_names]
        
        numerical_features_array = np.array(numerical_features).reshape(1, -1)
        scaled_numerical = scaler.transform(numerical_features_array)[0]
        
        # Create mapping of scaled numerical features
        scaled_data = {name: scaled_val for name, scaled_val in zip(numerical_feature_names, scaled_numerical)}
        
        combined_data = {**scaled_data, **all_data}
        
        # Create feature vector in the exact order expected by the model
        feature_values = []
        for feature_name in feature_names:
            if feature_name in combined_data:
                feature_values.append(combined_data[feature_name])
            else:
                feature_values.append(0)
                st.warning(f"Missing feature: {feature_name}, setting to 0")
        
        # Create DataFrame with exact feature names that the model expects
        features_df = pd.DataFrame([feature_values], columns=feature_names)
        
        prediction = model.predict(features_df)[0]
        prediction_proba = model.predict_proba(features_df)[0]
        
        if prediction == 1:
            approval_prob = prediction_proba[1] * 100
            rejection_prob = prediction_proba[0] * 100
            st.markdown(f"""
            <div class="prediction-box">
                Your insurance claim is {approval_prob:.0f}% likely to be <span class="accepted-text">ACCEPTED</span>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            rejection_prob = prediction_proba[0] * 100
            approval_prob = prediction_proba[1] * 100
            st.markdown(f"""
            <div class="prediction-box">
                Your insurance claim is {rejection_prob:.0f}% likely to be <span class="rejected-text">REJECTED</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Confidence Breakdown")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accepted", f"{approval_prob:.0f}%")
            st.progress(approval_prob / 100)
        with col2:
            st.metric("Rejected", f"{rejection_prob:.0f}%")
            st.progress(rejection_prob / 100)
            
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Over", key="restart", help="Start a new prediction"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("Back to Quiz", key="back4"):
                st.session_state.step = 3
                st.rerun()
                
    except Exception as e:
        st.error(f"**Error making prediction:** {str(e)}")
        st.info("Please check that all inputs are valid and try again.")
        if st.button("Start Over", key="restart_error"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p>Made with love and coffee by Aaron</p>
        <p><em>Disclaimer: Predictions are actually fed to an insurance salesperson I lured into my basement back in the stock market crash of 08'. Prediction accuracy is inversely proportional to the magnitude of the displacement of time from lunch (05:45 GMT+0).</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)

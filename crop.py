import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import os

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        with open('crop_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        return model, scaler
    
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first using the Jupyter notebook.")
        return None, None
    
model, scaler = load_models()

crop_info = {
    'rice': {'icon': '🍚', 'season': 'Kharif', 'duration': '3-6 months'},
    'maize': {'icon': '🌽', 'season': 'Kharif/Rabi', 'duration': '3-4 months'},
    'chickpea': {'icon': '🫘', 'season': 'Rabi', 'duration': '4-5 months'},
    'kidneybeans': {'icon': '🫘', 'season': 'Kharif', 'duration': '3-4 months'},
    'pigeonpeas': {'icon': '🫛', 'season': 'Kharif', 'duration': '5-6 months'},
    'mothbeans': {'icon': '🫘', 'season': 'Kharif', 'duration': '2-3 months'},
    'mungbean': {'icon': '🫛', 'season': 'Kharif/Summer', 'duration': '2-3 months'},
    'blackgram': {'icon': '⚫', 'season': 'Kharif/Rabi', 'duration': '2-3 months'},
    'lentil': {'icon': '🟤', 'season': 'Rabi', 'duration': '3-4 months'},
    'pomegranate': {'icon': '🍎', 'season': 'All year', 'duration': '6-7 months'},
    'banana': {'icon': '🍌', 'season': 'All year', 'duration': '9-12 months'},
    'mango': {'icon': '🥭', 'season': 'Summer', 'duration': '3-5 months'},
    'grapes': {'icon': '🍇', 'season': 'Summer', 'duration': '5-6 months'},
    'watermelon': {'icon': '🍉', 'season': 'Summer', 'duration': '2-3 months'},
    'muskmelon': {'icon': '🍈', 'season': 'Summer', 'duration': '2-3 months'},
    'apple': {'icon': '🍎', 'season': 'Winter', 'duration': '5-6 months'},
    'orange': {'icon': '🍊', 'season': 'Winter', 'duration': '6-8 months'},
    'papaya': {'icon': '🍈', 'season': 'All year', 'duration': '9-12 months'},
    'coconut': {'icon': '🥥', 'season': 'All year', 'duration': '12 months'},
    'cotton': {'icon': '☁️', 'season': 'Kharif', 'duration': '5-6 months'},
    'jute': {'icon': '🌾', 'season': 'Kharif', 'duration': '4-5 months'},
    'coffee': {'icon': '☕', 'season': 'All year', 'duration': '6-8 months'}
}

st.title("🌾 Crop Recommendation System")
st.markdown("""
This intelligent system recommends the best crop to grow based on soil nutrients and environmental conditions.
Enter the soil and climate parameters below to get personalized crop recommendations.
""")

if model is not None and scaler is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Soil Nutrients")
        nitrogen = st.slider("Nitrogen (N)", 0, 140, 50, help="Nitrogen content in soil (kg/ha)")
        phosphorus = st.slider("Phosphorus (P)", 5, 145, 50, help="Phosphorus content in soil (kg/ha)")
        potassium = st.slider("Potassium (K)", 5, 205, 50, help="Potassium content in soil (kg/ha)")
        ph = st.slider("pH Level", 3.5, 9.5, 6.5, 0.1, help="Soil pH level")
        
    with col2:
        st.subheader("🌤️ Climate Conditions")
        temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, 0.5, help="Average temperature")
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0, 0.5, help="Relative humidity")
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 1.0, help="Annual rainfall")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Get Crop Recommendation"):
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        
        
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100
        
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_crops = model.classes_[top_indices]
        top_probs = probabilities[top_indices] * 100
        
        st.markdown("---")
        st.subheader("📊 Recommendation Results")
        
        crop_icon = crop_info.get(prediction.lower(), {}).get('icon', '🌱')
        crop_season = crop_info.get(prediction.lower(), {}).get('season', 'N/A')
        crop_duration = crop_info.get(prediction.lower(), {}).get('duration', 'N/A')
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style='text-align: center;'>{crop_icon} {prediction.upper()}</h2>
            <h3 style='text-align: center; color: #4CAF50;'>Confidence: {confidence:.2f}%</h3>
            <p style='text-align: center;'><strong>Season:</strong> {crop_season} | <strong>Duration:</strong> {crop_duration}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Top 3 Recommendations")
        cols = st.columns(3)
        
        for idx, (col, crop, prob) in enumerate(zip(cols, top_crops, top_probs)):
            with col:
                c_icon = crop_info.get(crop.lower(), {}).get('icon', '🌱')
                c_season = crop_info.get(crop.lower(), {}).get('season', 'N/A')
                
                st.markdown(f"""
                <div style='text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h3>{c_icon}</h3>
                    <h4>{crop.upper()}</h4>
                    <p><strong>{prob:.1f}%</strong></p>
                    <p style='font-size: 12px;'>{c_season}</p>
                </div>
                """, unsafe_allow_html=True)
                
                
        st.markdown("---")
        st.subheader("📝 Input Summary")
        input_df = pd.DataFrame({
            'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
            'Value': [f"{nitrogen} kg/ha", f"{phosphorus} kg/ha", f"{potassium} kg/ha", 
                     f"{temperature}°C", f"{humidity}%", f"{ph}", f"{rainfall} mm"]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)
        
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system uses **Machine Learning** (Random Forest) to recommend crops based on:
    - **Soil Nutrients**: N, P, K levels and pH
    - **Climate**: Temperature, humidity, rainfall
    
    **How to use:**
    1. Adjust the sliders for your soil and climate conditions
    2. Click "Get Crop Recommendation"
    3. View the best crop to grow!
    """)
    
    st.markdown("---")
    st.header("📚 Tips")
    st.markdown("""
    - Get soil tested for accurate NPK values
    - Consider local climate patterns
    - Consult agricultural experts
    - Plan according to seasons
    """)
    
    st.markdown("---")
    st.markdown("**Made with ❤️ using Streamlit**")

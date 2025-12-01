# 🌾 Crop Recommendation System

An intelligent Machine Learning system that recommends the best crop to grow based on soil nutrients (N, P, K) and environmental conditions (temperature, humidity, pH, rainfall). Built with Random Forest algorithm achieving **99.55% accuracy**.

## 🎯 Project Overview

Agriculture is the backbone of many economies, and choosing the right crop is crucial for farmers' success. This ML-powered system helps farmers make data-driven decisions by analyzing soil and climate conditions to recommend the most suitable crop.

### Key Features

- 🎯 **High Accuracy**: 99.55% prediction accuracy with 99.43% cross-validation score
- 🌱 **22 Crop Types**: Supports rice, maize, chickpea, banana, mango, grapes, cotton, and 15 more
- 📊 **Interactive UI**: User-friendly Streamlit interface with real-time predictions
- 🔍 **Top 3 Recommendations**: Displays confidence scores for multiple crop options
- 📈 **Data Visualization**: Comprehensive EDA with feature importance analysis
- ⚡ **Fast Predictions**: Real-time recommendations in milliseconds

**Input Interface:**
- Adjust sliders for N, P, K, temperature, humidity, pH, and rainfall
- Real-time parameter validation

**Prediction Results:**
- Best crop recommendation with confidence score
- Top 3 alternatives with probabilities
- Crop information (season, duration, icon)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 99.55% |
| **Cross-Validation** | 99.43% (±0.48%) |
| **Training Accuracy** | 100.0% |
| **Generalization Gap** | 0.45% |
| **Precision (avg)** | 99.77% |
| **Recall (avg)** | 99.77% |
| **F1-Score (avg)** | 99.77% |

### Confusion Matrix Highlights
- 22 crops with balanced performance
- Only 2 misclassifications out of 440 test samples
- Perfect scores (1.00) for 18 out of 22 crops

## 🛠️ Tech Stack

**Machine Learning:**
- Python 3.8+
- Scikit-learn (Random Forest Classifier)
- Pandas, NumPy
- Matplotlib, Seaborn

**Web Application:**
- Streamlit
- Custom CSS styling

**Tools:**
- Jupyter Notebook for model development
- Pickle for model serialization

## 📁 Project Structure

```
crop-recommendation/
│
├── data/
│   └── Crop_recommendation.csv          # Dataset (2200 samples)
│
├── notebooks/
│   └── crop_recommendation.ipynb        # Model training & EDA
│
├── models/
│   ├── crop_model.pkl                   # Trained Random Forest model
│   └── scaler.pkl                       # StandardScaler for preprocessing
│
├── app.py                               # Streamlit web application                   
├── README.md                            # Project documentation
└── LICENSE                              # MIT License
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/abhigyaabs2/Crop-Recommendation-System.git
cd Crop-Recommendation-System
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Download Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset) 

### Step 4: Train the Model
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/crop_recommendation.ipynb
```

### Step 6: Run the Streamlit App
```bash
streamlit run crop.py
```
The app will open in your browser at `http://localhost:8501`

## 💻 Usage

### Using the Web Application

1. **Input Soil Parameters:**
   - Nitrogen (N): 0-140 kg/ha
   - Phosphorus (P): 5-145 kg/ha
   - Potassium (K): 5-205 kg/ha
   - pH Level: 3.5-9.5

2. **Input Climate Conditions:**
   - Temperature: 8-44°C
   - Humidity: 14-100%
   - Rainfall: 20-300mm

3. **Get Recommendations:**
   - Click "Get Crop Recommendation"
   - View the best crop with confidence score
   - Explore top 3 alternatives

### Using the Model Programmatically

```python
import pickle
import numpy as np

# Load model and scaler
with open('models/crop_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input
input_data = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.93]])
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

print(f"Recommended Crop: {prediction}")
print(f"Confidence: {max(probabilities)*100:.2f}%")
```

## 📈 Dataset Information

**Source:** [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset)

**Features:**
- **N**: Nitrogen content ratio in soil (kg/ha)
- **P**: Phosphorus content ratio in soil (kg/ha)
- **K**: Potassium content ratio in soil (kg/ha)
- **Temperature**: Temperature in degree Celsius
- **Humidity**: Relative humidity in %
- **pH**: pH value of the soil
- **Rainfall**: Rainfall in mm

**Target:** Crop label (22 different crops)

**Size:** 2,200 samples

## 🧪 Model Details

### Algorithm: Random Forest Classifier

**Hyperparameters:**
- `n_estimators=100`: Number of decision trees
- `max_depth=20`: Maximum tree depth to prevent overfitting
- `min_samples_split=5`: Minimum samples required to split a node
- `random_state=42`: For reproducibility

**Feature Importance (Top 5):**
1. Potassium (K)
2. Phosphorus (P)
3. Rainfall
4. Nitrogen (N)
5. Temperature

### Why Random Forest?
- Handles non-linear relationships well
- Robust to outliers
- Provides feature importance
- Excellent performance on tabular data
- Low risk of overfitting with proper tuning

## 🎓 Key Learnings

- **Data Preprocessing**: StandardScaler for feature normalization
- **Model Selection**: Compared multiple algorithms, Random Forest performed best
- **Hyperparameter Tuning**: Optimized max_depth and min_samples_split
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Deployment**: Built production-ready web application with Streamlit

## 🚀 Future Enhancements

- [ ] Add soil image analysis using CNN
- [ ] Integrate weather API for real-time climate data
- [ ] Multi-language support for regional farmers
- [ ] Mobile app development (Flutter/React Native)
- [ ] Crop yield prediction feature
- [ ] Regional crop database with local varieties
- [ ] Integration with IoT sensors for automated soil testing
- [ ] Chatbot for farming queries

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset)
- Inspired by the need to help farmers make data-driven decisions
- Built with ❤️ for the agricultural community

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

**Made with 🌱 for sustainable agriculture**

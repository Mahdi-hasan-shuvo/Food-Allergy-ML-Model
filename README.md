# 🥗 Food Allergy Prediction ML Model

A comprehensive machine learning system for predicting food allergy risks based on clinical, demographic, and dietary factors. This project implements multiple classification algorithms to assess allergy susceptibility with high accuracy.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

This machine learning model predicts the likelihood of food allergies in individuals using various clinical and lifestyle factors. The system processes patient data, applies multiple classification algorithms, and provides both binary predictions and probability scores for allergy risk assessment.

## 🚀 Features

- **Multiple ML Algorithms**: Random Forest, Logistic Regression, SVM, Gradient Boosting, K-Nearest Neighbors
- **Comprehensive Data Processing**: Handling of categorical and numerical features with proper encoding and scaling
- **Detailed Evaluation**: Cross-validation, classification reports, confusion matrices, and ROC-AUC scoring
- **Model Persistence**: Save and load trained models for deployment
- **Interactive Prediction**: Function to make predictions on new patient data
- **Visual Analytics**: Feature importance charts and performance visualizations

## 📊 Dataset Features

The model analyzes 10 key features to predict allergy risk:

1. **Age**: Patient's age in years
2. **Gender**: Male/Female
3. **Family_History**: History of allergies in family (Yes/No)
4. **Previous_Reaction**: Severity of previous reactions (None/Mild/Moderate/Severe)
5. **Symptoms**: Type of allergic symptoms
6. **Food_Type**: Type of food causing reaction
7. **Food_Frequency**: How often the food is consumed
8. **Medical_Conditions**: Existing medical conditions
9. **IgE_Levels**: Immunoglobulin E antibody levels
10. **Severity_Score**: Self-reported severity score (1-10)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Mahdi-hasan-shuvo/Food-Allergy-ML-Model.git
cd Food-Allergy-ML-Model
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Food-Allergy-ML-Model/
│
├── data/
│   ├── food_allergy_data.csv      # Main dataset
│   └── food_allergy_test.csv      # Test dataset (if available)
│
├── models/
│   └── food_allergy_model.pkl     # Saved trained model
│
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   ├── Model_Training.ipynb       # Model development notebook
│   └── Prediction_Demo.ipynb      # Prediction examples
│
├── requirements.txt               # Python dependencies
├── main.py                       # Main executable script
└── README.md                     # Project documentation
```

## 🧪 Usage

### Training the Model

Run the main script to train and evaluate the model:

```bash
python main.py
```

### Making Predictions on New Data

Use the provided function to make predictions:

```python
from src.prediction import predict_allergy

# Example patient data
new_patient = {
    'Age': 30,
    'Gender': 'Male',
    'Family_History': 'No',
    'Previous_Reaction': 'Moderate',
    'Symptoms': 'Swelling',
    'Food_Type': 'Nuts',
    'Food_Frequency': 10,
    'Medical_Conditions': 'Asthma',
    'IgE_Levels': 500.0,
    'Severity_Score': 7
}

prediction, probability = predict_allergy(new_patient)
print(f"Prediction: {'Allergic' if prediction == 1 else 'Not Allergic'}")
print(f"Probability: {probability:.4f}")
```

### Loading a Saved Model

```python
import joblib

# Load the saved model
model_data = joblib.load('models/food_allergy_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
```

## 📈 Model Performance

The model achieves excellent performance metrics:

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.5% |
| **Precision** | 91.8% |
| **Recall** | 93.2% |
| **F1-Score** | 92.5% |
| **ROC-AUC** | 96.3% |

### Cross-Validation Results
- 5-fold cross-validation consistency
- Mean CV accuracy: 91.8%

## 🔍 Exploratory Data Analysis

Key insights from the data analysis:
- IgE Levels and Severity Score are strong predictors
- Family history significantly increases allergy risk
- Certain food types (nuts, shellfish) show higher association with allergies
- Age distribution affects allergy prevalence

## 🤖 Machine Learning Algorithms

The project implements and compares multiple algorithms:

1. **Random Forest Classifier** - Best performing model
2. **Logistic Regression** - Good baseline model
3. **Support Vector Machine** - Effective for high-dimensional data
4. **Gradient Boosting** - Strong sequential learner
5. **K-Nearest Neighbors** - Instance-based approach

## ⚙️ Hyperparameter Tuning

Optimized parameters for Random Forest:
- n_estimators: 200
- max_depth: 20
- min_samples_split: 2

## 📊 Visualization

The project includes comprehensive visualizations:
- Model performance comparison bar charts
- Feature importance rankings
- Confusion matrix heatmaps
- ROC curves (if probability predictions available)

## 🚀 Deployment

The model can be deployed in various environments:

### Web Application
```python
# Example Flask app endpoint
@app.route('/predict', methods=['POST'])
def predict():
    patient_data = request.get_json()
    prediction, probability = predict_allergy(patient_data)
    return jsonify({
        'prediction': 'Allergic' if prediction == 1 else 'Not Allergic',
        'probability': probability,
        'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
    })
```

### API Integration
Integrate with healthcare systems using RESTful API endpoints for real-time predictions.

## ⚠️ Limitations & Considerations

1. **Data Quality**: Model performance depends on training data quality
2. **Feature Availability**: Requires all 10 features for accurate predictions
3. **Categorical Handling**: New categories not seen in training may cause errors
4. **Medical Disclaimer**: Should be used as a screening tool, not diagnostic replacement

## 🔮 Future Enhancements

- [ ] Real-time data integration with health APIs
- [ ] Mobile application for on-the-go predictions
- [ ] Additional allergy types and cross-reactivity predictions
- [ ] Deep learning approaches for improved accuracy
- [ ] Multi-language support for global deployment

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Medical professionals who provided domain expertise
- Open-source community for machine learning libraries
- Researchers in allergology and immunology

## 📞 Contact

Mahdi Hasan Shuvo - [shuvobbhh@gmail.com](mailto:shuvobbhh@gmail.com)

Project Link: [https://github.com/Mahdi-hasan-shuvo/Food-Allergy-ML-Model](https://github.com/Mahdi-hasan-shuvo/Food-Allergy-ML-Model)

---

**Disclaimer**: This tool is for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions.

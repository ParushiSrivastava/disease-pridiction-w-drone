KrishiDrishti
AI-Powered Plant Disease Detection with Severity Analysis

KrishiDrishti is a deep learning-based application designed to detect plant diseases from leaf images. It provides real-time predictions, severity estimation, and a foundation for future integration with drone-based agricultural solutions.

 Features
 Plant Disease Detection using EfficientNet
 Confidence Score for predictions
 Severity Analysis (Mild / Moderate / Severe)
 Image Enhancement & Processing
 Top-3 Predictions Display
 Farmer-Centric UI (Mobile Friendly)
 Future Scope: Drone Integration
 Future Scope: Medication Recommendations
 Model Details
Architecture: EfficientNet (Transfer Learning)
Framework: TensorFlow / Keras
Dataset: PlantVillage + Custom Dataset (Neem & Bougainvillea)
Input Size: 224 × 224
Output: Multi-class classification
 Project Structure

KrishiDrishti/
│
├── app.py
├── plant_disease_efficientnet.keras
├── class_indices.json
├── requirements.txt
├── README.md

 Installation & Setup
1. Clone the repository

git clone https://github.com/your-username/krishidrishti.git

cd krishidrishti

2. Install dependencies

pip install -r requirements.txt

3. Run the app

streamlit run app.py

 Run on Mobile



 How It Works
Upload a leaf image
Image is preprocessed
Model predicts disease
Severity is calculated
Results are displayed
 Real-World Impact
Helps farmers detect diseases early
Reduces crop loss
Supports smart farming
Enables future drone-based solutions
 Future Scope
Medication recommendation system
Drone integration (Drone Didi initiative)
Location-based support
Real-time camera detection
 References
PlantVillage Dataset
TensorFlow Documentation
EfficientNet Paper
OpenCV

 Author

Parushi Srivastava
B.Tech CSE


Developed as part of an academic project to explore AI in agriculture and support sustainable farming.# disease-pridiction-w-drone

# HR Analytics Engine MVP

## Overview
The HR Analytics Engine is a machine learning-powered tool designed to predict employee attrition and estimate burnout risk. This MVP (Minimum Viable Product) provides HR professionals with data-driven insights to proactively address workforce challenges.

## Features
- **Attrition Prediction**: ML model to identify employees at risk of leaving
- **Burnout Risk Assessment**: Algorithm to estimate employee burnout levels
- **Interactive Dashboard**: User-friendly Streamlit interface for data input and visualization

## Project Structure
```
hr-analytics-engine-mvp/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── run_hr_analytics.bat    # Windows batch file to run the application
├── data/                   # Dataset directory
│   └── hr_data.csv         # HR dataset
├── models/                 # Saved model files
│   ├── attrition_model.pkl # Trained XGBoost model
│   ├── label_encoders.pkl  # Categorical encoders
│   └── input_columns.pkl   # Column order reference
├── utils/                  # Utility functions
│   └── preprocess.py       # Data preprocessing utilities
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (optional)

### Setup
1. Clone the repository or download the source code:
   ```
   git clone https://github.com/Escanor244/hr-analytics-engine-mvp.git
   cd hr-analytics-engine-mvp
   ```

2. Create and activate a virtual environment:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the attrition prediction model:
```
python train_model.py
```
This will create model files in the `models/` directory.

### Running the Application
1. Start the Streamlit app:
   ```
   # Windows
   run_hr_analytics.bat
   
   # macOS/Linux
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501).

3. Enter employee information in the sidebar and view predictions in the main panel.

## Model Information
- **Algorithm**: XGBoost Classifier
- **Features**: 30+ employee attributes including demographics, job satisfaction, and work history
- **Target**: Binary classification (Will leave: Yes/No)
- **Performance**: Accuracy metrics available in model training output

## Future Enhancements
- Additional predictive models for other HR metrics
- Batch prediction capabilities for multiple employees
- Data visualization for trend analysis
- API integration with HRIS systems

## License
[Your License Information]

## Contact
[Your Contact Information] 

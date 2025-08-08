# HIV Drug Prediction Model

A machine learning model to predict which chemical compounds can fight HIV effectively, helping researchers focus on the most promising candidates.

## Overview

This project uses QSAR (Quantitative Structure-Activity Relationship) modeling to predict HIV drug compound efficacy. It helps pharmaceutical researchers identify promising compounds before expensive lab testing.

**Key Benefits:**
- Reduce screening time from months to hours
- Focus resources on high-probability compounds
- Improve success rates in drug discovery

## Features

- Complete ML pipeline from data preprocessing to deployment
- REST API for real-time predictions
- Model monitoring and performance tracking
- Batch processing for large compound libraries
- Docker containerization for easy deployment

## Dataset

**Source:** [NCI AIDS Antiviral Screen Data](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/158204006/AIDS+Antiviral+Screen+Data)

- 40,000+ HIV-tested compounds
- Activity classes: CA (Active), CM (Moderately Active), CI (Inactive)
- EC50/IC50 measurements
- Molecular structure data

## Technology Stack

- **Language:** Python 3.8+
- **ML Libraries:** scikit-learn, XGBoost, RDKit
- **API:** FastAPI
- **Database:** PostgreSQL
- **MLOps:** MLflow
- **Deployment:** Docker

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 87.3% |
| F1-Score | 0.84 |
| Cohen's Kappa | 0.79 |
| AUC-ROC | 0.91 |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/pari1jay/Prediction-Model-HC.git
cd Prediction-Model-HC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Development mode
python src/main.py

# Production with Docker
docker-compose up -d
```

### API Usage

```python
import requests

# Predict compound activity
response = requests.post(
    "http://localhost:8000/predict",
    json={"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"}
)
print(response.json())
```

## Usage Examples

### Training a Model

```python
from src.training.pipeline import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.load_data("data/hiv_compounds.csv")
pipeline.preprocess()
pipeline.train()
pipeline.evaluate()
```

### Making Predictions

```python
from src.prediction.predictor import CompoundPredictor

predictor = CompoundPredictor.load("models/best_model.pkl")
result = predictor.predict_smiles("CCO")
print(f"Activity: {result['activity']}, Confidence: {result['confidence']:.3f}")
```

### Batch Processing

```bash
python scripts/batch_predict.py --input compounds.csv --output predictions.csv
```

## Project Structure

```
Prediction-Model-HC/
├── src/
│   ├── data/           # Data processing
│   ├── features/       # Feature engineering
│   ├── models/         # ML models
│   ├── training/       # Training pipeline
│   ├── prediction/     # Prediction service
│   └── api/           # REST API
├── data/              # Datasets
├── models/            # Trained models
├── scripts/           # Utility scripts
├── tests/             # Test files
└── docs/              # Documentation
```

## Data Processing Pipeline

1. **Data Integration:** Merge screening results, EC50/IC50 values, and molecular structures
2. **Quality Control:** Handle duplicates, conflicts, and missing data
3. **Feature Engineering:** Calculate molecular descriptors and fingerprints
4. **Model Training:** Train and validate multiple ML models
5. **Evaluation:** Assess model performance using relevant metrics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- National Cancer Institute for the AIDS Antiviral Screen Data
- RDKit community for cheminformatics tools
- Open source contributors

## Contact

**Pari Jay** - [GitHub](https://github.com/pari1jay)

Project Link: [https://github.com/pari1jay/Prediction-Model-HC](https://github.com/pari1jay/Prediction-Model-HC)

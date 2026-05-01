# MediApp: AI-Powered Medical Image Diagnostic System

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/downloads/)
[![Django](https://img.shields.io/badge/django-5.0-green)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**An intelligent medical imaging diagnostic platform leveraging deep convolutional neural networks for automated disease detection across multiple modalities.**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Model Details](#model-details)

</div>

---

## Overview

MediApp is a production-ready web application that integrates state-of-the-art deep learning models for real-time diagnostic support across five critical medical imaging tasks. The system combines computer vision expertise with medical domain knowledge to provide clinicians with AI-assisted diagnostic capabilities.

---

## Features

### Core Capabilities
- **Multi-Model Inference Pipeline**: Unified framework for deploying heterogeneous CNN architectures
- **Real-Time Predictions**: Sub-100ms inference latency for all modalities
- **Probabilistic Output**: Confidence-calibrated predictions with class-wise probability distributions
- **Web Interface**: Intuitive Django-based UI for non-technical end-users
- **Image Preprocessing**: Automated normalization, resizing, and color space conversion

### Technical Highlights
- **Production-Grade Stack**: Django + Gunicorn + PostgreSQL
- **Deep Learning Inference**: Optimized TensorFlow/Keras model serving
- **Database Persistence**: Complete audit trail of all diagnoses
- **Containerization-Ready**: Procfile configured for cloud deployment
- **Static Asset Management**: CSS/JS optimization for responsive design

---

## Architecture

```
MediApp/
├── core_app/
│   ├── machinelearning.py      # Deep learning inference pipeline
│   ├── models.py               # Django ORM models (image storage)
│   ├── views.py                # Request routing & business logic
│   └── forms.py                # Image upload form validation
├── mediapp/
│   ├── settings.py             # Django configuration
│   ├── urls.py                 # URL routing
│   └── wsgi.py                 # WSGI application entry point
├── templates/                  # HTML templates (disease-specific UI)
├── static/
│   ├── css/styles.css          # Responsive styling
│   └── js/scripts.js           # Client-side interactions
├── weights/                    # Pre-trained model weights
│   ├── malaria.h5              # Malaria CNN (~45 MB)
│   ├── breastcancer.h5         # IDC CNN (~50 MB)
│   ├── brain_tumor.h5          # Brain tumor CNN (~55 MB)
│   ├── diabetes_retinopathy.h5 # DR grading CNN (~60 MB)
│   └── retina_OCT.h5           # OCT classification CNN (~58 MB)
└── requirements.txt            # Python dependencies
```

### Model Architecture Overview

Each pre-trained model is implemented as a **binary or multi-class CNN** with the following characteristics:
- **Input Resolution**: 128×128 pixels (RGB normalized)
- **Architecture**: Custom CNN with batch normalization and dropout
- **Output Layer**: Softmax with probabilistic class membership
- **Loss Function**: Categorical cross-entropy
- **Optimization**: Adam optimizer with adaptive learning rates

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum (for model inference)
- 500MB disk space (model weights)

### Setup Instructions

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/mediapp.git
cd mediapp
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Database Initialization**
```bash
python manage.py migrate
python manage.py createsuperuser  # Create admin user
```

5. **Collect Static Files**
```bash
python manage.py collectstatic --noinput
```

6. **Run Development Server**
```bash
python manage.py runserver
```

Access the application at `http://localhost:8000`

### Production Deployment

For DigitalOcean/Heroku deployment:
```bash
# Environment variables required
export SECRET_KEY='your-secret-key'
export DEBUG='False'
export ALLOWED_HOSTS='yourdomain.com'
export DATABASE_URL='postgresql://user:password@host:5432/dbname'

# Deploy via Procfile (Gunicorn + Django)
gunicorn mediapp.wsgi --workers 4 --bind 0.0.0.0:8000
```

---

## Usage

### Web Interface Workflow

1. **Navigate to Disease-Specific Module**
   - Home page offers five diagnostic pathways
   - Each modality has dedicated interface

2. **Upload Medical Image**
   - Supported formats: PNG, JPG, JPEG
   - Maximum file size: 10MB
   - Images stored in `media/` directory

3. **Automatic Inference**
   - Image preprocessed (resized to 128×128)
   - Model prediction computed in real-time
   - Results displayed with confidence scores

4. **Result Interpretation**
   - Class-wise probability distribution
   - Clinical decision support text
   - Audit trail saved to database

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

**Siddharth Saraswat**
- GitHub: [@siddharth1012](https://github.com/siddharth1012)
- LinkedIn: [Siddharth Saraswat](https://linkedin.com/in/siddharthsaraswat1)

---

## Acknowledgments

- **Dataset Sources**: NIH Malaria, BreakHis, Kaggle Brain MRI, EyePACS, OCT Dataset
- **Framework Credits**: Django, TensorFlow, Keras communities
---

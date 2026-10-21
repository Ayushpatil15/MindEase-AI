# 🧠 MindEase AI — Intelligent Student Mental-Wellness & Stress Management Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask_3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-TensorFlow%20%7C%20OpenCV-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Database](https://img.shields.io/badge/Database-SQLite%20%2F%20SQLAlchemy-003B5C?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-46A2F1?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **MindEase AI** is a privacy-first, full-stack mental wellness platform tailored for academic environments. It empowers students with self-awareness tools—combining validated psychological assessments, multi-modal mood tracking, real-time facial expression analysis (CNN), and personalized wellness recommendations—while giving campus administrators anonymized, aggregate wellness insights.

---

## ✨ Key Features

### 🎓 Student Wellness Portal
- 📊 **Validated Stress Assessment**: 11-question evidence-backed stress evaluation with instant scoring and longitudinal history tracking.
- 😊 **Multi-Modal Mood Check-In**: Quick daily mood logging with 7-day trend visualizations, contextual tags, and optional private journaling.
- 📷 **Facial Expression Analysis**: Edge-ready CNN facial expression recognition classifying emotional states using OpenCV Haar Cascades for face detection.
- ⚡ **Composite Wellness Score**: Explainable algorithmic wellness index calculated from stress levels, mood consistency, and completed activities.
- 🎯 **Smart Recommendation Engine**: Tailored, rule-based activity suggestions dynamically suited to the student's current stress and mood state.
- ✅ **Wellness Action Plan**: Personalized micro-task tracking system encouraging daily mindfulness, physical activity, and stress management habits.
- 🆘 **Crisis & Support Directory**: Immediate access to campus counseling services, emergency hotlines, and peer support contacts.

### 🏛️ Institutional Admin Portal
- 📈 **Aggregated Analytics**: Institutional dashboard displaying anonymized stress distribution, campus mood trends, and resource engagement.
- 🔒 **Privacy Guarantee**: Strict isolation between student private entries and admin analytics—no individual journal entries or personal data are exposed.

---

## 🏗️ System Architecture & Project Structure

```
MindEase AI/
├── app/
│   ├── config/             # Application configuration & support resource definitions
│   ├── forms/              # Flask-WTF forms (auth, mood check-in, stress survey)
│   ├── ml/                 # Machine Learning pipeline (CNN model, preprocessor, inference engine)
│   ├── models/             # SQLAlchemy ORM models (User, Mood, Stress, Recommendation, Task)
│   ├── routes/             # Modular Blueprint controllers (Auth, Student, Admin, Emotion, Stress, Mood)
│   ├── services/           # Domain logic (Wellness calculator, Daily plan, Activity, History)
│   ├── static/             # Frontend assets (CSS design system, JS charts, Haar cascades)
│   ├── templates/          # Jinja2 HTML templates (responsive dark/light-ready UI)
│   └── utils/              # Custom decorators, security helpers, and input validators
├── data/                   # Dataset directory for FER-2013 (git-ignored)
├── instance/               # SQLite runtime database (git-ignored)
├── migrations/             # Alembic database schema migrations
├── models/                 # Pre-trained CNN weights & TFLite quantized models
├── scripts/                # Benchmark, quantization, and latency audit utilities
├── tests/                  # Pytest test suite (unit & integration coverage)
├── .env.example            # Environment template configuration
├── .gitignore              # Git ignore rules for virtualenvs, secrets, models & DBs
├── requirements.txt        # Production dependencies
└── run.py                  # Application entry point
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10+** installed
- **Git** installed

### 1. Clone the Repository
```bash
git clone https://github.com/Ayushpatil15/MindEase-AI.git
cd MindEase-AI
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to create your local `.env` configuration:
```bash
# Windows PowerShell
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
*(Optionally open `.env` and set custom `SECRET_KEY` and configuration values)*

### 5. Initialize Database & Run App
```bash
python run.py
```
The application will launch locally at **`http://127.0.0.1:5000`**.

---

## 🔑 Administrative Setup

To create an admin account for accessing the aggregate analytics dashboard:

```bash
flask create-admin
```
Follow the interactive prompt to set username, email, and password.

---

## 🧪 Machine Learning Pipeline & Model Training

MindEase AI uses a custom Convolutional Neural Network (CNN) trained on facial expression data for real-time, low-latency emotion detection.

### Training the Model (Optional)
1. Download the **FER-2013** dataset from Kaggle and place `fer2013.csv` in `data/`.
2. Execute the training script:
```bash
python -m app.ml.train --data data/fer2013.csv --epochs 15 --output models/fer2013_cnn.h5
```

### Model Quantization & Benchmark
To run low-latency audits and TFLite conversions:
```bash
python scripts/latency_audit.py
python scripts/phase510_benchmark.py
```

---

## 🧪 Running Unit & Integration Tests

Ensure full system reliability with `pytest`:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=app --cov-report=term-missing
```

---

## 🛡️ Privacy & Ethical Design Principles

- 🔐 **Privacy First**: MindEase AI is a self-awareness and wellness tracking tool—not a diagnostic medical device.
- 🙈 **Zero Private Data Exposure**: Admin views display strictly aggregated and anonymized metrics. Personal journal entries and individual scores are never exposed.
- 🆘 **Safety Net**: The safety service keyword check is a simple safety net, not a reliable detector and not a diagnostic tool. In-memory crisis language detection connects students to 24/7 care resources (Tele-MANAS 14416 and emergency 112) without ever persisting, logging, or exposing private reflections.
- 🤖 **Transparent AI**: Explainable scoring algorithms provide full breakdown of how wellness metrics are derived.

---

## 📜 License

Distributed under the **MIT License**.

# MindEase AI

A student mental-wellness platform designed to help students understand and manage everyday stress, mood, and wellbeing.

## Features (MVP)

- **Stress Assessment** — 11-question validated assessment with scoring and history
- **Mood Check-In** — Daily mood tracking with 7-day trend visualization
- **Facial Expression Check-In** — CNN-based facial expression analysis (5 classes)
- **Personal Wellness Score** — Composite, explainable wellness indicator
- **Personalized Recommendations** — Rule-based suggestions based on actual user state
- **Wellness Tasks** — Personal to-do system for wellness activities
- **Private Journal** — Optional journaling with mood check-ins
- **Support Resources** — Campus counseling and crisis resource links
- **Admin Analytics** — Aggregated, anonymized institutional dashboard

## Quick Start

### 1. Clone & set up environment

```bash
cd "MindEase AI"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env with your SECRET_KEY
```

### 3. Run the application

```bash
python run.py
```

The app will be available at `http://localhost:5000`.

### 4. Create an admin user

```bash
flask create-admin
```

You will be prompted for username, email, and password.

### 5. Train the facial expression model (optional)

Download the FER-2013 dataset from Kaggle, then:

```bash
python -m app.ml.train --data path/to/fer2013.csv --epochs 10 --output models/fer2013_cnn.h5
```

### 6. Run tests

```bash
pytest
```

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask (application factory) |
| Database | SQLite (SQLAlchemy ORM) |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Authentication | Flask-Login, Werkzeug password hashing |
| Computer Vision | OpenCV (Haar Cascade) |
| Machine Learning | TensorFlow/Keras (CNN) |
| Testing | Pytest |
| Optional AI | Google Gemini / Mistral (provider abstraction) |

## Important Notes

- This is a **wellness/self-awareness tool**, not a clinical or diagnostic application.
- Facial expression analysis is an estimate and may not reflect actual emotional state.
- No fake data, simulated predictions, or hardcoded statistics are used.
- Journal entries and reflections are private student data, never exposed to admins.
- The application gracefully handles empty databases and missing ML models.

## License

This project is for educational purposes.

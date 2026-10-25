# 🧠 MindEase AI — Intelligent Student Mental-Wellness & Stress Management Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask_3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Database](https://img.shields.io/badge/Database-PostgreSQL_%7C_Neon-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://neon.tech)
[![AI Engine](https://img.shields.io/badge/AI-Google_Gemini-8E75C2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![ML](https://img.shields.io/badge/CV%20%2F%20ML-TensorFlow_Lite_%7C_OpenCV-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Tests](https://img.shields.io/badge/Tests-336%20Passing-success?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![Deployment](https://img.shields.io/badge/Deployment-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **MindEase AI** is a privacy-first, full-stack mental wellness and academic stress-management platform tailored for higher-education students. It bridges the gap between student daily life and evidence-informed self-care by combining validated psychological assessments (PSS), multi-modal mood tracking, real-time edge facial expression analysis (TFLite CNN), personalized guided micro-activities, and a supportive, non-diagnostic AI wellness companion (Google Gemini)—all backed by a serverless PostgreSQL architecture and privacy-preserving institutional analytics.

---

## 🌐 Live Application & Source Code

- **Live Deployed Web Application:** [https://mindease-ai-6kjf.onrender.com](https://mindease-ai-6kjf.onrender.com)
- **Official GitHub Repository:** [https://github.com/Ayushpatil15/MindEase-AI](https://github.com/Ayushpatil15/MindEase-AI)
- **Target Audience:** College and university students, campus counselors, and institutional administrators.

---

## 📋 Table of Contents

- [The Problem \& Solution](#-the-problem--solution)
- [Key Features](#-key-features)
- [Interactive Guided Activities](#-interactive-guided-activities)
- [Screenshots \& Preview](#-screenshots--preview)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [How the Application Works](#-how-the-application-works)
- [Project Directory Structure](#-project-directory-structure)
- [Local Installation \& Setup](#-local-installation--setup)
- [Environment Configuration](#-environment-configuration)
- [Database Setup \& Migrations](#-database-setup--migrations)
- [Testing \& Quality Assurance](#-testing--quality-assurance)
- [Production Deployment](#-production-deployment)
- [Privacy, Ethics \& Safety Framework](#-privacy-ethics--safety-framework)
- [Known Limitations \& Future Roadmap](#-known-limitations--future-roadmap)
- [License \& Acknowledgements](#-license--acknowledgements)

---

## 🎯 The Problem & Solution

### The Challenge
University students face unprecedented academic pressure, exam anxiety, sleep disruption, and burnout. Traditional campus counseling services are frequently bottlenecked by long appointment queues, social stigma, and lack of continuous, low-friction self-monitoring tools.

### The MindEase AI Approach
MindEase AI acts as an early, accessible **first-line self-awareness tool**:
1. **Low Friction:** Enables 30-second daily check-ins across multiple modalities (mood valence, facial expression, and structured assessments).
2. **Actionable Micro-Interventions:** Translates wellness metrics into immediate 5-to-30-minute evidence-informed coping tools (box breathing, sensory grounding, pomodoro focus).
3. **Compassionate AI Support:** Provides an on-demand, non-diagnostic conversational companion powered by Google Gemini, equipped with memory of non-sensitive student study preferences.
4. **Safety Net:** Automatically intercepts crisis-related language and distress triggers to provide instant, confidential emergency helpline access (Tele-MANAS, 112).
5. **Privacy by Design:** Employs strict in-memory camera frame processing (no photos or video streams are ever stored) and $K \ge 5$ cohort anonymization for institutional admin analytics.

---

## ✨ Key Features

### 🎓 Student Wellness Hub

- 📊 **Validated Stress Assessment (PSS-Derived)**
  - 11-question evidence-backed stress survey evaluating perceived control and academic pressure.
  - Generates an instant, explainable score (0–44) classified into **Low**, **Moderate**, or **High** stress tiers.
  - Longitudinal history charts with timestamped records and optional private reflections.

- 😊 **Multi-Modal Mood Check-In**
  - Daily mood logging across a 5-point valence spectrum (*Very good*, *Good*, *Okay*, *Low*, *Very low*).
  - Optional private journaling notes to record personal context, thoughts, and triggers.
  - 7-day and 30-day interactive valence trend visualizations powered by Chart.js.

- 📷 **Facial Expression Analysis (Edge CNN Inference)**
  - Browser-based webcam capture with real-time face detection using OpenCV Haar Cascades.
  - Low-latency classification into 5 distinct emotional classes (*Angry*, *Fear*, *Happy*, *Sad*, *Neutral*) using an optimized TensorFlow Lite neural network (`challenger_fp32.tflite`).
  - **Zero Image Storage Guarantee:** Video frames are decoded in memory, evaluated for facial landmarks, classified, and immediately discarded.

- ⚡ **Composite Wellness Score (0–100 Index)**
  - An explainable, algorithmic composite index computed dynamically:
    $$\text{Wellness Score} = (0.40 \times S) + (0.40 \times M) + (0.20 \times C)$$
    - **$S$ (Stress Normalization):** Normalized from latest PSS score: $((44 - \text{Score}) / 44) \times 100$.
    - **$M$ (Mood Normalization):** 7-day average mood valence mapped to a 0–100 scale: $((\text{Valence} + 2) / 4) \times 100$.
    - **$C$ (Consistency Index):** Active check-in days within the 7-day window divided by elapsed registration days.
  - Automatic mathematical fallback weighting when only one modality is recorded.

- 🤖 **AI Wellness Companion (Google Gemini REST Integration)**
  - Empathetic, supportive conversational companion designed specifically for higher-education challenges.
  - Enforces strict non-diagnostic boundaries: provides encouragement, study tips, and mindfulness exercises without attempting psychiatric diagnosis or prescribing medical interventions.
  - Context-aware system prompt dynamically injected with non-sensitive user wellness preferences.
  - Client-side sanitized conversation history limited to 10 alternating turns to prevent prompt injection and context exhaustion.

- 🧠 **Persistent Chat Memory**
  - Transparent storage of non-sensitive student study preferences, sleep goals, and hobbies.
  - Strict user isolation and boundary checks—clinical details, journal notes, and assessments are never stored as memory facts.
  - Students retain full visibility and agency with one-click toggles and individual memory deletion controls.

- 🎯 **Personalized Recommendation Engine & Daily Tasks**
  - Rule-based recommendation engine suggesting specific mindfulness, academic, physical, or social micro-activities tailored to current stress/mood state.
  - Personal daily task tracker allowing students to convert recommendations into actionable, checkable habits.

- 🆘 **Crisis & Support Directory**
  - In-memory safety screening evaluating distress signals from mood drops, high stress scores, or crisis keywords.
  - Immediate, non-diagnostic support cards displaying verified 24/7 helplines: **Tele-MANAS (14416 / 1800-891-4416)** and **National Emergency (112)**.

### 🏛️ Institutional Admin Portal

- 📈 **Aggregated Campus Analytics:** High-level dashboard displaying anonymized stress distribution, campus mood trajectories, and resource engagement.
- 🔒 **Privacy Threshold Enforcement:** Strict minimum group size constraint ($K \ge 5$) prevents queries on small cohorts, ensuring individual student anonymity.
- 🚫 **Zero Private Reflection Exposure:** Student journal entries, facial scan timestamps, and individual assessments are strictly inaccessible to administrators.

---

## 🧘 Interactive Guided Activities

MindEase AI features 6 dedicated, interactive guided wellness activities engineered for academic self-regulation:

| Activity | Category | Duration | Description & Interactive Mechanics |
| :--- | :---: | :---: | :--- |
| **Box Breathing** | Mindfulness | 5 min | 4-4-4-4 pacing cycle (Inhale, Hold, Exhale, Hold) with a synchronized visual timer and calming audio guidance. |
| **5-4-3-2-1 Sensory Grounding** | Mindfulness | 5 min | Step-by-step cognitive grounding tool with interactive tap-to-complete sensory counter anchors. |
| **Pomodoro Focus Session** | Academic | 30 min | 25-minute deep-work interval followed automatically by a 5-minute restorative rest break with pause/reset controls. |
| **Task Chunking & Micro-Steps** | Academic | 5 min | Interactive deconstruction tool breaking overwhelming projects into 3 manageable tasks, automatically syncable to the daily plan. |
| **Self-Compassion Reflection** | Reflection | 5 min | Structured guided cognitive reframing prompts targeting imposter syndrome and academic perfectionism. |
| **Desk Stretch & Reset** | Physical | 5 min | 5-step ergonomic stretching sequence targeting neck, shoulder, wrist, and spinal strain from prolonged study sessions. |

---

## 📸 Screenshots & Preview

Real application screenshots from the MindEase AI interface:

| Guided Activities Catalog | Box Breathing Interactive Timer |
| :---: | :---: |
| ![Activities Catalog](docs/screenshots/catalog_desktop.png) | ![Box Breathing](docs/screenshots/box_breathing_desktop.png) |

| 5-4-3-2-1 Sensory Grounding | Pomodoro Focus Session |
| :---: | :---: |
| ![Sensory Grounding](docs/screenshots/grounding_desktop.png) | ![Pomodoro Session](docs/screenshots/pomodoro_desktop.png) |

| Task Chunking Tool | Desk Stretch & Reset |
| :---: | :---: |
| ![Task Chunking](docs/screenshots/task_chunking_desktop.png) | ![Stretch Reset](docs/screenshots/stretch_reset_desktop.png) |

---

## 🏛️ System Architecture

The following diagram illustrates the complete, production request flow and component separation of MindEase AI:

```mermaid
flowchart TD
    subgraph Client["Student / Client Browser"]
        UI["Responsive HTML5 / CSS3 Interface"]
        Webcam["Webcam MediaStream API"]
        Charts["Chart.js Data Visualizations"]
    end

    subgraph Gateway["Reverse Proxy & Hosting (Render)"]
        TLS["TLS Termination & HTTPS"]
        ProxyFix["Werkzeug ProxyFix Middleware"]
    end

    subgraph AppServer["Flask 3.0 Application Server (Gunicorn)"]
        CSRF["Flask-WTF CSRF Protection & Rate Limiter"]
        
        subgraph Blueprints["Modular Route Controllers"]
            AuthBP["Auth Blueprint (/auth)"]
            StudentBP["Student Portal (/, /history, /activities, /support)"]
            MoodBP["Mood Blueprint (/mood)"]
            StressBP["Stress Blueprint (/stress)"]
            EmotionBP["Emotion API (/emotion)"]
            ChatBP["Chat Blueprint (/chat)"]
            AdminBP["Admin Blueprint (/admin)"]
            HealthBP["Health API (/api/health, /health)"]
        end

        subgraph Services["Core Domain Services"]
            CalcService["Wellness Calculator (Score 0-100)"]
            SafetyService["In-Memory Crisis & Safety Service"]
            RecService["Rule-Based Recommendation Engine"]
            TaskService["Daily Plan & Task Management"]
            ActivityService["Guided Activities Registry"]
            ChatContext["Chat Context & Dynamic Prompts"]
        end

        subgraph MLInference["Edge ML Pipeline"]
            Haar["OpenCV Haar Cascade Face Detector"]
            TFLite["TensorFlow Lite Engine (challenger_fp32.tflite)"]
        end
    end

    subgraph DataTier["Managed Persistence Tier"]
        NeonDB[("Neon Serverless PostgreSQL (Production) / SQLite (Dev)")]
    end

    subgraph ExternalServices["External AI Services"]
        GeminiAPI["Google Gemini REST API (v1beta generateContent)"]
    end

    %% Client Interactions
    UI -->|HTTPS Requests + CSRF| TLS
    Webcam -->|Base64 JPEG Frame Buffer| TLS
    TLS --> ProxyFix --> CSRF

    %% Route Distribution
    CSRF --> Blueprints

    %% Service Calls
    StudentBP --> CalcService
    StudentBP --> TaskService
    StudentBP --> ActivityService
    MoodBP --> SafetyService
    StressBP --> SafetyService
    EmotionBP --> MLInference
    ChatBP --> SafetyService
    ChatBP --> ChatContext

    %% ML Processing
    MLInference -->|1. Detect Face Crop 48x48| Haar
    Haar -->|2. In-Memory Tensor| TFLite
    TFLite -->|3. Emotion Class & Confidence| EmotionBP

    %% Chat Processing
    ChatContext -->|Non-Diagnostic Prompt + History| GeminiAPI
    GeminiAPI -->|Filtered Wellness Response| ChatBP

    %% Database Operations
    Blueprints -->|SQLAlchemy ORM (psycopg2-binary)| NeonDB
    Services -->|Persist Mood, Stress, Tasks, Memories| NeonDB
```

---

## 💻 Technology Stack

| Layer | Technologies | Rationale / Implementation |
| :--- | :--- | :--- |
| **Frontend** | HTML5, Vanilla CSS3, Vanilla JavaScript, Chart.js 4.4 | Zero heavy framework overhead; custom glassmorphism design system, mobile-responsive layout, and bundled Chart.js for data visualization. |
| **Backend Framework** | Python 3.10+ / 3.12, Flask 3.0+ | Lightweight, highly modular WSGI application structured with Flask Blueprints and application factory pattern. |
| **Authentication & Security** | Flask-Login, Flask-WTF, Werkzeug Security, Flask-Limiter | Session-based authentication with `scrypt` password hashing, strict CSRF validation, HTTPOnly/SameSite cookies, and IP rate limiting. |
| **Database & ORM** | PostgreSQL 16 (Neon Serverless), SQLAlchemy 3.1, Psycopg2-binary | Robust relational schema, connection pooling, and explicit driver normalization (`postgresql+psycopg2://`). SQLite supported for local dev. |
| **Database Migrations** | Flask-Migrate, Alembic | Tracked schema versioning across development, testing, and production environments. |
| **Machine Learning** | TensorFlow Lite 2.16+, OpenCV (`opencv-python-headless 4.9+`), NumPy | Thread-safe TFLite inference (`challenger_fp32.tflite`) for 5-class facial emotion classification; OpenCV Haar Cascade for face detection. |
| **Generative AI** | Google Gemini REST API (`gemini-3.5-flash-lite` / `gemini-1.5-flash`) | Direct HTTP REST integration via `requests` with strict system instructions, dynamic context injection, and timeout protection. |
| **Production WSGI** | Gunicorn 21.2+, Werkzeug ProxyFix | Multi-worker production WSGI server behind Render's reverse proxy with forwarded header normalization. |
| **Cloud Hosting** | Render (Web Service), Neon (PostgreSQL) | Continuous GitHub-to-Render deployment pipeline with automated health checks. |

---

## 🔄 How the Application Works

```mermaid
sequenceDiagram
    autonumber
    actor Student as Student
    participant Browser as Web Browser
    participant Flask as MindEase Backend
    participant ML as TFLite Engine
    participant DB as Neon PostgreSQL
    participant Gemini as Google Gemini API

    Student->>Browser: Complete Daily Mood / Stress / Emotion Check-In
    alt Facial Emotion Scan
        Browser->>Flask: POST /emotion/api/predict (In-Memory Image Frame)
        Flask->>ML: Detect face (Haar) & predict emotion (TFLite)
        ML-->>Flask: Emotion: Happy (92% confidence)
        Flask->>DB: Record EmotionScan (Discard raw image frame)
    else Stress Assessment
        Browser->>Flask: POST /stress/ (11 PSS Responses)
        Flask->>DB: Calculate score & record StressAssessment
    end

    Flask->>DB: Query 7-day mood & stress history
    Flask->>Flask: Compute Composite Wellness Score (0-100)
    Flask-->>Browser: Render Dashboard with Insights & Recommendations

    opt Chat with AI Wellness Companion
        Student->>Browser: "Feeling overwhelmed with upcoming exams"
        Browser->>Flask: POST /chat/api/message
        Flask->>Flask: In-memory safety screening (crisis keywords check)
        Flask->>DB: Fetch user preferences & non-sensitive memories
        Flask->>Gemini: REST POST /v1beta/models/gemini-... (System prompt + History)
        Gemini-->>Flask: Empathetic response + Box Breathing suggestion
        Flask-->>Browser: Render AI reply & suggest guided activity
    end
```

---

## 📁 Project Directory Structure

```text
MindEase-AI/
├── app/
│   ├── content/                # Content registries (activities catalog, support resources)
│   ├── forms/                  # Flask-WTF validated form definitions (Auth, Check-Ins)
│   ├── ml/                     # ML pipeline (TFLite predictor, preprocessor, training script)
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── chat_memory.py      # Non-sensitive chat memory model
│   │   ├── emotion.py          # 5-class facial emotion scan model
│   │   ├── mood.py             # Mood check-in model with valence mapping
│   │   ├── recommendation.py   # Tailored recommendation model
│   │   ├── stress.py           # 11-question PSS stress assessment model
│   │   ├── task.py             # Personal wellness task model
│   │   ├── user.py             # User account and authentication model
│   │   └── user_preferences.py# Onboarding goals, timezone, and memory toggle
│   ├── routes/                 # Modular Flask Blueprints
│   │   ├── admin.py            # Aggregate institutional analytics
│   │   ├── api.py              # REST API endpoints & health check
│   │   ├── auth.py             # Registration, login, logout, onboarding
│   │   ├── chat.py             # AI companion messaging & memory management
│   │   ├── emotion.py          # Facial emotion scan interface and predict endpoint
│   │   ├── mood.py             # Mood check-in and 7-day history
│   │   ├── stress.py           # Stress assessment and scoring
│   │   └── student.py          # Dashboard, check-in hub, activities, support
│   ├── services/               # Core domain logic
│   │   ├── activity_service.py # 6 interactive guided activities definitions
│   │   ├── chat_service.py     # Gemini REST client, safety guards & fallback
│   │   ├── chat_context_service.py # Context prompt builder & memory extractor
│   │   ├── safety_service.py   # In-memory crisis language screening & support card
│   │   ├── task_service.py     # Wellness task management & sync
│   │   └── wellness_calculator.py # Authoritative 0-100 wellness score algorithm
│   ├── static/                 # Frontend static assets
│   │   ├── css/                # MindEase responsive design system
│   │   ├── haar/               # Pre-packaged Haar Cascade face classifier XML
│   │   ├── js/                 # Client-side UI controllers and chart integrations
│   │   └── vendor/             # Local Chart.js library (chart.umd.min.js)
│   ├── templates/              # Jinja2 HTML templates
│   │   ├── admin/              # Institutional aggregate analytics views
│   │   ├── auth/               # Login, registration, and onboarding views
│   │   ├── chat/               # Clean AI wellness companion chat interface
│   │   ├── emotion/            # Camera check-in with borderless frame
│   │   ├── mood/               # Daily mood check-in view
│   │   ├── stress/             # 11-item assessment and result screens
│   │   └── student/            # Dashboard, Check-In Hub, History, Activities, Support
│   ├── utils/                  # Security helpers, validators, and custom decorators
│   ├── config.py               # Environment configuration classes
│   └── extensions.py           # Shared SQLAlchemy, LoginManager, Migrate, CSRF instances
├── docs/                       # Project documentation & real feature screenshots
├── migrations/                 # Alembic database migrations
├── models/                     # Trained CNN weights and deployment TFLite models
│   └── deployment/             # Production champion: challenger_fp32.tflite
├── tests/                      # Automated test suite (336 passing unit/integration tests)
├── .env.example                # Environment variable configuration template
├── .gitignore                  # Git exclusion rules for secrets, DBs, and virtualenvs
├── LICENSE                     # MIT Open Source License
├── requirements.txt            # Production dependencies
└── run.py                      # Application entrypoint script
```

---

## 🛠️ Local Installation & Setup

### Prerequisites
- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **Git**
- Webcam (optional, for facial emotion check-in)

### Step 1: Clone the Repository
```powershell
git clone https://github.com/Ayushpatil15/MindEase-AI.git
cd MindEase-AI
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If execution policies restrict PowerShell scripts, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### Step 3: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Local Environment Variables
Copy `.env.example` to create your local `.env`:
```powershell
copy .env.example .env
```

Open `.env` in your text editor and adjust keys as needed (see [Environment Configuration](#-environment-configuration)). For local development, the default SQLite database path will be used automatically.

### Step 5: Initialize Database Schema
Run database migrations to generate all required tables:
```powershell
flask db upgrade
```

### Step 6: Create an Administrator Account (Optional)
To access the institutional analytics dashboard:
```powershell
flask create-admin
```
Follow the interactive prompt to set an admin username, email, and password.

### Step 7: Launch the Development Server
```powershell
python run.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`**.

---

## ⚙️ Environment Configuration

MindEase AI reads environment variables via `python-dotenv`. The supported variables are:

| Variable Name | Required | Default / Placeholder | Description |
| :--- | :---: | :--- | :--- |
| `APP_ENV` | No | `development` | Environment tier: `development`, `testing`, or `production`. |
| `SECRET_KEY` | **Yes (Prod)** | `dev-secret-key-change-in-production` | Secret cryptographic key for session cookies and CSRF tokens. |
| `DATABASE_URL` | No | `sqlite:///instance/mindease.db` | PostgreSQL or SQLite connection URI (e.g., `postgresql+psycopg2://user:pass@ep-xyz.neon.tech/mindease?sslmode=require`). |
| `GEMINI_API_KEY` | Optional | `your-gemini-api-key-here` | Google Gemini REST API key for the AI companion. If omitted, chat routes gracefully report unavailable. |
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Model identifier for the AI companion (e.g. `gemini-3.5-flash-lite` or `gemini-1.5-flash`). |
| `TFLITE_MODEL_PATH`| No | `models/deployment/challenger_fp32.tflite` | Path to the compiled TFLite neural network artifact. |
| `TFLITE_NUM_THREADS` | No | `4` | Number of CPU threads allocated to TFLite inference. |
| `BEHIND_PROXY` | No | `false` | Enable Werkzeug `ProxyFix` when deployed behind Render or cloud reverse proxies. |
| `MIN_AGGREGATION_GROUP_SIZE` | No | `5` | Minimum student cohort size required for admin aggregate analytics ($K$-anonymity). |
| `EMOTION_MIN_CONFIDENCE` | No | `0.5` | Minimum softmax confidence threshold required for facial emotion scans. |
| `TF_ENABLE_ONEDNN_OPTS` | No | `1` | Enable oneDNN CPU performance optimizations in TensorFlow. |

> [!CAUTION]
> Never commit `.env` or production credentials to version control. The repository `.gitignore` strictly excludes credentials and database manifest files.

---

## 🗄️ Database Setup & Migrations

MindEase AI uses **SQLAlchemy** with **Alembic / Flask-Migrate** for database schema management:

- **Local Development:** Defaults to SQLite stored in `instance/mindease.db`.
- **Production:** Hosted on **Neon Serverless PostgreSQL**. The application automatically normalizes connection strings to `postgresql+psycopg2://` ensuring compatibility across PaaS environments.

### Useful Database Commands
```powershell
# Apply all schema migrations
flask db upgrade

# Generate a new migration after model edits
flask db migrate -m "describe_schema_change"

# Rollback to the previous migration
flask db downgrade
```

---

## 🧪 Testing & Quality Assurance

The application features comprehensive test coverage spanning authentication, scoring algorithms, safety screening, activity flows, timezone handling, rate limiting, and chat routes.

### Running the Full Test Suite
To execute all **336 automated unit and integration tests** using `pytest`:

```powershell
.\venv\Scripts\pytest.exe
```

### Running Core Production Feature Tests
To run core feature tests with Python's standard `unittest` test runner:

```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```

**Verified Test Results:**
- **336 passed** across the full test suite in `pytest` (0 failures, 0 errors).
- Tests cover full edge cases: mathematical score bounds ($0 \le \text{Score} \le 100$), prompt injection sanitization, session security, race-condition mitigation on concurrent task completions, and fallback handling when external AI APIs are unreachable.

---

## 🚀 Production Deployment

MindEase AI is deployed in production on **Render** connected to a serverless **Neon PostgreSQL** database.

### Deployment Flow
1. **Source Control:** Pushes to the `main` branch of `https://github.com/Ayushpatil15/MindEase-AI` automatically trigger a build on Render.
2. **Build Phase:** Dependencies are installed from `requirements.txt`, and database schema is synchronized via Alembic.
3. **Execution Phase:** The application is served using **Gunicorn** with `ProxyFix` middleware ensuring correct HTTPS protocol and IP resolution:
   ```bash
   gunicorn run:app --bind 0.0.0.0:$PORT
   ```
4. **Health Check:** Live uptime and readiness are actively monitored via the `/api/health` and `/health` endpoints.

---

## 🛡️ Privacy, Ethics & Safety Framework

MindEase AI was engineered with ethical safeguards prioritizing student privacy and safety:

> [!IMPORTANT]
> **Non-Clinical Disclaimer:** MindEase AI is a self-awareness, reflection, and wellness education tool. It is **NOT** a diagnostic medical device, psychiatric tool, or substitute for professional medical, clinical, or psychological diagnosis and treatment.

1. **Camera Privacy & Zero Video Storage:**
   - Facial emotion analysis runs entirely as a transient check-in.
   - Decoded image frames are processed in system memory for face cropping and neural classification, and are **immediately discarded**. No photos, video feeds, or biometric templates are ever saved to disk or database.
2. **Strict Journal Isolation:**
   - Student journal reflections and mood notes belong exclusively to the student.
   - Admin views and aggregate queries **never** query or display personal reflections.
3. **In-Memory Safety Net:**
   - The safety service analyzes student input for self-harm or acute distress phrasing in memory.
   - Safety checks trigger immediate, empathetic support cards connecting students to free, confidential 24/7 care (**Tele-MANAS: 14416 / 1800-891-4416** and **Emergency: 112**).
   - Distress evaluations do not create punitive database flags, crisis tags, or administrative alerts.
4. **Institutional $K$-Anonymity:**
   - Campus analytics enforce a minimum cohort size ($K \ge 5$). Cohorts smaller than 5 students are masked to prevent identification.
5. **Student Agency & Memory Consent:**
   - AI companion memory can be reviewed, individually deleted, or toggled off completely by the student at any time.

---

## ⚠️ Known Limitations & Future Roadmap

### Verified Known Limitations
- **Camera Lighting & Angle:** Facial emotion detection accuracy relies on adequate front-facing ambient lighting and standard face alignment; extreme angles or low light can yield unclassified results.
- **Safety Keyword Screening:** The safety screening mechanism is a simple keyword safety net, not a natural-language clinical diagnostic detector.
- **Single AI Provider:** The conversational companion currently integrates with Google Gemini REST API; disruptions to Gemini service trigger graceful fallback messages.

### Planned Future Roadmap
- [ ] Multi-turn voice check-ins using Web Speech API.
- [ ] Integration with wearable sleep and physical activity metrics (Apple HealthKit / Google Health Connect).
- [ ] Multi-lingual support for regional Indian languages across all interactive guided activities.
- [ ] Offline Progressive Web App (PWA) caching for box breathing and sensory grounding exercises.

---

## 📜 License & Acknowledgements

### License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

### Acknowledgements & References
- **Perceived Stress Scale (PSS):** Cohen, S., Kamarck, T., & Mermelstein, R. (1983). A global measure of perceived stress. *Journal of Health and Social Behavior*, 385–396.
- **FER-2013 Dataset:** Facial Expression Recognition challenge dataset for CNN emotion classification benchmarking.
- **Tele-MANAS:** National Tele Mental Health Programme of India (Ministry of Health and Family Welfare, NIMHANS) for student crisis resources (**14416**).
- **Open-Source Libraries:** Built with gratitude to the Flask, SQLAlchemy, OpenCV, TensorFlow, and Chart.js developer communities.

---

<p align="center">
  <b>MindEase AI</b> • Built with care for student mental wellness and academic balance.
</p>

# 🚀 DataGenius PRO - Next-Gen Auto Data Scientist

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

**DataGenius PRO** to zaawansowana platforma do automatycznej analizy danych i Machine Learning, wyposażona w inteligentne agenty AI i AI Mentora mówiącego po polsku.

## ✨ Kluczowe Funkcje

### 🤖 Automatyczne AI Agents
- **Data Understanding Agent**: Automatyczna detekcja typu problemu i kolumny target
- **EDA Agent**: Kompleksowa analiza eksploracyjna danych
- **ML Agent**: Automatyczne trenowanie i tuning modeli (PyCaret)
- **AI Mentor**: Asystent tłumaczący wyniki po polsku (Claude AI)

### 📊 Continuous Monitoring
- Wykrywanie model drift i concept drift
- Performance tracking w czasie rzeczywistym
- Automatyczne alerty
- Scheduler do retrainingu

### 📚 Pipeline Registry
- Historia wszystkich sesji i eksperymentów
- Wersjonowanie modeli
- Reprodukowalne eksperymenty
- Export do MLflow/Weights & Biases

### 📄 Auto Reports
- Raporty EDA (PDF/HTML)
- ML performance reports
- Monitoring dashboards
- Scheduled delivery

---

## 🛠️ Tech Stack

### Core
- **Python 3.11+**
- **Streamlit** - Interactive UI
- **PyCaret** - AutoML framework
- **Anthropic Claude / OpenAI GPT** - LLM dla AI Mentora

### Data & ML
- **Pandas, Polars** - Data manipulation
- **Scikit-learn** - ML algorithms
- **XGBoost, LightGBM, CatBoost** - Gradient boosting
- **SHAP, LIME** - Model interpretability
- **Plotly** - Interactive visualizations

### Infrastructure
- **PostgreSQL / SQLite** - Database
- **Redis** - Caching
- **Docker** - Containerization
- **MLflow** - Experiment tracking

---

## 📦 Instalacja

### 1. Clone Repository
```bash
git clone https://github.com/your-org/datagenius-pro.git
cd datagenius-pro
```

### 2. Setup Environment

#### Opcja A: Conda/Mamba (rekomendowane)
```bash
conda env create -f environment.yml
conda activate datagenius-pro
```

#### Opcja B: pip + venv
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Konfiguracja

Skopiuj template i wypełnij swoimi danymi:
```bash
cp .env.template .env
```

Edytuj `.env` i dodaj swoje klucze API:
```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here  # opcjonalne
```

### 4. Inicjalizacja Bazy Danych
```bash
python scripts/init_db.py
```

### 5. Uruchom Aplikację
```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod: `http://localhost:8501`

---

## 🚀 Quick Start

### 1. Załaduj Dane
- Prześlij swój CSV/Excel/JSON
- Lub wybierz przykładowy dataset (Iris, Titanic, House Prices)

### 2. Eksploracja (EDA)
- Automatyczna analiza statystyczna
- Wykrywanie outliers i braków danych
- Interaktywne wizualizacje
- Analiza korelacji

### 3. Trenowanie Modelu
- Automatyczne porównanie 15+ algorytmów
- Hyperparameter tuning
- Feature importance (SHAP)
- Explainable AI

### 4. AI Mentor
- Zadawaj pytania po polsku
- Otrzymuj wyjaśnienia wyników
- Konkretne rekomendacje
- Pomoc w feature engineering

### 5. Monitoring
- Śledzenie performance
- Drift detection
- Automatyczne alerty

---

## 📁 Struktura Projektu

```
datagenius-pro/
│
├── app.py                      # 🎯 Główny punkt wejścia
├── requirements.txt            # 📦 Zależności
├── environment.yml             # 🐍 Conda environment
├── Makefile                    # 🛠️ Automation commands
│
├── config/                     # ⚙️ Konfiguracja
│   ├── settings.py            # Centralny config (Pydantic)
│   ├── logging_config.py      # Logowanie
│   ├── model_registry.py      # Rejestr modeli ML
│   └── constants.py           # Stałe aplikacji
│
├── core/                       # 🧠 Rdzeń aplikacji
│   ├── base_agent.py          # Bazowa klasa agentów
│   ├── llm_client.py          # Wrapper dla LLM
│   ├── data_loader.py         # Loader danych
│   ├── data_validator.py      # Walidacja danych
│   ├── state_manager.py       # Session state
│   └── utils.py               # Utility functions
│
├── agents/                     # 🤖 AI Agents
│   ├── data_understanding/    # Agent rozumienia danych
│   ├── eda/                   # Agent EDA
│   ├── preprocessing/         # Agent preprocessingu
│   ├── ml/                    # Agent ML
│   ├── monitoring/            # Agent monitoringu
│   └── mentor/                # AI Mentor
│
├── frontend/                   # 🎨 Frontend (Streamlit)
│   ├── pages/                 # Strony aplikacji
│   ├── components/            # Komponenty UI
│   └── styling/               # Style i theme
│
├── db/                         # 🗄️ Database
│   ├── models.py              # SQLAlchemy models
│   ├── crud.py                # CRUD operations
│   └── schemas/               # SQL schemas
│
├── tests/                      # 🧪 Testy
│   ├── unit/                  # Testy jednostkowe
│   ├── integration/           # Testy integracyjne
│   └── e2e/                   # Testy end-to-end
│
├── docs/                       # 📚 Dokumentacja
│   ├── ARCHITECTURE.md        # Architektura systemu
│   ├── API.md                 # Dokumentacja API
│   └── guides/                # Przewodniki
│
└── data/                       # 📊 Dane
    ├── samples/               # Przykładowe datasety
    ├── uploads/               # Przesłane pliki
    └── processed/             # Przetworzone dane
```

---

## 🎯 Przykłady Użycia

### Podstawowy Workflow

```python
from core.data_loader import get_data_loader
from agents.eda.eda_orchestrator import EDAOrchestrator
from agents.ml.ml_orchestrator import MLOrchestrator

# 1. Załaduj dane
loader = get_data_loader()
data = loader.load("data/samples/iris.csv")

# 2. EDA
eda_agent = EDAOrchestrator()
eda_results = eda_agent.run(data=data, target_column="species")

# 3. Trenuj model
ml_agent = MLOrchestrator()
ml_results = ml_agent.run(
    data=data,
    target_column="species",
    problem_type="classification"
)

# 4. Wyniki
print(f"Best Model: {ml_results.data['summary']['best_model']}")
print(f"Accuracy: {ml_results.data['summary']['best_score']:.4f}")
```

### AI Mentor

```python
from agents.mentor.mentor_orchestrator import MentorOrchestrator

mentor = MentorOrchestrator()

# Zapytaj AI Mentora
response = mentor.run(
    query="Jak mogę poprawić wyniki mojego modelu?",
    context={
        "ml_results": ml_results.data,
        "eda_results": eda_results.data
    }
)

print(response.data["response"])
```

---

## 🔧 Konfiguracja

### Environment Variables

Najważniejsze zmienne w `.env`:

```bash
# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/datagenius
# lub SQLite dla developmentu
DATABASE_URL=sqlite:///./data/datagenius.db

# ML Settings
ENABLE_HYPERPARAMETER_TUNING=true
ENABLE_ENSEMBLE=true
MAX_TRAINING_TIME_MINUTES=30

# Monitoring
ENABLE_MONITORING=true
MONITORING_SCHEDULE=weekly

# Feature Flags
ENABLE_AI_MENTOR=true
ENABLE_AUTO_EDA=true
ENABLE_AUTO_ML=true
```

---

## 🐳 Docker

### Build i Run

```bash
# Build image
docker-compose build

# Start containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Aplikacja będzie dostępna pod: `http://localhost:8501`

---

## 🧪 Testing

### Uruchom wszystkie testy
```bash
make test
```

### Testy jednostkowe
```bash
pytest tests/unit/ -v
```

### Testy integracyjne
```bash
pytest tests/integration/ -v
```

### Coverage
```bash
pytest --cov=. --cov-report=html
```

---

## 📊 Monitoring & MLOps

### MLflow Integration

```bash
# Start MLflow server
mlflow ui --backend-store-uri sqlite:///mlflow.db

# W .env
ENABLE_MLFLOW_LOGGING=true
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Weights & Biases

```bash
# W .env
ENABLE_WANDB_LOGGING=true
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=datagenius-pro
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run linters
make lint

# Format code
make format
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👥 Team

### Zespoły
- **Z1 - EDA**: Exploratory Data Analysis
- **Z2 - ML**: Machine Learning Pipeline
- **Z3 - Backend**: Application Logic
- **Z4 - Frontend**: Streamlit UI
- **Z5 - Database**: Data Persistence
- **Z6 - QA**: Testing & Deployment
- **Z7 - UX/Docs**: UX Design & Documentation

---

## 📞 Support

- 📧 Email: support@datagenius-pro.com
- 💬 Discord: [Join our community](https://discord.gg/datagenius)
- 🐙 GitHub Issues: [Report a bug](https://github.com/datagenius-pro/issues)
- 📚 Documentation: [docs.datagenius-pro.com](https://docs.datagenius-pro.com)

---

## 🗺️ Roadmap

### v2.1 (Q1 2025)
- [ ] Multi-user support
- [ ] Real-time predictions API
- [ ] Advanced ensemble methods
- [ ] Custom model marketplace

### v2.2 (Q2 2025)
- [ ] Deep Learning support
- [ ] Time series forecasting
- [ ] AutoML hyperparameter optimization
- [ ] Multi-language support

### v3.0 (Q3 2025)
- [ ] Enterprise features
- [ ] Role-based access control
- [ ] Advanced integrations
- [ ] Cloud deployment options

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ by DataGenius Team**

*Making Data Science accessible to everyone!*
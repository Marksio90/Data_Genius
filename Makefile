# DataGenius PRO - Makefile
# Automation commands for development and deployment

.PHONY: help install setup run test lint format clean docker-build docker-up docker-down db-init db-migrate docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)DataGenius PRO - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ===========================================
# Setup & Installation
# ===========================================

install: ## Install dependencies using pip
	@echo "$(CYAN)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(CYAN)Installing dev dependencies...$(NC)"
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

setup: ## Setup project (create .env, init db)
	@echo "$(CYAN)Setting up DataGenius PRO...$(NC)"
	cp .env.template .env
	@echo "$(YELLOW)⚠ Please edit .env file with your configuration$(NC)"
	python scripts/setup_env.py
	python scripts/init_db.py
	@echo "$(GREEN)✓ Setup complete$(NC)"

setup-conda: ## Setup conda environment
	@echo "$(CYAN)Creating conda environment...$(NC)"
	conda env create -f environment.yml
	@echo "$(GREEN)✓ Conda environment created$(NC)"
	@echo "$(YELLOW)Activate with: conda activate datagenius-pro$(NC)"

# ===========================================
# Running the Application
# ===========================================

run: ## Run Streamlit application
	@echo "$(CYAN)Starting DataGenius PRO...$(NC)"
	streamlit run app.py

run-dev: ## Run in development mode with auto-reload
	@echo "$(CYAN)Starting in development mode...$(NC)"
	streamlit run app.py --server.runOnSave true

run-api: ## Run FastAPI backend
	@echo "$(CYAN)Starting FastAPI server...$(NC)"
	uvicorn backend.api.routes:app --reload --host 0.0.0.0 --port 8000

# ===========================================
# Testing
# ===========================================

test: ## Run all tests
	@echo "$(CYAN)Running tests...$(NC)"
	pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	@echo "$(CYAN)Running integration tests...$(NC)"
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	@echo "$(CYAN)Running e2e tests...$(NC)"
	pytest tests/e2e/ -v

test-cov: ## Run tests with coverage report
	@echo "$(CYAN)Running tests with coverage...$(NC)"
	pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

# ===========================================
# Code Quality
# ===========================================

lint: ## Run linters (flake8, mypy)
	@echo "$(CYAN)Running linters...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	mypy . --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(CYAN)Formatting code...$(NC)"
	black .
	isort .
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting
	@echo "$(CYAN)Checking code format...$(NC)"
	black . --check
	isort . --check
	@echo "$(GREEN)✓ Format check complete$(NC)"

# ===========================================
# Database
# ===========================================

db-init: ## Initialize database
	@echo "$(CYAN)Initializing database...$(NC)"
	python scripts/init_db.py
	@echo "$(GREEN)✓ Database initialized$(NC)"

db-migrate: ## Run database migrations
	@echo "$(CYAN)Running migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(NC)"

db-migrate-create: ## Create new migration (use NAME=migration_name)
	@echo "$(CYAN)Creating new migration...$(NC)"
	alembic revision --autogenerate -m "$(NAME)"
	@echo "$(GREEN)✓ Migration created$(NC)"

db-reset: ## Reset database (WARNING: deletes all data)
	@echo "$(RED)⚠ This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python scripts/init_db.py --reset; \
		echo "$(GREEN)✓ Database reset$(NC)"; \
	fi

# ===========================================
# Docker
# ===========================================

docker-build: ## Build Docker image
	@echo "$(CYAN)Building Docker image...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-up: ## Start Docker containers
	@echo "$(CYAN)Starting Docker containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Containers started$(NC)"

docker-down: ## Stop Docker containers
	@echo "$(CYAN)Stopping Docker containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Containers stopped$(NC)"

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-shell: ## Open shell in app container
	docker-compose exec app bash

# ===========================================
# Documentation
# ===========================================

docs: ## Build documentation
	@echo "$(CYAN)Building documentation...$(NC)"
	cd docs && mkdocs build
	@echo "$(GREEN)✓ Documentation built$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(CYAN)Serving documentation...$(NC)"
	cd docs && mkdocs serve

# ===========================================
# Data & Models
# ===========================================

generate-samples: ## Generate sample datasets
	@echo "$(CYAN)Generating sample data...$(NC)"
	python scripts/generate_samples.py
	@echo "$(GREEN)✓ Sample data generated$(NC)"

clean-uploads: ## Clean uploaded files
	@echo "$(CYAN)Cleaning uploads...$(NC)"
	find data/uploads -type f ! -name '.gitkeep' -delete
	@echo "$(GREEN)✓ Uploads cleaned$(NC)"

clean-models: ## Clean saved models
	@echo "$(CYAN)Cleaning models...$(NC)"
	find models -type f ! -name '.gitkeep' ! -name 'README.md' -delete
	@echo "$(GREEN)✓ Models cleaned$(NC)"

# ===========================================
# Cleanup
# ===========================================

clean: ## Clean temporary files
	@echo "$(CYAN)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean clean-uploads clean-models ## Clean everything
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

# ===========================================
# Deployment
# ===========================================

deploy-staging: ## Deploy to staging
	@echo "$(CYAN)Deploying to staging...$(NC)"
	@echo "$(YELLOW)Not implemented yet$(NC)"

deploy-prod: ## Deploy to production
	@echo "$(RED)⚠ Production deployment$(NC)"
	@echo "$(YELLOW)Not implemented yet$(NC)"

# ===========================================
# Monitoring
# ===========================================

logs: ## View application logs
	@echo "$(CYAN)Showing logs...$(NC)"
	tail -f logs/app.log

monitor: ## Start monitoring dashboard
	@echo "$(CYAN)Starting monitoring...$(NC)"
	@echo "$(YELLOW)Not implemented yet$(NC)"

# ===========================================
# Utility
# ===========================================

check: lint test ## Run all checks (lint + test)
	@echo "$(GREEN)✓ All checks passed$(NC)"

version: ## Show version info
	@echo "$(CYAN)DataGenius PRO$(NC)"
	@python -c "from config.settings import settings; print(f'Version: {settings.APP_VERSION}')"
	@python --version

info: ## Show system info
	@echo "$(CYAN)System Information:$(NC)"
	@python --version
	@echo "Pip: $$(pip --version)"
	@echo "Conda: $$(conda --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"
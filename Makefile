.PHONY: dev start stop backend frontend services logs clean install help

# Default target
help:
	@echo "🧠 EXPLAINIUM - Quick Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev      - Start development environment (quick start)"
	@echo "  make start    - Deploy enhanced system (full)"
	@echo "  make stop     - Stop all services"
	@echo ""
	@echo "Individual Services:"
	@echo "  make backend  - Start backend only"
	@echo "  make frontend - Start frontend only"
	@echo "  make services - Start supporting services only"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs     - View all service logs"
	@echo "  make clean    - Clean up containers and volumes"
	@echo "  make install  - Install frontend dependencies"

# Development mode - quick start via enhanced deploy script
dev:
	@echo "🚀 Starting development (quick)..."
	@./deploy-enhanced.sh --quick

# Full deploy
start:
	@echo "⚡ Deploying enhanced EXPLAINIUM..."
	@./deploy-enhanced.sh

# Stop all services
stop:
	@echo "🛑 Stopping all services..."
	@docker-compose down --remove-orphans

# Backend only
backend:
	@echo "🔧 Starting backend services..."
	@docker-compose up app celery_worker celery_beat

# Frontend only
frontend:
	@echo "⚛️  Starting frontend..."
	@cd src/frontend && npm start

# Supporting services only
services:
	@echo "📦 Starting supporting services..."
	@docker-compose up -d db redis elasticsearch tika

# View logs
logs:
	@docker-compose logs -f

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose down -v
	@docker system prune -f

# Install frontend dependencies
install:
	@echo "📦 Installing frontend dependencies..."
	@cd src/frontend && npm install
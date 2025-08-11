#!/bin/bash

# EXPLAINIUM - Application Startup Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=================================================================="
echo -e "${BLUE}EXPLAINIUM - Starting Application${NC}"
echo "=================================================================="

# Move to repo root (directory of this script)
cd "$(dirname "$0")" || exit 1

# Activate virtual environment (support .venv or venv)
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
  source venv/bin/activate
else
  echo -e "${BLUE}No virtual environment found (.venv/venv). Continuing with system Python...${NC}"
fi

# Environment safeguards for local LLM stability
# Disable llama.cpp by default on Linux unless explicitly enabled
UNAME_S=$(uname -s)
if [ "$UNAME_S" = "Linux" ]; then
  export EXPLAINIUM_DISABLE_LLM=${EXPLAINIUM_DISABLE_LLM:-true}
fi
# Do not use LLM for media (video/image) by default; can be enabled by setting EXPLAINIUM_LLM_MEDIA=true
export EXPLAINIUM_LLM_MEDIA=${EXPLAINIUM_LLM_MEDIA:-false}

# Function to start backend
start_backend() {
    echo -e "${GREEN}Starting Backend API...${NC}"
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
}

# Function to start frontend
start_frontend() {
    echo -e "${GREEN}Starting Frontend Dashboard...${NC}"
    streamlit run src/frontend/knowledge_table.py --server.port 8501 --server.address localhost &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
}

# Start services
start_backend
sleep 3  # Give backend time to start
start_frontend

echo ""
echo "=================================================================="
echo -e "${GREEN}EXPLAINIUM is now running.${NC}"
echo "=================================================================="
echo "Frontend Dashboard: http://localhost:8501"
echo "Backend API:        http://localhost:8000"
echo "API Documentation:  http://localhost:8000/docs"
echo "=================================================================="
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
trap 'echo -e "\n${BLUE}Shutting down...${NC}"; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait

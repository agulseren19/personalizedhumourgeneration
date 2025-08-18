#!/bin/bash

# 🃏 AI Cards Against Humanity - Quick Start Script
# This script starts both the backend API server and frontend dev server

echo "🃏 Starting AI Cards Against Humanity System..."
echo "================================================"

# Check if we're in the right directory
if [ ! -d "agent_system" ] || [ ! -d "nextjs-boilerplate-main" ]; then
    echo "❌ Error: Please run this script from the CAH project root directory"
    echo "   Expected directories: agent_system/ and nextjs-boilerplate-main/"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

if ! command_exists npm; then
    echo "❌ Node.js/npm is required but not installed"
    exit 1
fi

if ! command_exists pip; then
    echo "❌ pip is required but not installed"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Check for API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. Please set it for humor generation:"
    echo "   export OPENAI_API_KEY='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'"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ OpenAI API key detected"
fi

# Install backend dependencies
echo ""
echo "📦 Installing backend dependencies..."
cd agent_system
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found in agent_system/"
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi
echo "✅ Backend dependencies installed"

# Install frontend dependencies
echo ""
echo "📦 Installing frontend dependencies..."
cd ../nextjs-boilerplate-main

if [ ! -f "package.json" ]; then
    echo "❌ package.json not found in nextjs-boilerplate-main/"
    exit 1
fi

npm install
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi
echo "✅ Frontend dependencies installed"

# Create .env.local if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
    echo "✅ Created .env.local for frontend"
fi

cd ..

# Function to start backend
start_backend() {
    echo "🚀 Starting CrewAI backend API server..."
    python agent_system/api/cah_crewai_api.py &
    BACKEND_PID=$!
    echo "CrewAI Backend PID: $BACKEND_PID"
    
    # Wait for backend to start
    echo "⏳ Waiting for backend to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000 > /dev/null 2>&1; then
            echo "✅ Backend is ready at http://localhost:8000"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo "❌ Backend failed to start within 30 seconds"
            kill $BACKEND_PID 2>/dev/null
            exit 1
        fi
    done
}

# Function to start frontend
start_frontend() {
    echo "🚀 Starting frontend dev server..."
    cd nextjs-boilerplate-main
    npm run dev &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
    cd ..
    
    # Wait for frontend to start
    echo "⏳ Waiting for frontend to start..."
    for i in {1..30}; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            echo "✅ Frontend is ready at http://localhost:3000"
            break
        fi
        sleep 2
        if [ $i -eq 30 ]; then
            echo "❌ Frontend failed to start within 60 seconds"
            kill $FRONTEND_PID 2>/dev/null
            exit 1
        fi
    done
}

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend stopped"
    fi
    echo "👋 Goodbye!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo ""
echo "🚀 Starting servers..."
echo "======================"

# Start backend
start_backend

# Start frontend
start_frontend

echo ""
echo "🎉 AI Cards Against Humanity is ready!"
echo "======================================"
echo ""
echo "🌐 Access your application:"
echo "   • Frontend: http://localhost:3000"
echo "   • Game:     http://localhost:3000/cah"
echo "   • API:      http://localhost:8000"
echo ""
echo "📋 What you can do:"
echo "   • Play the CAH game with AI humor generation"
echo "   • Rate responses to train the AI to your preferences"
echo "   • View analytics showing how the AI learns about you"
echo "   • Explore different AI personas and humor styles"
echo ""
echo "⚠️  Note: Keep this terminal open. Press Ctrl+C to stop both servers."
echo ""

# Keep script running and show logs
echo "📊 System Status:"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Waiting for shutdown signal..."

# Wait for user interrupt
while true; do
    sleep 1
    
    # Check if processes are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process died unexpectedly"
        cleanup
    fi
    
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend process died unexpectedly"
        cleanup
    fi
done 
#!/bin/bash

# BTC Prediction MCP Server Startup Script

echo "🚀 Starting BTC Prediction MCP Server..."

# Set environment variables
export MCP_SERVER_NAME="btc-predictor"
export MCP_SERVER_VERSION="1.0.0"

# Check if FastMCP is installed
if ! python3 -c "import fastmcp" 2>/dev/null; then
    echo "📦 Installing FastMCP..."
    pip install fastmcp
fi

# Check if pydantic is installed
if ! python3 -c "import pydantic" 2>/dev/null; then
    echo "📦 Installing pydantic..."
    pip install pydantic
fi

# Kill any existing MCP server on port 5002
echo "🔍 Checking for existing MCP server..."
lsof -ti:5002 | xargs -r kill -9 2>/dev/null

# Navigate to MCP server directory
cd "$(dirname "$0")"

# Start the MCP server
echo "✅ Starting MCP server on port 5002..."
python3 -m fastmcp run mcp_server.py

# Alternative: Run with uvicorn for production
# uvicorn mcp_server:app --host 0.0.0.0 --port 5002
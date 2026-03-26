#!/bin/bash
# MARC2 project setup
set -e

echo "=== MARC2 Setup ==="

# 1. Clone ARC-AGI2 dataset
if [ ! -d "data/arc-agi2" ]; then
    echo "Cloning ARC-AGI2 dataset..."
    mkdir -p data
    git clone https://github.com/arcprize/ARC-AGI-2.git data/arc-agi2
    echo "  Done. $(ls data/arc-agi2/training/*.json 2>/dev/null | wc -l | tr -d ' ') training tasks, $(ls data/arc-agi2/evaluation/*.json 2>/dev/null | wc -l | tr -d ' ') evaluation tasks."
else
    echo "ARC-AGI2 dataset already present."
fi

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 3. Initialize database
echo "Initializing database..."
python -c "
import sqlite3
conn = sqlite3.connect('marc2.db')
conn.executescript(open('schema.sql').read())
conn.close()
print('  marc2.db initialized.')
"

echo ""
echo "=== Setup complete ==="
echo "Next: python solve.py solve --model claude-sonnet-4-6 --split training --concurrency 8"

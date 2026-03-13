#!/bin/bash

# ECG Monitor Launcher Script for macOS
# This script launches the ECG Monitor application

echo "🚀 Starting ECG Monitor..."
echo "📱 Application: ECG_Monitor.app"
echo "📁 Location: $(pwd)/dist/ECG_Monitor.app"
echo ""

# Check if the app exists
if [ ! -d "dist/ECG_Monitor.app" ]; then
    echo "❌ Error: ECG_Monitor.app not found!"
    echo "Please run 'pyinstaller ecg_monitor.spec --clean' first to build the application."
    exit 1
fi

# Launch the application
echo "🎯 Launching ECG Monitor..."
open "dist/ECG_Monitor.app"

echo "✅ ECG Monitor launched successfully!"
echo "💡 Tip: You can also double-click ECG_Monitor.app in Finder to launch it."

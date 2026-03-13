#!/usr/bin/env python3
"""
Force activate demo mode script
This script will force activate demo mode to show fixed ECG values
"""

import sys
import os

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from dashboard.dashboard import Dashboard
    from PyQt5.QtWidgets import QApplication
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create dashboard instance
    dashboard = Dashboard()
    
    # Force activate demo mode
    print("Force activating demo mode...")
    
    # Check if demo toggle exists and activate it
    if hasattr(dashboard, 'demo_toggle'):
        dashboard.demo_toggle.setChecked(True)
        dashboard.demo_toggle.setText("Demo Mode: ON")
        print("Demo mode activated successfully!")
        
        # Trigger demo mode setup
        if hasattr(dashboard, 'toggle_demo_mode'):
            dashboard.toggle_demo_mode(True)
            print("Demo mode setup completed!")
    else:
        print("Demo toggle not found!")
        
    print("Demo mode should now be active with fixed values:")
    print("- BPM: 100")
    print("- PR: 174ms") 
    print("- QRS: 90ms")
    print("- QT/QTc: 286/369ms")
    
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to exit...")

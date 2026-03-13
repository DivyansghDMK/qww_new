#!/usr/bin/env python3
"""
Dashboard Configuration File
Edit this file to change dashboard settings without modifying the main code.
"""

# Background Configuration
DASHBOARD_BACKGROUND = {
    # Set to False to disable animated backgrounds (better performance)
    "use_gif_background": False,
    
    # Choose your preferred background:
    # - "plasma.gif" (not available in current assets)
    # - "tenor.gif" (232KB - recommended for performance)
    # - "v.gif" (6.0MB - may affect performance)
    # - "none" (solid color background)
    "preferred_background": "none",
    
    # Custom solid color background (used when GIF is disabled or not found)
    "solid_color": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef)"
}

# Performance Settings
PERFORMANCE = {
    # Reduce animation frame rate for better performance
    "gif_frame_rate": 30,  # Frames per second
    
    # Enable/disable heartbeat animation
    "enable_heartbeat_animation": True,
    
    # Enable/disable background GIF
    "enable_background_gif": False
}

# UI Settings
UI = {
    # Default theme
    "default_theme": "light",  # "light", "dark", or "medical"
    
    # Show debug information in console
    "show_debug_info": True,
    
    # Enable asset path testing at startup
    "test_assets_at_startup": True
}

# Quick Background Presets
BACKGROUND_PRESETS = {
    "performance": {
        "use_gif_background": False,
        "preferred_background": "none",
        "description": "Best performance - solid color background"
    },
    "balanced": {
        "use_gif_background": True,
        "preferred_background": "tenor.gif",
        "description": "Good balance - small animated background"
    },
    "animated": {
        "use_gif_background": True,
        "preferred_background": "v.gif",
        "description": "Full animation - may affect performance"
    }
}

def get_background_config(preset_name=None):
    """
    Get background configuration, optionally using a preset.
    
    Args:
        preset_name (str): Name of preset to use ("performance", "balanced", "animated")
    
    Returns:
        dict: Background configuration
    """
    if preset_name and preset_name in BACKGROUND_PRESETS:
        return BACKGROUND_PRESETS[preset_name]
    return DASHBOARD_BACKGROUND

def print_available_presets():
    """Print all available background presets."""
    print("Available Background Presets:")
    for name, config in BACKGROUND_PRESETS.items():
        print(f"  {name}: {config['description']}")
        print(f"    - GIF Background: {config['use_gif_background']}")
        print(f"    - Preferred: {config['preferred_background']}")
        print()

if __name__ == "__main__":
    print("Dashboard Configuration File")
    print("=" * 40)
    print(f"Current Background: {DASHBOARD_BACKGROUND['preferred_background']}")
    print(f"GIF Background Enabled: {DASHBOARD_BACKGROUND['use_gif_background']}")
    print()
    print_available_presets()
    
    print("To change the background:")
    print("1. Edit this file and change the values")
    print("2. Or use a preset: get_background_config('performance')")
    print("3. Restart the dashboard application")


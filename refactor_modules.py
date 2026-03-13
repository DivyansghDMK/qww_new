#!/usr/bin/env python3
"""
Modular refactoring script for ECG codebase
Splits large files into smaller modules while preserving all logic
"""
import os
import re
import ast
from pathlib import Path

def analyze_file_structure(filepath):
    """Analyze Python file structure to identify classes and functions"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    classes = []
    functions = []
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append({
                'name': node.name,
                'line': node.lineno,
                'end_line': node.end_lineno if hasattr(node, 'end_lineno') else None
            })
        elif isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
            functions.append({
                'name': node.name,
                'line': node.lineno,
                'end_line': node.end_lineno if hasattr(node, 'end_lineno') else None
            })
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(content, node))
    
    return {
        'classes': classes,
        'functions': functions,
        'imports': imports,
        'total_lines': len(content.split('\n'))
    }

def create_module_structure():
    """Create the new modular folder structure"""
    base_path = Path('src')
    
    folders = [
        'ecg/ui',
        'ecg/serial',
        'ecg/metrics',
        'ecg/plotting',
        'ecg/utils',
        'dashboard/ui',
        'dashboard/widgets',
        'dashboard/metrics',
        'reports/generators',
        'reports/templates'
    ]
    
    for folder in folders:
        (base_path / folder).mkdir(parents=True, exist_ok=True)
        # Create __init__.py
        init_file = base_path / folder / '__init__.py'
        if not init_file.exists():
            init_file.write_text(f'"""{folder.replace("/", " ")} module"""\n')

if __name__ == '__main__':
    print("Creating modular structure...")
    create_module_structure()
    print("Structure created!")
    
    # Analyze main files
    files_to_analyze = [
        'src/ecg/twelve_lead_test.py',
        'src/dashboard/dashboard.py',
        'src/ecg/ecg_report_generator.py'
    ]
    
    for filepath in files_to_analyze:
        if os.path.exists(filepath):
            print(f"\nAnalyzing {filepath}...")
            structure = analyze_file_structure(filepath)
            print(f"  Total lines: {structure['total_lines']}")
            print(f"  Classes: {len(structure['classes'])}")
            print(f"  Functions: {len(structure['functions'])}")
            for cls in structure['classes']:
                print(f"    - {cls['name']} (line {cls['line']})")

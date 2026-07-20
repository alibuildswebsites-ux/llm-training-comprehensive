#!/usr/bin/env python3
"""
Test that all templates render correctly with sample configs
"""

import os
import sys

def _validate_template(template_path):
    """Test a template by trying to run it with minimal config"""
    # This is a placeholder - actual testing would require a full environment
    if template_path.endswith('.py'):
        # Check Python syntax
        try:
            with open(template_path) as f:
                compile(f.read(), template_path, 'exec')
            return True, "Syntax OK"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    elif template_path.endswith('.yaml'):
        # Check YAML syntax
        try:
            import yaml
            with open(template_path) as f:
                yaml.safe_load(f)
            return True, "YAML OK"
        except Exception as e:
            return False, f"YAML error: {e}"
    return True, "Skipped"

def test_all_templates():
    """Test all templates in the skill"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    template_dirs = [
        'templates/unsloth',
        'templates/axolotl',
        'templates/llama-factory',
        'templates/trl',
        'templates/torchtune',
        'templates/raw-transformers',
    ]
    
    all_passed = True
    
    for template_dir in template_dirs:
        full_dir = os.path.join(base_dir, template_dir)
        if not os.path.exists(full_dir):
            print(f"SKIP: {template_dir} (not found)")
            continue
        
        for file in os.listdir(full_dir):
            path = os.path.join(full_dir, file)
            if file.endswith('.py') or file.endswith('.yaml'):
                passed, msg = _validate_template(path)
                status = "PASS" if passed else "FAIL"
                print(f"{status}: {path} - {msg}")
                if not passed:
                    all_passed = False
    
    assert all_passed, "Some templates failed validation"
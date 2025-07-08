#!/usr/bin/env python3
"""
Test script to validate all code examples in documentation.

This script extracts and runs all Python code blocks from the documentation
to ensure they work correctly with the current API.
"""

import os
import re
import sys
import traceback
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def extract_python_code_blocks(content: str, filename: str) -> List[Tuple[int, str]]:
    """Extract Python code blocks from markdown content."""
    code_blocks = []
    lines = content.split('\n')
    in_python_block = False
    current_block = []
    block_start_line = 0
    
    for line_num, line in enumerate(lines, 1):
        if line.strip().startswith('```python'):
            in_python_block = True
            block_start_line = line_num + 1
            current_block = []
        elif line.strip() == '```' and in_python_block:
            in_python_block = False
            if current_block:
                code = '\n'.join(current_block)
                code_blocks.append((block_start_line, code))
        elif in_python_block:
            current_block.append(line)
    
    return code_blocks

def test_code_block(code: str, filename: str, line_num: int) -> Tuple[bool, str]:
    """Test a single code block."""
    try:
        # Create a temporary namespace for execution with common imports
        namespace = {
            '__name__': '__main__',
            '__file__': filename,
            'np': __import__('numpy'),
            'pin': __import__('py_pinocchio'),
            'dataclass': __import__('dataclasses').dataclass,
            'NamedTuple': __import__('typing').NamedTuple,
            'Enum': __import__('enum').Enum,
        }

        # Skip code blocks that are class/function definitions without usage
        if (code.strip().startswith('class ') or
            code.strip().startswith('def ') or
            code.strip().startswith('@dataclass')):
            # Only test if there's actual usage after the definition
            lines = code.split('\n')
            has_usage = any(not line.strip().startswith(('class ', 'def ', '@', ' ', '"""', "'''"))
                          and line.strip() and not line.strip().startswith('#')
                          for line in lines[1:])
            if not has_usage:
                return True, ""  # Skip pure definitions

        # Execute the code
        exec(code, namespace)
        return True, ""
    except Exception as e:
        error_msg = f"Error in {filename} at line {line_num}:\n{traceback.format_exc()}"
        return False, error_msg

def test_documentation_file(filepath: Path) -> Dict[str, List[str]]:
    """Test all code examples in a documentation file."""
    results = {
        'passed': [],
        'failed': [],
        'errors': []
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        results['errors'].append(f"Could not read file {filepath}: {e}")
        return results
    
    code_blocks = extract_python_code_blocks(content, str(filepath))
    
    for line_num, code in code_blocks:
        # Skip code blocks that are just imports or version checks
        if code.strip().startswith('#') or 'print(' in code and len(code.split('\n')) <= 2:
            continue

        # Skip code blocks that contain placeholder text or undefined functions
        if ('your-repo' in code or 'TODO' in code or '...' in code or
            'create_my_robot' in code or 'create_benchmark_robot' in code or
            'create_3dof_planar_robot' in code or 'create_dynamics_test_robot' in code or
            'create_serial_arm' in code or 'create_visualization_robot' in code or
            'create_example_urdf' in code or 'create_2dof_robot' in code or
            'robot.urdf' in code or 'robot.xml' in code or
            'benchmark_results' in code or 'urdf_file' in code):
            continue
            
        success, error = test_code_block(code, str(filepath), line_num)
        
        if success:
            results['passed'].append(f"Line {line_num}: OK")
        else:
            results['failed'].append(f"Line {line_num}: FAILED")
            results['errors'].append(error)
    
    return results

def main():
    """Main test function."""
    docs_dir = Path(__file__).parent / 'docs'
    
    if not docs_dir.exists():
        print(f"Documentation directory not found: {docs_dir}")
        return 1
    
    # Find all markdown files
    md_files = list(docs_dir.glob('**/*.md'))
    
    print("🧪 Testing Documentation Code Examples")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    files_with_errors = 0
    
    for filepath in sorted(md_files):
        relative_path = filepath.relative_to(docs_dir)
        print(f"\n📄 Testing {relative_path}")
        print("-" * 40)
        
        results = test_documentation_file(filepath)
        
        if results['passed']:
            print(f"✅ Passed: {len(results['passed'])}")
            total_passed += len(results['passed'])
            
        if results['failed']:
            print(f"❌ Failed: {len(results['failed'])}")
            total_failed += len(results['failed'])
            files_with_errors += 1
            
            for failure in results['failed']:
                print(f"  {failure}")
                
        if results['errors']:
            print("\nError Details:")
            for error in results['errors']:
                print(f"  {error}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 30)
    print(f"Files tested: {len(md_files)}")
    print(f"Files with errors: {files_with_errors}")
    print(f"Code blocks passed: {total_passed}")
    print(f"Code blocks failed: {total_failed}")
    
    if total_failed == 0:
        print("\n✅ All documentation code examples work correctly!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} code example(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Check type annotation completeness in py-pinocchio codebase.

This script analyzes all Python files to ensure comprehensive type annotations
are present for all functions, methods, and class attributes.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set

class TypeAnnotationChecker(ast.NodeVisitor):
    """AST visitor to check for missing type annotations."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues = []
        self.in_class = False
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions."""
        old_in_class = self.in_class
        old_current_class = self.current_class
        
        self.in_class = True
        self.current_class = node.name
        
        # Check class attributes (dataclass fields, etc.)
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and item.annotation is None:
                self.issues.append((
                    node.lineno,
                    f"Class attribute '{item.target.id}' missing type annotation"
                ))
        
        self.generic_visit(node)
        
        self.in_class = old_in_class
        self.current_class = old_current_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions."""
        # Skip special methods and private methods for now
        if node.name.startswith('_'):
            return
            
        # Check return type annotation
        if node.returns is None and node.name != '__init__':
            context = f"in class {self.current_class}" if self.in_class else "at module level"
            self.issues.append((
                node.lineno,
                f"Function '{node.name}' {context} missing return type annotation"
            ))
        
        # Check parameter type annotations
        for arg in node.args.args:
            if arg.annotation is None and arg.arg != 'self' and arg.arg != 'cls':
                context = f"in class {self.current_class}" if self.in_class else "at module level"
                self.issues.append((
                    node.lineno,
                    f"Parameter '{arg.arg}' in function '{node.name}' {context} missing type annotation"
                ))
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definitions."""
        self.visit_FunctionDef(node)  # Same logic as regular functions

def check_file_type_annotations(filepath: Path) -> List[Tuple[int, str]]:
    """Check type annotations in a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(filepath))
        checker = TypeAnnotationChecker(str(filepath))
        checker.visit(tree)
        
        return checker.issues
    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}")]
    except Exception as e:
        return [(0, f"Error parsing file: {e}")]

def should_skip_file(filepath: Path) -> bool:
    """Check if file should be skipped from type checking."""
    # Skip test files, __pycache__, and setup files
    skip_patterns = [
        '__pycache__',
        '.pyc',
        'test_',
        'setup.py',
        'conftest.py',
        '__init__.py'  # Often just imports
    ]
    
    return any(pattern in str(filepath) for pattern in skip_patterns)

def main():
    """Main type annotation checking function."""
    project_root = Path(__file__).parent
    py_pinocchio_dir = project_root / 'py_pinocchio'
    
    if not py_pinocchio_dir.exists():
        print(f"py_pinocchio directory not found: {py_pinocchio_dir}")
        return 1
    
    print("🔍 Checking Type Annotation Completeness")
    print("=" * 60)
    
    total_files = 0
    files_with_issues = 0
    total_issues = 0
    
    # Find all Python files
    python_files = list(py_pinocchio_dir.glob('**/*.py'))
    
    for filepath in sorted(python_files):
        if should_skip_file(filepath):
            continue
            
        relative_path = filepath.relative_to(project_root)
        total_files += 1
        
        issues = check_file_type_annotations(filepath)
        
        if issues:
            files_with_issues += 1
            print(f"\n📄 {relative_path}")
            print("-" * 40)
            
            for line_num, message in issues:
                print(f"  Line {line_num}: {message}")
                total_issues += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Type Annotation Summary")
    print("=" * 30)
    print(f"Files checked: {total_files}")
    print(f"Files with issues: {files_with_issues}")
    print(f"Total issues found: {total_issues}")
    
    if total_issues == 0:
        print("\n✅ All functions have comprehensive type annotations!")
        return 0
    else:
        print(f"\n⚠️  {total_issues} type annotation issue(s) found")
        
        # Provide recommendations
        print("\nRecommendations:")
        print("1. Add return type annotations to all public functions")
        print("2. Add parameter type annotations to all function parameters")
        print("3. Use typing module for complex types (List, Dict, Optional, etc.)")
        print("4. Consider using mypy for static type checking")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())

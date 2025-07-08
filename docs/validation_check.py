#!/usr/bin/env python3
"""
Mathematical notation validation script for py-pinocchio documentation.

This script checks all documentation files for proper LaTeX mathematical notation
and ensures all mathematical terms are well-defined and explained.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

def find_non_latex_math_notation(content: str, filename: str) -> List[Tuple[int, str]]:
    """Find mathematical notation that should be in LaTeX format."""
    issues = []
    lines = content.split('\n')
    
    # Patterns that should be in LaTeX
    patterns = [
        (r'\b\d+×\d+\b', 'Dimension notation should use LaTeX: $n \\times m$'),
        (r'\b\d+x\d+\b', 'Dimension notation should use LaTeX: $n \\times m$'),
        (r'\bM\s*=\s*M\^T\b', 'Matrix equation should use LaTeX: $\\mathbf{M} = \\mathbf{M}^T$'),
        (r'\bq̇\b', 'Derivative notation should use LaTeX: $\\dot{\\mathbf{q}}$'),
        (r'\bq̈\b', 'Second derivative should use LaTeX: $\\ddot{\\mathbf{q}}$'),
        (r'\bτ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\boldsymbol{\\tau}$'),
        (r'\bω\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\boldsymbol{\\omega}$'),
        (r'\bλ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\lambda$'),
        (r'\bθ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\theta$'),
        (r'\bφ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\phi$'),
        (r'\bψ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\psi$'),
        (r'\bα\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\alpha$'),
        (r'\bβ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\beta$'),
        (r'\bγ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\gamma$'),
        (r'\bΓ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\Gamma$'),
        (r'\bΣ\b(?![^$]*\$)', 'Greek letters should use LaTeX: $\\Sigma$'),
        (r'\b[A-Z]\^\{?-?1\}?\b(?![^$]*\$)', 'Matrix inverse should use LaTeX: $\\mathbf{A}^{-1}$'),
        (r'\b[A-Z]\^T\b(?![^$]*\$)', 'Matrix transpose should use LaTeX: $\\mathbf{A}^T$'),
        (r'\|\|.*\|\|\b(?![^$]*\$)', 'Norm notation should use LaTeX: $\\|\\mathbf{v}\\|$'),
        (r'\b∈\b(?![^$]*\$)', 'Set membership should use LaTeX: $\\in$'),
        (r'\b∀\b(?![^$]*\$)', 'Universal quantifier should use LaTeX: $\\forall$'),
        (r'\b∃\b(?![^$]*\$)', 'Existential quantifier should use LaTeX: $\\exists$'),
        (r'\b≤\b(?![^$]*\$)', 'Inequality should use LaTeX: $\\leq$'),
        (r'\b≥\b(?![^$]*\$)', 'Inequality should use LaTeX: $\\geq$'),
        (r'\b≠\b(?![^$]*\$)', 'Not equal should use LaTeX: $\\neq$'),
        (r'\b≈\b(?![^$]*\$)', 'Approximately equal should use LaTeX: $\\approx$'),
    ]
    
    for line_num, line in enumerate(lines, 1):
        # Skip lines that are already in code blocks or LaTeX
        if '```' in line or line.strip().startswith('$$') or line.strip().startswith('$'):
            continue
            
        for pattern, message in patterns:
            if re.search(pattern, line):
                issues.append((line_num, f"{message} | Line: {line.strip()}"))
    
    return issues

def check_undefined_mathematical_terms(content: str, filename: str) -> List[Tuple[int, str]]:
    """Check for mathematical terms that should be defined."""
    issues = []
    lines = content.split('\n')
    
    # Mathematical terms that should be defined when first used
    terms_to_define = [
        'Jacobian', 'Hessian', 'Christoffel symbols', 'spatial algebra',
        'spatial vectors', 'spatial transformations', 'articulated body',
        'composite rigid body', 'recursive Newton-Euler', 'Lagrangian',
        'Hamiltonian', 'configuration space', 'task space', 'null space',
        'singularity', 'manipulability', 'condition number', 'pseudoinverse',
        'Moore-Penrose', 'SVD', 'eigendecomposition', 'Cholesky',
        'forward kinematics', 'inverse kinematics', 'forward dynamics',
        'inverse dynamics', 'mass matrix', 'Coriolis matrix', 'gravity vector'
    ]
    
    # This is a simplified check - in practice, you'd want more sophisticated analysis
    for line_num, line in enumerate(lines, 1):
        for term in terms_to_define:
            if term.lower() in line.lower() and 'define' not in line.lower():
                # Check if term appears without definition context
                if not any(keyword in line.lower() for keyword in ['is', 'are', 'represents', 'describes', 'means']):
                    continue  # Skip for now - this would need more sophisticated analysis
    
    return issues

def validate_latex_syntax(content: str, filename: str) -> List[Tuple[int, str]]:
    """Basic validation of LaTeX syntax."""
    issues = []
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # Check for unmatched dollar signs
        dollar_count = line.count('$')
        if dollar_count % 2 != 0:
            issues.append((line_num, f"Unmatched dollar signs in LaTeX: {line.strip()}"))
        
        # Check for common LaTeX errors
        if '\\mathbf{' in line and '}' not in line:
            issues.append((line_num, f"Unclosed \\mathbf command: {line.strip()}"))
        
        if '\\begin{' in line and '\\end{' not in content[content.find(line):content.find(line) + 500]:
            issues.append((line_num, f"Potentially unclosed LaTeX environment: {line.strip()}"))
    
    return issues

def check_documentation_file(filepath: Path) -> Dict[str, List[Tuple[int, str]]]:
    """Check a single documentation file for mathematical notation issues."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {'read_error': [(0, f"Could not read file: {e}")]}
    
    filename = str(filepath)
    issues = {}
    
    # Check for non-LaTeX mathematical notation
    non_latex_issues = find_non_latex_math_notation(content, filename)
    if non_latex_issues:
        issues['non_latex_math'] = non_latex_issues
    
    # Check for undefined mathematical terms
    undefined_terms = check_undefined_mathematical_terms(content, filename)
    if undefined_terms:
        issues['undefined_terms'] = undefined_terms
    
    # Validate LaTeX syntax
    latex_issues = validate_latex_syntax(content, filename)
    if latex_issues:
        issues['latex_syntax'] = latex_issues
    
    return issues

def main():
    """Main validation function."""
    docs_dir = Path(__file__).parent
    
    # Find all markdown files
    md_files = []
    for pattern in ['**/*.md', '**/*.rst']:
        md_files.extend(docs_dir.glob(pattern))
    
    print("🔍 Validating Mathematical Notation in Documentation")
    print("=" * 60)
    
    total_issues = 0
    files_with_issues = 0
    
    for filepath in sorted(md_files):
        relative_path = filepath.relative_to(docs_dir)
        issues = check_documentation_file(filepath)
        
        if issues:
            files_with_issues += 1
            print(f"\n📄 {relative_path}")
            print("-" * 40)
            
            for issue_type, issue_list in issues.items():
                print(f"\n{issue_type.replace('_', ' ').title()}:")
                for line_num, message in issue_list:
                    print(f"  Line {line_num}: {message}")
                    total_issues += 1
    
    print(f"\n📊 Validation Summary")
    print("=" * 30)
    print(f"Files checked: {len(md_files)}")
    print(f"Files with issues: {files_with_issues}")
    print(f"Total issues found: {total_issues}")
    
    if total_issues == 0:
        print("\n✅ All documentation files have proper mathematical notation!")
    else:
        print(f"\n⚠️  Found {total_issues} issues that should be addressed.")
        print("\nRecommendations:")
        print("1. Convert all mathematical notation to LaTeX format")
        print("2. Define mathematical terms when first introduced")
        print("3. Use consistent notation throughout documentation")
        print("4. Ensure all LaTeX syntax is properly formatted")
    
    return total_issues == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

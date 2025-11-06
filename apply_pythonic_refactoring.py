#!/usr/bin/env python3
"""
Script to apply Pythonic refactoring patterns across all modules.

This script applies the following improvements:
1. Module-level constants for validation
2. Helper functions for repeated patterns
3. Ternary operators for simple conditionals
4. List comprehensions where appropriate
5. f-strings for formatting
"""

import re
import os
from pathlib import Path

# Patterns to refactor
REFACTORING_PATTERNS = [
    # Pattern 1: Alternative validation with list -> set
    {
        'pattern': r'if alternative not in \["two-sided", "greater", "less"\]:',
        'replacement': 'if alternative not in VALID_ALTERNATIVES:',
        'add_constant': 'VALID_ALTERNATIVES = {"two-sided", "greater", "less"}',
    },
    # Pattern 2: Simple if/else -> ternary
    {
        'pattern': r'if alternative == "two-sided":\s+tails = 2\s+else:\s+tails = 1',
        'replacement': 'tails = 2 if alternative == "two-sided" else 1',
    },
]

def add_module_constants(content: str, constants: list) -> str:
    """Add module-level constants after imports."""
    import_end = 0
    lines = content.split('\n')
    
    # Find the end of imports
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end = i + 1
        elif import_end > 0 and line.strip() and not line.startswith('#'):
            break
    
    # Check if constants already exist
    constants_section = '\n# Module-level constants\n' + '\n'.join(constants) + '\n'
    
    if 'Module-level constants' not in content:
        lines.insert(import_end, constants_section)
        return '\n'.join(lines)
    
    return content

def refactor_file(filepath: Path) -> tuple[bool, str]:
    """Refactor a single Python file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Check if file needs VALID_ALTERNATIVES constant
        if 'alternative not in ["two-sided", "greater", "less"]' in content:
            if 'VALID_ALTERNATIVES' not in content:
                content = add_module_constants(content, ['VALID_ALTERNATIVES = {"two-sided", "greater", "less"}'])
                changes_made.append('Added VALID_ALTERNATIVES constant')
            
            # Replace validation pattern
            content = content.replace(
                'if alternative not in ["two-sided", "greater", "less"]:',
                'if alternative not in VALID_ALTERNATIVES:'
            )
            content = content.replace(
                '"alternative must be \'two-sided\', \'greater\', or \'less\'"',
                'f"alternative must be one of {VALID_ALTERNATIVES}"'
            )
            changes_made.append('Updated alternative validation')
        
        # Apply other simple replacements
        # Replace sum([...]) with sum(...)
        content = re.sub(r'sum\(\[(.*?)\]\)', r'sum(\1)', content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True, '; '.join(changes_made)
        
        return False, 'No changes needed'
    
    except Exception as e:
        return False, f'Error: {str(e)}'

def main():
    """Main refactoring function."""
    real_simple_stats_dir = Path(__file__).parent / 'real_simple_stats'
    
    # Files to refactor
    files_to_check = [
        'bayesian_stats.py',
        'resampling.py',
        'multivariate.py',
        'effect_sizes.py',
        'time_series.py',
        'hypothesis_testing.py',
        'linear_regression_utils.py',
        'chi_square_utils.py',
        'probability_distributions.py',
        'sampling_and_intervals.py',
    ]
    
    print("=" * 60)
    print("Pythonic Refactoring Script")
    print("=" * 60)
    
    results = []
    for filename in files_to_check:
        filepath = real_simple_stats_dir / filename
        if filepath.exists():
            changed, message = refactor_file(filepath)
            status = "✅ CHANGED" if changed else "⏭️  SKIPPED"
            results.append((filename, status, message))
            print(f"{status}: {filename}")
            if message:
                print(f"  └─ {message}")
        else:
            print(f"⚠️  NOT FOUND: {filename}")
    
    print("\n" + "=" * 60)
    print(f"Summary: {sum(1 for _, s, _ in results if 'CHANGED' in s)} files changed")
    print("=" * 60)

if __name__ == '__main__':
    main()

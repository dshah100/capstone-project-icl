"""
Validation Script for Round 6 Submission
=========================================
Checks your queries and reflection before submission to catch common errors.
"""

import json
import re
import os
from pathlib import Path


class SubmissionValidator:
    """Validates Round 6 submission for common errors."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
    def validate_query_format(self, query_string, expected_dims):
        """
        Validate a single query string.
        
        Returns: (valid, message)
        """
        # Check format: x1-x2-x3-...-xn
        parts = query_string.split('-')
        
        # Check dimension count
        if len(parts) != expected_dims:
            return False, f"Expected {expected_dims} dimensions, got {len(parts)}"
        
        # Check each value
        for i, part in enumerate(parts):
            # Check if starts with 0
            if not part.startswith('0.'):
                return False, f"Dimension {i+1} must start with '0.' (got '{part}')"
            
            # Check decimal places
            if '.' in part:
                decimal_part = part.split('.')[1]
                if len(decimal_part) != 6:
                    return False, f"Dimension {i+1} must have exactly 6 decimal places (got {len(decimal_part)})"
            
            # Check if valid float
            try:
                val = float(part)
                if val < 0 or val >= 1:
                    return False, f"Dimension {i+1} must be in [0, 1) (got {val})"
            except ValueError:
                return False, f"Dimension {i+1} is not a valid number: '{part}'"
        
        return True, "OK"
    
    def validate_queries_file(self, filepath='round_6_portal_submission.txt'):
        """Validate the portal submission file."""
        print("\n" + "="*70)
        print("VALIDATING QUERIES")
        print("="*70)
        
        if not os.path.exists(filepath):
            self.errors.append(f"Query file not found: {filepath}")
            return False
        
        expected_dims = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract queries using regex
        found_functions = set()
        
        for func_id in range(1, 9):
            # Look for "Function X:" followed by the query
            pattern = rf"Function {func_id}:\s*\n\s*([0-9.-]+)"
            match = re.search(pattern, content)
            
            if not match:
                self.errors.append(f"Function {func_id}: Query not found in file")
                continue
            
            query = match.group(1).strip()
            found_functions.add(func_id)
            
            # Validate format
            valid, message = self.validate_query_format(query, expected_dims[func_id])
            
            if valid:
                self.info.append(f"✓ Function {func_id}: {message}")
                print(f"  ✓ Function {func_id} ({expected_dims[func_id]}D): Valid")
            else:
                self.errors.append(f"Function {func_id}: {message}")
                print(f"  ✗ Function {func_id}: {message}")
        
        # Check if all functions present
        missing = set(range(1, 9)) - found_functions
        if missing:
            self.errors.append(f"Missing queries for functions: {sorted(missing)}")
        
        return len(self.errors) == 0
    
    def validate_query_history(self, filepath='query_history.json'):
        """Validate the query history JSON file."""
        print("\n" + "="*70)
        print("VALIDATING QUERY HISTORY")
        print("="*70)
        
        if not os.path.exists(filepath):
            self.warnings.append(f"Query history not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in query history: {e}")
            return False
        
        expected_dims = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
        
        for func_id in range(1, 9):
            key = f"function_{func_id}"
            
            if key not in history:
                self.warnings.append(f"Function {func_id}: No history found")
                continue
            
            observations = history[key]
            n_obs = len(observations)
            
            if n_obs < 3:
                self.warnings.append(
                    f"Function {func_id}: Only {n_obs} observations (need at least 3 for GP)"
                )
            elif n_obs < 5:
                self.warnings.append(
                    f"Function {func_id}: Only {n_obs} observations (Round 6 should have 5+)"
                )
            else:
                self.info.append(f"✓ Function {func_id}: {n_obs} observations")
            
            # Validate each observation
            for i, obs in enumerate(observations):
                # Check required fields
                if 'query' not in obs or 'result' not in obs:
                    self.errors.append(
                        f"Function {func_id}, observation {i+1}: Missing 'query' or 'result'"
                    )
                    continue
                
                # Validate query dimensions
                query = obs['query']
                if len(query) != expected_dims[func_id]:
                    self.errors.append(
                        f"Function {func_id}, observation {i+1}: "
                        f"Expected {expected_dims[func_id]} dimensions, got {len(query)}"
                    )
                
                # Validate result
                result = obs['result']
                if not isinstance(result, (int, float)) or result < 0:
                    self.warnings.append(
                        f"Function {func_id}, observation {i+1}: "
                        f"Unusual result value: {result}"
                    )
            
            print(f"  Function {func_id}: {n_obs} observations")
        
        return True
    
    def validate_reflection(self, filepath='round_6_reflection.md'):
        """Validate the reflection document."""
        print("\n" + "="*70)
        print("VALIDATING REFLECTION")
        print("="*70)
        
        if not os.path.exists(filepath):
            self.warnings.append(f"Reflection not found: {filepath}")
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count words (approximate - between ### headers)
        # Remove markdown formatting
        text = re.sub(r'[#*`\-]', '', content)
        text = re.sub(r'\n+', ' ', text)
        words = text.split()
        word_count = len(words)
        
        print(f"\n  Word count: {word_count}")
        
        if word_count > 700:
            self.errors.append(
                f"Reflection exceeds 700 words ({word_count} words). "
                "Please trim to meet requirements."
            )
        elif word_count < 400:
            self.warnings.append(
                f"Reflection is quite short ({word_count} words). "
                "Consider adding more specific examples."
            )
        else:
            self.info.append(f"✓ Word count appropriate: {word_count} words")
        
        # Check for required sections
        required_prompts = [
            "Progressive Feature Extraction",
            "Breakthrough Parallels",
            "Balancing Depth",
            "Learning from Data",
            "Real-World Deployment"
        ]
        
        missing_sections = []
        for prompt in required_prompts:
            if prompt.lower() not in content.lower():
                missing_sections.append(prompt)
        
        if missing_sections:
            self.warnings.append(
                f"Possibly missing sections: {', '.join(missing_sections)}"
            )
        else:
            self.info.append("✓ All required sections appear to be present")
        
        # Check for template placeholders
        placeholders = [
            "[Your Name]",
            "[replace with",
            "[Add specific",
            "[Describe a case",
            "[Your count",
            "Function X"  # Should be replaced with actual function IDs
        ]
        
        found_placeholders = []
        for placeholder in placeholders:
            if placeholder in content:
                found_placeholders.append(placeholder)
        
        if found_placeholders:
            self.warnings.append(
                f"Template placeholders found (replace before submission): "
                f"{', '.join(found_placeholders[:3])}..."
            )
        else:
            self.info.append("✓ No obvious template placeholders found")
        
        # Check for specific examples
        if "function 1" not in content.lower() and "function 2" not in content.lower():
            self.warnings.append(
                "Consider adding specific function references (e.g., 'Function 1', 'Function 2')"
            )
        
        # Check for numeric data
        if not re.search(r'\d+\.\d{3,}', content):
            self.warnings.append(
                "Consider adding specific numeric results (e.g., distances, improvements)"
            )
        
        return True
    
    def check_file_structure(self):
        """Check that all required files exist."""
        print("\n" + "="*70)
        print("CHECKING FILE STRUCTURE")
        print("="*70)
        
        required_files = {
            '1st.log': 'Initial queries',
            'bbo_capstone_framework.py': 'Main BBO framework',
            'quick_round6_generator.py': 'Query generator'
        }
        
        optional_files = {
            'query_history.json': 'Query history data',
            'round_6_portal_submission.txt': 'Portal submission',
            'round_6_reflection.md': 'Reflection document'
        }
        
        print("\n  Required files:")
        for filename, description in required_files.items():
            if os.path.exists(filename):
                print(f"    ✓ {filename} - {description}")
            else:
                print(f"    ✗ {filename} - {description} (MISSING)")
                self.errors.append(f"Required file missing: {filename}")
        
        print("\n  Submission files:")
        for filename, description in optional_files.items():
            if os.path.exists(filename):
                print(f"    ✓ {filename} - {description}")
            else:
                print(f"    ⚠ {filename} - {description} (not found)")
                self.warnings.append(f"File not found: {filename}")
    
    def generate_report(self):
        """Generate a comprehensive validation report."""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        # Summary
        if not self.errors and not self.warnings:
            print("\n  🎉 EXCELLENT! Your submission looks good!")
            print("\n  You're ready to submit:")
            print("    1. Queries to the capstone portal")
            print("    2. Reflection to the discussion board")
        elif not self.errors:
            print("\n  ✓ GOOD! No critical errors found.")
            print(f"  ⚠ {len(self.warnings)} warning(s) to review:")
        else:
            print("\n  ✗ ISSUES FOUND! Please fix before submitting.")
            print(f"  ✗ {len(self.errors)} error(s)")
            print(f"  ⚠ {len(self.warnings)} warning(s)")
        
        # Errors
        if self.errors:
            print("\n  🔴 ERRORS (must fix):")
            for i, error in enumerate(self.errors, 1):
                print(f"    {i}. {error}")
        
        # Warnings
        if self.warnings:
            print("\n  ⚠ WARNINGS (should review):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"    {i}. {warning}")
        
        # Info
        if self.info and not self.errors:
            print("\n  ℹ INFO:")
            for info in self.info[:5]:  # Show first 5
                print(f"    • {info}")
        
        # Next steps
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        
        if not self.errors:
            print("\n  1. Review warnings above (if any)")
            print("  2. Submit queries to capstone portal")
            print("  3. Post reflection to discussion board")
            print("  4. Engage with peers' posts")
            print("  5. Record portal results for Round 7")
        else:
            print("\n  1. Fix the errors listed above")
            print("  2. Re-run this validation script")
            print("  3. Once validation passes, submit to portal")
        
        print("\n")
        
        return len(self.errors) == 0


def main():
    """Main validation function."""
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*18 + "ROUND 6 SUBMISSION VALIDATOR" + " "*22 + "║")
    print("╚"+"═"*68+"╝")
    
    print("\nThis script will check:")
    print("  ✓ Query format and dimensions")
    print("  ✓ Query history completeness")
    print("  ✓ Reflection word count and content")
    print("  ✓ File structure")
    
    validator = SubmissionValidator()
    
    # Run validations
    validator.check_file_structure()
    validator.validate_query_history()
    validator.validate_queries_file()
    validator.validate_reflection()
    
    # Generate report
    success = validator.generate_report()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

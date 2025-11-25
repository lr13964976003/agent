#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path

# Define paths
output_dir = Path("../outputs/2025-11-24-17-20-17")
dot_files = [
    "baseline_dag_final.dot",
    "helix_dag_final.dot"
]

def generate_svg(dot_file):
    """Generate SVG image from DOT file."""
    dot_path = output_dir / dot_file
    svg_path = output_dir / dot_file.replace('.dot', '.svg')
    
    if not dot_path.exists():
        print(f"Error: {dot_path} does not exist")
        return False
    
    try:
        # Use dot to generate SVG
        result = subprocess.run([
            'dot', '-Tsvg', str(dot_path), '-o', str(svg_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Generated {svg_path}")
            return True
        else:
            print(f"✗ Error generating {svg_path}: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception generating {svg_path}: {e}")
        return False

def validate_dag_structure(dot_file):
    """Validate DAG structure."""
    dot_path = output_dir / dot_file
    
    try:
        # Check for cycles using dot
        result = subprocess.run([
            'dot', '-Tdot', str(dot_path), '-o', '/dev/null'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {dot_file} is a valid DAG (no cycles)")
            return True
        else:
            if "cycle" in result.stderr.lower():
                print(f"✗ {dot_file} contains cycles: {result.stderr}")
            else:
                print(f"✗ {dot_file} validation error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception validating {dot_file}: {e}")
        return False

def main():
    print("=== DAG Generation and Validation ===")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for dot_file in dot_files:
        print(f"\n--- Processing {dot_file} ---")
        
        # Validate DAG
        valid = validate_dag_structure(dot_file)
        
        # Generate SVG
        svg_generated = generate_svg(dot_file)
        
        results.append({
            'file': dot_file,
            'valid': valid,
            'svg_generated': svg_generated,
            'path': str(output_dir / dot_file)
        })
    
    # Generate submission JSON
    submission = {
        "baseline_dag": str(output_dir / "baseline_dag_final.dot"),
        "helix_dag": str(output_dir / "helix_dag_final.dot"),
        "baseline_svg": str(output_dir / "baseline_dag_final.svg"),
        "helix_svg": str(output_dir / "helix_dag_final.svg"),
        "validation_results": results
    }
    
    import json
    with open(output_dir / "final_submission_paths.json", "w") as f:
        json.dump(submission, f, indent=2)
    
    print(f"\n✓ Final submission paths written to {output_dir / 'final_submission_paths.json'}")
    
    # Print summary
    print("\n=== Final Summary ===")
    for result in results:
        status = "✓" if result['valid'] and result['svg_generated'] else "✗"
        print(f"{status} {result['file']}: Valid={result['valid']} SVG={result['svg_generated']}")

if __name__ == "__main__":
    main()
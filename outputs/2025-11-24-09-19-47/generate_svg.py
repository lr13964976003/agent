import subprocess
import os

# Change to the correct directory
os.chdir('../outputs/2025-11-24-09-19-47')

# Generate SVG for baseline
subprocess.run(['dot', '-Tsvg', 'baseline_concise_corrected.dot', '-o', 'baseline_concise_corrected.svg'], 
               check=True, stderr=subprocess.DEVNULL)

# Generate SVG for proposed
subprocess.run(['dot', '-Tsvg', 'proposed_concise_corrected.dot', '-o', 'proposed_concise_corrected.svg'], 
               check=True, stderr=subprocess.DEVNULL)

print("SVG files generated successfully")
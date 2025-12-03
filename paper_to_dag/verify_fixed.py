#!/usr/bin/env python3
"""
Fixed Deployment Verification Script
"""

import json
import sys
import os

# Add the output directory to the path
sys.path.append('/home/wzc/app/agent/paper_to_dag/outputs/2025-12-03-11-11-34')

# Change to the correct directory
os.chdir('/home/wzc/app/agent/paper_to_dag/outputs/2025-12-03-11-11-34')

# Copy the DeploymentVerifier class from the original file
exec(open('/home/wzc/app/agent/paper_to_dag/outputs/2025-12-03-11-11-34/verify_deployment.py').read())

# Run the verification
if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
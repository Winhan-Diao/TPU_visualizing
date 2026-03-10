#!/usr/bin/env python3
"""
Wrapper module to interface with the TPU analyzer from Node.js
"""

import sys
import os
import json
from my_tpu_ana import run_simulation

def main():
    """
    Main function to be called from Node.js using subprocess
    Expects command line arguments and returns JSON result
    """
    # Parse command line arguments
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            key = sys.argv[i][2:]  # Remove '--'
            if i + 1 < len(sys.argv) and not sys.argv[i+1].startswith('--'):
                args[key] = sys.argv[i+1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1

    # Set defaults
    n_layers = int(args.get('n_layers', 24))
    n_nodes = int(args.get('n_nodes', 3))
    batch_size = int(args.get('batch_size', 32))
    n_length = int(args.get('n_length', 3))

    # Run simulation
    result = run_simulation(n_layers, n_nodes, batch_size, n_length)
    
    # Output as JSON
    print(json.dumps(result))

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
extract_user_ids.py

Loads a JSON file (`romberg.json`) containing a list of ROMBERG files,
extracts the 5-character user ID from the start of each filename
(e.g. "C0001Pre00romberg.mat" → "C0001"), converts it to uppercase,
and outputs the unique IDs.
"""

import json
import sys
from pathlib import Path

def extract_user_ids(input_path):
    # Load the JSON
    with open(input_path, 'r') as f:
        data = json.load(f)

    user_ids = set()
    for entry in data.get('romberg_files', []):
        filename = entry.get('filename', '')
        if len(filename) >= 5:
            user_id = filename[:5].upper()
            user_ids.add(user_id)
        else:
            # Warn if the filename is unexpectedly short
            print(f"Warning: filename too short to extract user_id: '{filename}'", file=sys.stderr)

    return sorted(user_ids)

def main():
    # Allow the input filename to be specified on the command line
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'romberg_cleaned.json'
    if not Path(input_file).exists():
        print(f"Error: '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    ids = extract_user_ids(input_file)

    # Print the list of unique user IDs
    print("Extracted user IDs:")
    for uid in ids:
        print(uid)

    # Optionally, write to a JSON file
    out_path = Path(input_file).with_name('user_ids.json')
    with open(out_path, 'w') as f:
        json.dump(ids, f, indent=2)
    print(f"\nUnique user IDs written to {out_path}")

if __name__ == '__main__':
    main()


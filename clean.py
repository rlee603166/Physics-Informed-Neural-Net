import json

def normalize_and_filter(input_path, output_path):
    # Load original JSON
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Normalize summary to lowercase strings (if needed)
    summary = {
        k: (v.lower() if isinstance(v, str) else v)
        for k, v in data.get('summary', {}).items()
    }
    
    # Process and filter romberg_files
    filtered = []
    for entry in data.get('romberg_files', []):
        # Lowercase all string fields
        normalized = {
            k: (v.lower() if isinstance(v, str) else v)
            for k, v in entry.items()
        }
        # Filter for filename containing "romberg" and timing_category == "pre00"
        if 'romberg' in normalized.get('filename', '') and normalized.get('timing_category') == 'pre00':
            filtered.append(normalized)
    
    # Update summary counts
    summary['total_romberg_pre00_files'] = len(filtered)
    summary['pre00_files'] = len(filtered)
    summary['total_files_in_folder'] = len(filtered)  # or leave original if you prefer
    summary['filter_applied'] = 'only romberg + pre00 files included'
    
    # Build cleaned dict
    cleaned = {
        'summary': summary,
        'romberg_files': filtered
    }
    
    # Write out cleaned JSON
    with open(output_path, 'w') as f:
        json.dump(cleaned, f, indent=2)
    print(f"Cleaned JSON written to {output_path}")

if __name__ == '__main__':
    normalize_and_filter('romberg.json', 'romberg_cleaned.json')


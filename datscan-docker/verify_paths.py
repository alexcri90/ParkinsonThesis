# verify_paths.py
import os
import pandas as pd
import argparse

def verify_paths(csv_path, verbose=True):
    """Verify that file paths in the CSV exist."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False
    
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'file_path' not in df.columns:
        print(f"Error: CSV does not contain a 'file_path' column")
        return False
    
    print(f"CSV contains {len(df)} file paths")
    
    # Check paths
    existing_paths = 0
    missing_paths = 0
    
    # Print current working directory for reference
    print(f"Current working directory: {os.getcwd()}")
    
    # Check first few paths
    for i, row in df.head(10).iterrows():
        file_path = row['file_path']
        exists = os.path.exists(file_path)
        
        if exists:
            existing_paths += 1
        else:
            missing_paths += 1
        
        if verbose:
            print(f"Path {i+1}: {file_path}")
            print(f"  Exists: {exists}")
    
    # Check total counts
    total_existing = df['file_path'].apply(os.path.exists).sum()
    total_missing = len(df) - total_existing
    
    print(f"\nPath verification summary:")
    print(f"  Total paths in CSV: {len(df)}")
    print(f"  Existing paths: {total_existing} ({total_existing/len(df)*100:.1f}%)")
    print(f"  Missing paths: {total_missing} ({total_missing/len(df)*100:.1f}%)")
    
    # If all paths are missing, suggest common issues
    if total_existing == 0:
        print("\nAll paths are missing. Common issues:")
        print("1. Paths in CSV are absolute but need to be relative")
        print("2. Paths in CSV use backslashes (\\) but need forward slashes (/)")
        print("3. The volume mapping in Docker is incorrect")
        print("4. The working directory inside Docker is different than expected")
        
        # Print example of the first path for inspection
        if len(df) > 0:
            example_path = df.iloc[0]['file_path']
            print(f"\nExample path from CSV: {example_path}")
            
            # Try different path formats
            print("Attempting different path formats...")
            
            # Try with forward slashes
            alt_path = example_path.replace('\\', '/')
            print(f"  With forward slashes: {alt_path}")
            print(f"  Exists: {os.path.exists(alt_path)}")
            
            # Try relative to parent directory
            if os.path.isabs(example_path):
                base_name = os.path.basename(example_path)
                print(f"  Just filename: {base_name}")
                print(f"  Exists: {os.path.exists(base_name)}")
    
    return total_existing > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify file paths in CSV")
    parser.add_argument('--csv', type=str, default='data/validated_file_paths.csv',
                        help='Path to CSV file with file paths')
    args = parser.parse_args()
    
    verify_paths(args.csv)
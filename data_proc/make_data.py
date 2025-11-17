import os
import argparse
from glob import glob

def main():
    parser = argparse.ArgumentParser(description="Generate data.txt listing all opencv_cameras.json in submeta_files.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing view meta_files')
    args = parser.parse_args()

    data_dir = args.data_dir
    meta_files = glob(os.path.join(data_dir, "*.json"))

    output_path = os.path.join(data_dir, "data.txt")
    with open(output_path, "w") as f:
        for file in sorted(meta_files):
            f.write(file + "\n")

    print(f"Written {len(meta_files)} entries to {output_path}")

if __name__ == "__main__":
    main()

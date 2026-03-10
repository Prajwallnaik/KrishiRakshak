import os
from PIL import Image

class LightCleaner:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.allowed_extensions = {".jpg", ".jpeg", ".png"}
        # Summary statistics
        self.scanned_files = 0
        self.empty_files_dropped = 0
        self.wrong_formats_removed = 0
        self.corrupt_files_removed = 0
        self.extensions_standardized = 0

    def clean(self):
        print(f"Starting Light Cleaning on directory: {self.data_dir}")
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                self.scanned_files += 1
                file_path = os.path.join(root, file)

                # 1. Drop Empty/Invalid Files
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    print(f"Removed empty file: {file_path}")
                    self.empty_files_dropped += 1
                    continue

                # 2. Remove Wrong Formats
                ext = os.path.splitext(file)[1].lower()
                if ext not in self.allowed_extensions:
                    os.remove(file_path)
                    print(f"Removed invalid format file: {file_path}")
                    self.wrong_formats_removed += 1
                    continue

                # 3. Verify Image Integrity & 4. Standardize Extensions
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Check for corruption without fully loading
                    
                    # If verified, standardize extensions to lowercase .jpg or .png
                    if os.path.splitext(file)[1] != ext: # if extension is like .JPG
                        new_path = os.path.join(root, os.path.splitext(file)[0] + ext)
                        os.rename(file_path, new_path)
                        self.extensions_standardized += 1
                        
                except Exception as e:
                    # If PIL throws an error, the file is corrupt
                    try:
                        os.remove(file_path)
                        print(f"Removed corrupt image: {file_path} (Error: {e})")
                        self.corrupt_files_removed += 1
                    except PermissionError:
                        print(f"Could not remove {file_path} due to permissions.")
        
        self.print_summary()

    def print_summary(self):
        print("\n=== Light Cleaning Summary ===")
        print(f"Total Files Scanned: {self.scanned_files}")
        print(f"Empty Files Dropped: {self.empty_files_dropped}")
        print(f"Wrong Formats Removed: {self.wrong_formats_removed}")
        print(f"Corrupt Files Deleted: {self.corrupt_files_removed}")
        print(f"Files Standardized (e.g. .JPG to .jpg): {self.extensions_standardized}")
        print("===")

if __name__ == "__main__":
    cleaner = LightCleaner(data_dir="../data")
    cleaner.clean()

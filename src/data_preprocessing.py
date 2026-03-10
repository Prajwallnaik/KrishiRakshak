import os
import numpy as np
from PIL import Image

class DataPreprocessor:
    def __init__(self, raw_dir="../data/raw", processed_dir="../data/processed", target_size=(224, 224)):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.target_size = target_size
        
    def execute_preprocessing(self):
        """
        Pulls raw images, resizes them, and saves to the processed folder.
        """
        print("--- Starting Targeted Preprocessing (Raw -> Processed) ---")
        
        # In a real scenario, this paths from the src folder to the main project data folder
        base_raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
        base_proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
        
        if not os.path.exists(base_raw_dir):
            print(f"Error: Raw directory '{base_raw_dir}' does not exist.")
            return

        classes = [d for d in os.listdir(base_raw_dir) if os.path.isdir(os.path.join(base_raw_dir, d))]
        
        for cls in classes:
            raw_cls_dir = os.path.join(base_raw_dir, cls)
            processed_cls_dir = os.path.join(base_proc_dir, cls)
            os.makedirs(processed_cls_dir, exist_ok=True)
            
            images = os.listdir(raw_cls_dir)
            print(f"Processing {len(images)} images for class '{cls}'...")
            
            for img_name in images:
                src_path = os.path.join(raw_cls_dir, img_name)
                dest_path = os.path.join(processed_cls_dir, img_name)
                
                try:
                    with Image.open(src_path) as img:
                        # Resize and save to processed directory
                        img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
                        img_resized.save(dest_path)
                except Exception as e:
                    print(f"Skipping corrupt file {img_name}: {e}")

        print("\nFinished saving all resized images to the 'processed' folder.")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.execute_preprocessing()

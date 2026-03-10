import os
import shutil
import numpy as np

class DataSplitter:
    def __init__(self, processed_dir="../data/processed", train_ratio=0.8, val_ratio=0.1):
        self.processed_dir = processed_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # In a real scenario, this resolves paths from the src folder to the main project data folder
        self.base_proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
        self.base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        
        self.train_dir = os.path.join(self.base_data_dir, 'train')
        self.val_dir = os.path.join(self.base_data_dir, 'val')
        self.test_dir = os.path.join(self.base_data_dir, 'test')
        
    def execute_split(self):
        """
        Takes the preprocessed images and explicitly splits them into train, val, test folders.
        """
        print("--- Starting Dataset Allocation Split ---")
        
        if not os.path.exists(self.base_proc_dir):
            print(f"Error: Processed directory '{self.base_proc_dir}' does not exist.")
            return

        # Ensure target directories exist
        for d in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(d, exist_ok=True)
            
        classes = [d for d in os.listdir(self.base_proc_dir) if os.path.isdir(os.path.join(self.base_proc_dir, d))]
        
        for cls in classes:
            processed_cls_dir = os.path.join(self.base_proc_dir, cls)
            
            # Make class subdirectories in target folders
            train_cls_dir = os.path.join(self.train_dir, cls)
            val_cls_dir = os.path.join(self.val_dir, cls)
            test_cls_dir = os.path.join(self.test_dir, cls)
            
            os.makedirs(train_cls_dir, exist_ok=True)
            os.makedirs(val_cls_dir, exist_ok=True)
            os.makedirs(test_cls_dir, exist_ok=True)
            
            images = os.listdir(processed_cls_dir)
            
            # Shuffle deterministically
            np.random.seed(42)
            np.random.shuffle(images)
            
            # Calculate splits
            num_imgs = len(images)
            train_idx = int(num_imgs * self.train_ratio)
            val_idx = train_idx + int(num_imgs * self.val_ratio)
            
            train_imgs = images[:train_idx]
            val_imgs = images[train_idx:val_idx]
            test_imgs = images[val_idx:]
            
            print(f"Splitting '{cls}': {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test")
            
            # Copy files over
            for img in train_imgs:
                shutil.copy2(os.path.join(processed_cls_dir, img), os.path.join(train_cls_dir, img))
            for img in val_imgs:
                shutil.copy2(os.path.join(processed_cls_dir, img), os.path.join(val_cls_dir, img))
            for img in test_imgs:
                shutil.copy2(os.path.join(processed_cls_dir, img), os.path.join(test_cls_dir, img))

        print("\nDataset successfully distributed to train, val, and test folders!")

if __name__ == "__main__":
    splitter = DataSplitter(train_ratio=0.8, val_ratio=0.1)
    splitter.execute_split()

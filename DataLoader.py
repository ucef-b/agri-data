import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import random

class DatasetLoader:
    """
    DataLoader for Agriculture-Vision preprocessed dataset
    """
    def __init__(
        self,
        working_path: str,
        batch_size: int = 8,
        export_type: str = "RGBN",
        outputs_type: str = "both",
        augmentation: bool = False,
        shuffle: bool = False
    ):
        """
        Args:
            working_path (str): Path to preprocessed dataset
            batch_size (int): Number of samples per batch
            export_type (str): Type of input image ('RGBN', 'NDVI', or 'RGB')
            outputs_type (str): Type of output ('mask_only', 'class_only', or 'both')
            augmentation (bool): Whether to apply data augmentation
            shuffle (bool): Whether to shuffle the dataset
        """
        self.working_path = Path(working_path)
        self.batch_size = batch_size
        self.export_type = export_type
        self.outputs_type = outputs_type
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        # Load file paths
        self.input_files = sorted(list((self.working_path / "inputs").glob("*.npy")))
        self.label_files = sorted(list((self.working_path / "labels").glob("*.npy")))
        
        # Validation checks
        assert len(self.input_files) == len(self.label_files), "Mismatch in input and label files"
        assert export_type in ["RGBN", "NDVI", "RGB"], "Invalid export_type"
        assert outputs_type in ["mask_only", "class_only", "both"], "Invalid outputs_type"
        
        self.num_samples = len(self.input_files)
        self.current_index = 0
        
    def __len__(self) -> int:
        return self.num_samples
    
    def calculate_ndvi(self, rgbn_image: np.ndarray) -> np.ndarray:
        nir = rgbn_image[:, :, 3].astype(np.float32) / 255.0
        red = rgbn_image[:, :, 0].astype(np.float32) / 255.0
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-8)  # ranges from -1 to +1
        
        # Scale NDVI to [0, 255]
        ndvi_scaled = ((ndvi + 1) * 127.5).astype(np.uint8)
        
        return np.expand_dims(ndvi_scaled, axis=-1)
    
    def process_input(self, rgbn_image: np.ndarray) -> np.ndarray:
        """Process input image based on export_type"""
        if self.export_type == "RGBN":
            return rgbn_image
        elif self.export_type == "NDVI":
            return self.calculate_ndvi(rgbn_image)
        else:  # RGB
            return rgbn_image[:, :, :3]
    
    def augment_data(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        if not self.augmentation:
            return image, mask
            
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            
        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
            
        return image, mask
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[np.ndarray, Dict]:
        """Get next batch of data"""
        if self.current_index >= self.num_samples:
            self.current_index = 0
            if self.shuffle:
                # Shuffle both lists together
                combined = list(zip(self.input_files, self.label_files))
                random.shuffle(combined)
                self.input_files, self.label_files = zip(*combined)
            raise StopIteration
            
        batch_images = []
        batch_masks = []
        batch_labels = []
        
        for i in range(self.batch_size):
            if self.current_index >= self.num_samples:
                break
                
            # Load data
            rgbn_image = np.load(self.input_files[self.current_index])
            mask = np.load(self.label_files[self.current_index])
            
            # Process input
            processed_image = self.process_input(rgbn_image)
            
            # Apply augmentation
            processed_image, mask = self.augment_data(processed_image, mask)
            
            # Append to batch
            batch_images.append(processed_image)
            batch_masks.append(mask)
            batch_labels.append((mask.sum(axis=(0,1)) > 0).astype(int))
            
            self.current_index += 1
            
        # Stack batches
        batch_x = np.stack(batch_images)
        batch_masks = np.stack(batch_masks)
        batch_labels = np.stack(batch_labels)
        
        # Prepare output based on outputs_type
        if self.outputs_type == "mask_only":
            batch_y = batch_masks
        elif self.outputs_type == "class_only":
            batch_y = batch_labels
        else:  # both
            batch_y = {
                'masks': batch_masks,
                'labels': batch_labels
            }
            
        return batch_x, batch_y

    def reset(self):
        """Reset the iterator"""
        self.current_index = 0
        if self.shuffle:
            combined = list(zip(self.input_files, self.label_files))
            random.shuffle(combined)
            self.input_files, self.label_files = zip(*combined)
import os
import numpy as np
import cv2
from glob import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil
import pandas as pd
from collections import defaultdict

def clean_and_process_data(source_path, output_path, selected_classes, no_label_classes, 
                          img_size=(512, 512), include_normal=False, max_samples=100):
    """
    Enhanced data preparation with proper normal case handling and conflict resolution
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    
    # Create output directories
    inputs_dir = output_path / "inputs"
    labels_dir = output_path / "labels"
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Precompute image-class mapping
    available_classes = [d.name for d in (source_path / "labels").iterdir() if d.is_dir()]
    image_class_map = defaultdict(set)
    
    print("Precomputing class mappings...")
    for cls in tqdm(available_classes):
        cls_dir = source_path / "labels" / cls
        for mask_path in cls_dir.glob("*.png"):
            image_id = mask_path.stem
            image_class_map[image_id].add(cls)

    # Get all RGB and NIR images
    rgb_paths = {Path(p).stem: p for p in glob(str(source_path / "images" / "rgb" / "*.jpg"))}
    nir_paths = {Path(p).stem: p for p in glob(str(source_path / "images" / "nir" / "*.jpg"))}

    # Validate existence of all images
    all_image_ids = set(rgb_paths.keys()) & set(nir_paths.keys())
    image_class_map = {k: v for k, v in image_class_map.items() if k in all_image_ids}

    # Track statistics and metadata
    stats = {cls: 0 for cls in selected_classes}
    csv_data = []
    class_mapping = {cls: idx for idx, cls in enumerate(selected_classes)}
    processed_count = 0
    num_classes = len(selected_classes)
    print("\nProcessing images...")
    for image_id in tqdm(image_class_map):
        classes_present = image_class_map[image_id]
        if max_samples and processed_count >= max_samples:
            break
        # Initialize flags
        has_selected = False
        has_normal = False
        not_selected = False
        class_mask = np.zeros((*img_size, num_classes), dtype=np.uint8)
        
        # First pass: Check for selected classes
        for cls in list(classes_present & set(no_label_classes)):
            mask_path = source_path / "labels" / cls / f"{image_id}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
            
            if np.any(mask > 0):
                not_selected = True

        if not_selected:
            continue
        
        selected_present = set()
        for cls in list(classes_present & set(selected_classes)):
            mask_path = source_path / "labels" / cls / f"{image_id}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                continue
            
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
            
            if np.any(mask > 0):
                class_idx = class_mapping[cls]
                class_mask[:, :, class_idx] = (mask > 0).astype(np.uint8)
                stats[cls] += 1
                selected_present.add(cls)
                has_selected = True

        # Only save if we have valid data
        if has_selected or has_normal:
            # Load and process image data
            rgb = cv2.imread(rgb_paths[image_id])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, img_size, interpolation=cv2.INTER_AREA)
            
            nir = cv2.imread(nir_paths[image_id], cv2.IMREAD_GRAYSCALE)
            nir = cv2.resize(nir, img_size, interpolation=cv2.INTER_AREA)
            
            # Create 4-channel RGBN image
            rgbn = np.dstack((rgb, nir))
            
            # Save processed data
            np.save(inputs_dir / f"{image_id}.npy", rgbn)
            np.save(labels_dir / f"{image_id}.npy", class_mask)
            
            # Record metadata
            record = {
                "image_id": image_id,
                **{cls: int(cls in selected_present) for cls in selected_classes},
            }
            csv_data.append(record)

            processed_count+=1

    # Save metadata and statistics
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path / "metadata.csv", index=False)
    
    print(f"\nFinal statistics:")
    print(f"Total processed images: {len(df)}")
    for cls, count in stats.items():
        if count > 0:
            print(f"{cls}: {count} ({count/len(df)*100:.1f}%)")
    
    return len(df), stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agriculture Vision Data Preparer")
    parser.add_argument("--source", required=True, help="Source dataset directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--selected", nargs="+", 
                       default=["nutrient_deficiency", "drydown", "planter_skip", "water"],
                       help="Target classes for monitoring")
    parser.add_argument("--no-labels", nargs="+", 
                       default=["endrow", "waterway", "double_plant"],
                       help="Classes to consider as normal cases")
    parser.add_argument("--include-normal", action="store_true", 
                       help="Include normal samples (no anomaly)")
    parser.add_argument("--img-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum number of samples to process (default: process all)")

    args = parser.parse_args()

    # Process data
    total, stats = clean_and_process_data(
        args.source,
        args.output,
        selected_classes=args.selected,
        no_label_classes=args.no_labels,
        img_size=tuple(args.img_size),
        include_normal=args.include_normal,
        max_samples=args.max_samples
    )

    # Create train-test split
    if args.test_split > 0:
        input_files = list((Path(args.output) / "inputs").glob("*.npy"))
        np.random.shuffle(input_files)
        split_idx = int(len(input_files) * (1 - args.test_split))
        
        for subset, files in [("train", input_files[:split_idx]), ("test", input_files[split_idx:])]:
            subset_dir = Path(args.output) / subset
            os.makedirs(subset_dir / "inputs", exist_ok=True)
            os.makedirs(subset_dir / "labels", exist_ok=True)
            
            for f in files:
                shutil.copy(f, subset_dir / "inputs" / f.name)
                shutil.copy(Path(args.output) / "labels" / f.name, subset_dir / "labels" / f.name)

        # Split metadata
        df = pd.read_csv(Path(args.output) / "metadata.csv")
        train_ids = [f.stem for f in input_files[:split_idx]]
        test_ids = [f.stem for f in input_files[split_idx:]]
        
        df[df.image_id.isin(train_ids)].to_csv(Path(args.output) / "train_metadata.csv", index=False)
        df[df.image_id.isin(test_ids)].to_csv(Path(args.output) / "test_metadata.csv", index=False)
        print(f"\nDataset split into {len(train_ids)} train and {len(test_ids)} test samples")
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




def clean_and_process_data(source_path, output_path, selected_classes, not_relevant_classes,
                           img_size=(512, 512), max_samples=None,
                           max_per_class=5000):
    source_path = Path(source_path)
    output_path = Path(output_path)

    inputs_dir = output_path / "inputs"
    labels_dir = output_path / "labels"
    boundaries_dir = output_path / "boundaries"

    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(boundaries_dir, exist_ok=True)

    available_classes = [d.name for d in (source_path / "labels").iterdir() if d.is_dir()]
    image_class_map = defaultdict(set)

    all_relevant_classes = set(selected_classes) | set(not_relevant_classes)

    print("Precomputing class mappings...")
    for cls in tqdm(available_classes):
        if cls in all_relevant_classes: 
            cls_dir = source_path / "labels" / cls
            for mask_path in cls_dir.glob("*.png"):
                image_id = mask_path.stem
                image_class_map[image_id].add(cls)

    # get all RGB and NIR images
    
    rgb_paths = {Path(p).stem: p for p in glob(str(source_path / "images" / "rgb" / "*.jpg"))}
    nir_paths = {Path(p).stem: p for p in glob(str(source_path / "images" / "nir" / "*.jpg"))}

    boundaries = {Path(p).stem: p for p in glob(str(source_path / "boundaries" / "*.png"))}

    # validate existence of RGB and NIR images
    all_image_ids = set(rgb_paths.keys()) & set(nir_paths.keys())

    # ensure the image has at least one relevant class mapped

    image_class_map = {
        k: v for k, v in image_class_map.items()
        if k in all_image_ids and any(cls in all_relevant_classes for cls in v)
    }

    # Setup for tracking stats and metadata
    num_selected_classes = len(selected_classes)
    num_output_channels = num_selected_classes + 1 # Selected classes + 1 merged no-stress channel
    print(f"Output mask will have {num_output_channels} channels.")

    # Class mapping for selected classes (channels 0 to num_selected_classes-1)
    selected_class_mapping = {cls: idx for idx, cls in enumerate(selected_classes)}
    # Index for the merged no-stress channel
    no_stress_channel_idx = num_selected_classes

    stats = {cls: 0 for cls in selected_classes}
    stats["no_stress_found"] = 0 # Count images with any no-stress class active
    csv_data = []
    processed_count = 0
    # Track samples per selected class for max_per_class limit
    per_selected_class_count = {cls: 0 for cls in selected_classes}

    print("\nProcessing images...")
    image_ids_to_process = list(image_class_map.keys())

    for image_id in tqdm(image_ids_to_process):
        classes_present_in_image = image_class_map[image_id]

        # Check max_per_class limit for selected classes present in this image
        skip_due_to_limit = False
        temp_selected_present = set() # Track selected classes before limit check

        for cls in classes_present_in_image:
            if cls in selected_class_mapping:
                 # Check if mask file exists and has content before checking limit
                 mask_path = source_path / "labels" / cls / f"{image_id}.png"
                 if mask_path.exists():
                     # Quick check if mask likely has content (optional, could read mask here)
                     # For simplicity, assume if file exists, it might have content
                     temp_selected_present.add(cls)
                 else:
                     continue # Skip if mask file doesn't exist

        # Now check limits only for classes potentially present
        for cls in temp_selected_present:
            if per_selected_class_count[cls] >= max_per_class:
                skip_due_to_limit = True
                break

        if skip_due_to_limit:
            continue

        # Determine if image has any content to process (selected or no-label)
        has_selected_content = False
        has_no_stress_content = False
        output_mask = np.zeros((*img_size, num_output_channels), dtype=np.uint8)
        processed_classes_in_image = set() # Track which classes were actually processed

        # Process selected classes first
        for cls in list(temp_selected_present): # Iterate over classes confirmed to exist
            mask_path = source_path / "labels" / cls / f"{image_id}.png"
            # Re-check existence (though checked above)
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None: continue # Skip if loading failed
                mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0).astype(np.uint8)

                if np.any(binary_mask):
                    class_idx = selected_class_mapping[cls]
                    output_mask[:, :, class_idx] = binary_mask
                    has_selected_content = True
                    processed_classes_in_image.add(cls)
                    # Increment count only if the class has content and limit not reached
                    per_selected_class_count[cls] += 1


        # Process no-label classes and merge into the last channel

        for cls in classes_present_in_image:
            boundarie = cv2.imread(str(boundaries[image_id]), cv2.IMREAD_GRAYSCALE)
            if boundarie is None: continue
            boundarie = cv2.resize(boundarie, img_size, interpolation=cv2.INTER_NEAREST)
            if cls in not_relevant_classes:
                mask_path = source_path / "labels" / cls / f"{image_id}.png"
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None: continue
                    mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
                    binary_mask = (mask_resized > 0).astype(np.uint8)
                    boundarie = (boundarie > 0).astype(np.uint8) # Convert to binary
                    

                    if np.any(binary_mask):
                        # Merge into the last channel using logical OR
                        output_mask[:, :, no_stress_channel_idx] = np.logical_or(
                            output_mask[:, :, no_stress_channel_idx], boundarie, binary_mask
                        ).astype(np.uint8)
                        has_no_stress_content = True
                        processed_classes_in_image.add(cls) # Track that a no-stress class was processed


        # Only save if the image had relevant content and passed limits
        if has_selected_content or has_no_stress_content:

            boundarie = cv2.imread(str(boundaries[image_id]), cv2.IMREAD_GRAYSCALE)
            if boundarie is None: continue
            boundarie = cv2.resize(boundarie, img_size, interpolation=cv2.INTER_NEAREST)
            
            # --- Load Image Data ---
            rgb = cv2.imread(str(rgb_paths[image_id]))
            if rgb is None: continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, img_size, interpolation=cv2.INTER_AREA)

            nir = cv2.imread(str(nir_paths[image_id]), cv2.IMREAD_GRAYSCALE)
            if nir is None: continue
            nir = cv2.resize(nir, img_size, interpolation=cv2.INTER_AREA)

            # --- Save Processed Data ---
            rgbn = np.dstack((rgb, nir))
            
            np.save(inputs_dir / f"{image_id}.npy", rgbn)
            np.save(labels_dir / f"{image_id}.npy", output_mask)
            np.save(boundaries_dir / f"{image_id}.npy", boundarie)

            # --- Record Metadata ---
            record = { "image_id": image_id }
            # Record presence for selected classes processed
            for sel_cls in selected_classes:
                 record[sel_cls] = int(sel_cls in processed_classes_in_image)
            # Record presence for the merged no-stress category
            record["no_stress_found"] = int(has_no_stress_content)
            csv_data.append(record)

            # --- Update Global Stats ---
            for cls in processed_classes_in_image:
                 if cls in selected_classes:
                     stats[cls] += 1 # Increment global count for selected classes
            if has_no_stress_content:
                stats["no_stress_found"] += 1 # Increment global count for no-stress category

            processed_count += 1
            if max_samples and processed_count >= max_samples:
                print(f"\nReached max_samples limit: {max_samples}")
                break

    # --- Save Metadata and Final Statistics ---
    if not csv_data:
        print("\nNo images were processed.")
        return 0, {}
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path / "metadata.csv", index=False)

    print(f"\nFinal statistics (based on images processed):")
    print(f"Total processed images: {processed_count}") # Use actual processed count
    # Use accumulated stats dictionary for counts
    total_processed_for_stats = len(df) # Base percentage on saved images
    if total_processed_for_stats > 0:
        for cls, count in stats.items():
             if count > 0:
                 print(f"- {cls}: {count} images ({count/total_processed_for_stats*100:.1f}%)")
    else:
        print("No images included in final statistics.")

    return total_processed_for_stats, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agriculture Vision Data Preparer (Simplified Multi-Channel)")
    parser.add_argument("--source", required=True, help="Source dataset directory")
    parser.add_argument("--output", required=True, help="Output directory")
    # Class selection
    parser.add_argument("--selected", nargs="+",
                        default=["nutrient_deficiency", "drydown", "planter_skip", "water", "weed_cluster", "storm_damage"],
                        help="Target classes representing stress (each gets its own channel)")
    parser.add_argument("--no-labels", nargs="+",
                        default=["endrow", "waterway", "double_plant"],
                        help="Classes representing 'no-stress' (merged into a single channel)")
    # Processing options
    parser.add_argument("--img-size", type=int, nargs=2, default=[512, 512], help="Target image size (height, width)")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction for test set (0 to disable)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process (default: all)")
    parser.add_argument("--max-per-class", type=int, default=5000,
                        help="Max samples per SELECTED stress class (default: 5000)")

    args = parser.parse_args()
    args.img_size = tuple(args.img_size)

    # --- Process Data ---
    total, stats_dict = clean_and_process_data(
        args.source,
        args.output,
        selected_classes=args.selected,
        not_relevant_classes=args.no_labels,
        img_size=args.img_size,
        max_samples=args.max_samples,
        max_per_class=args.max_per_class
    )

    # --- Create Train-Test Split (same logic as before, using the new metadata) ---
    if args.test_split > 0 and total > 0:
        output_base = Path(args.output)
        metadata_path = output_base / "metadata.csv"
        if not metadata_path.exists():
             print(f"Error: Metadata file not found at {metadata_path}. Cannot perform train/test split.")
        else:
            all_ids_df = pd.read_csv(metadata_path)
            if 'image_id' not in all_ids_df.columns:
                 print("Error: 'image_id' column missing in metadata.csv. Cannot perform train/test split.")
            else:
                all_image_ids = all_ids_df['image_id'].tolist()
                np.random.shuffle(all_image_ids)
                split_idx = int(len(all_image_ids) * (1 - args.test_split))
                train_ids = all_image_ids[:split_idx]
                test_ids = all_image_ids[split_idx:]
                print(f"\nSplitting dataset: {len(train_ids)} train, {len(test_ids)} test samples...")
                # (Copying logic remains the same)
                for subset, ids in [("train", train_ids), ("test", test_ids)]:
                    subset_dir = output_base / subset
                    subset_inputs = subset_dir / "inputs"
                    subset_labels = subset_dir / "labels"
                    os.makedirs(subset_inputs, exist_ok=True)
                    os.makedirs(subset_labels, exist_ok=True)
                    for image_id in tqdm(ids, desc=f"Copying {subset} files"):
                        src_input = output_base / "inputs" / f"{image_id}.npy"
                        if src_input.exists(): shutil.copy(src_input, subset_inputs / f"{image_id}.npy")
                        src_label = output_base / "labels" / f"{image_id}.npy"
                        if src_label.exists(): shutil.copy(src_label, subset_labels / f"{image_id}.npy")
                df_train = all_ids_df[all_ids_df.image_id.isin(train_ids)]
                df_test = all_ids_df[all_ids_df.image_id.isin(test_ids)]
                df_train.to_csv(output_base / "train_metadata.csv", index=False)
                df_test.to_csv(output_base / "test_metadata.csv", index=False)
                print(f"Train/Test split metadata saved.")

    elif total == 0: print("\nNo images processed, skipping train/test split.")
    else: print("\nTest split not requested (test_split=0).")
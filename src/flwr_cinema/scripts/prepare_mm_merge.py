"""Prepare a unified MM_merge directory from raw MnMs1 and MnMs2 downloads.

This script takes the original MnMs1 (OpenDataset) and MnMs2 directories,
extracts ED/ES frames, renames patients with unique sequential IDs,
and generates the master CSV required by data/mnms_merged/preprocess.py.

MnMs1 structure (4D files, need frame extraction):
  OpenDataset/
    Training/Labeled/{pid}/{pid}_sa.nii.gz       (4D)
    Training/Labeled/{pid}/{pid}_sa_gt.nii.gz     (4D)
    Validation/{pid}/...
    Testing/{pid}/...
    211230_M&Ms_Dataset_information_diagnosis_opendataset.csv

MnMs2 structure (3D files, already split):
  MnM2/
    dataset/{pid:03d}/{pid:03d}_SA_ED.nii.gz      (3D)
    dataset/{pid:03d}/{pid:03d}_SA_ED_gt.nii.gz    (3D)
    dataset/{pid:03d}/{pid:03d}_SA_ES.nii.gz       (3D)
    dataset/{pid:03d}/{pid:03d}_SA_ES_gt.nii.gz    (3D)
    dataset_information.csv

Output MM_merge structure (what data/mnms_merged/preprocess.py expects):
  MM_merge/
    {MERGING_CODE:03d}/{MERGING_CODE:03d}_ED.nii.gz
    {MERGING_CODE:03d}/{MERGING_CODE:03d}_ED_gt.nii.gz
    {MERGING_CODE:03d}/{MERGING_CODE:03d}_ES.nii.gz
    {MERGING_CODE:03d}/{MERGING_CODE:03d}_ES_gt.nii.gz
    20241305_dataset_information.csv
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_and_save_frame(image_4d_path: Path, frame_index: int, out_path: Path, is_label: bool = False) -> None:
    """Extract a single frame from a 4D NIfTI and save as 3D.

    Args:
        image_4d_path: Path to 4D NIfTI file.
        frame_index: Frame index to extract (0-based).
        out_path: Output path for the 3D NIfTI.
        is_label: If True, read as UInt8 (for segmentation masks).
    """
    if is_label:
        img = sitk.ReadImage(str(image_4d_path), outputPixelType=sitk.sitkUInt8)
    else:
        img = sitk.ReadImage(str(image_4d_path))
    frame = img[:, :, :, frame_index]
    sitk.WriteImage(frame, str(out_path), useCompression=True)


def prepare_mnms1(mnms1_dir: Path, out_dir: Path, start_code: int) -> tuple:
    """Process MnMs1 dataset: extract ED/ES frames from 4D files.

    Args:
        mnms1_dir: Path to MnMs1 OpenDataset directory.
        out_dir: Output MM_merge directory.
        start_code: Starting MERGING_CODE integer.

    Returns:
        Tuple of (DataFrame with records, next available code).
    """
    logger.info("Processing MnMs1 dataset...")

    csv_path = mnms1_dir / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    if not csv_path.exists():
        logger.error(f"MnMs1 CSV not found at {csv_path}")
        return pd.DataFrame(), start_code

    df = pd.read_csv(csv_path)
    # Drop the first unnamed index column if present
    if df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]

    records = []
    current_code = start_code

    # Search for patient dirs across all splits
    split_dirs = {
        "Training/Labeled": "train",
        "Validation": "val",
        "Testing": "test",
    }

    # Build a map: pid -> (patient_dir, split)
    pid_to_info = {}
    for split_subdir, split_name in split_dirs.items():
        split_path = mnms1_dir / split_subdir
        if not split_path.exists():
            continue
        for patient_dir in split_path.iterdir():
            if patient_dir.is_dir():
                pid_to_info[patient_dir.name] = (patient_dir, split_name)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MnMs1"):
        pid = str(row["External code"])
        if pid not in pid_to_info:
            logger.warning(f"MnMs1: Patient {pid} directory not found. Skipping.")
            continue

        patient_dir, original_split = pid_to_info[pid]
        sa_path = patient_dir / f"{pid}_sa.nii.gz"
        sa_gt_path = patient_dir / f"{pid}_sa_gt.nii.gz"

        if not sa_path.exists() or not sa_gt_path.exists():
            logger.warning(f"MnMs1: Missing NIfTI for {pid}. Skipping.")
            continue

        ed_index = int(row["ED"])
        es_index = int(row["ES"])

        new_pid = f"{current_code:03d}"
        new_dir = out_dir / new_pid
        new_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract ED/ES frames from 4D -> 3D
            extract_and_save_frame(sa_path, ed_index, new_dir / f"{new_pid}_ED.nii.gz", is_label=False)
            extract_and_save_frame(sa_gt_path, ed_index, new_dir / f"{new_pid}_ED_gt.nii.gz", is_label=True)
            extract_and_save_frame(sa_path, es_index, new_dir / f"{new_pid}_ES.nii.gz", is_label=False)
            extract_and_save_frame(sa_gt_path, es_index, new_dir / f"{new_pid}_ES_gt.nii.gz", is_label=True)
        except Exception as e:
            logger.warning(f"MnMs1: Error extracting frames for {pid}: {e}. Skipping.")
            shutil.rmtree(new_dir, ignore_errors=True)
            continue

        records.append({
            "MERGING_CODE": current_code,
            "ID_MM1": pid,
            "ID_MM2": np.nan,
            "Vendor1": row.get("Vendor", row.get("VendorName", "")),
            "Vendor2": np.nan,
            "Pathology1": row.get("Pathology", ""),
            "Pathology2": np.nan,
            "Age": row.get("Age", np.nan),
            "Sex": row.get("Sex", ""),
            "Height": row.get("Height", np.nan),
            "Weight": row.get("Weight", np.nan),
            "Centre": row.get("Centre", ""),
            "Original_Split": original_split,
            "Source": "MnMs1",
        })
        current_code += 1

    logger.info(f"MnMs1: Processed {len(records)} patients (codes {start_code}-{current_code - 1})")
    return pd.DataFrame(records), current_code


def prepare_mnms2(mnms2_dir: Path, out_dir: Path, start_code: int) -> tuple:
    """Process MnMs2 dataset: copy 3D ED/ES files with new IDs.

    Args:
        mnms2_dir: Path to MnMs2 directory.
        out_dir: Output MM_merge directory.
        start_code: Starting MERGING_CODE integer.

    Returns:
        Tuple of (DataFrame with records, next available code).
    """
    logger.info("Processing MnMs2 dataset...")

    csv_path = mnms2_dir / "dataset_information.csv"
    if not csv_path.exists():
        logger.error(f"MnMs2 CSV not found at {csv_path}")
        return pd.DataFrame(), start_code

    df = pd.read_csv(csv_path).dropna()
    data_dir = mnms2_dir / "dataset"
    if not data_dir.exists():
        # Maybe files are directly in the root
        data_dir = mnms2_dir

    records = []
    current_code = start_code

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MnMs2"):
        pid_int = int(row["SUBJECT_CODE"])
        pid_old = f"{pid_int:03d}"
        patient_dir = data_dir / pid_old

        if not patient_dir.exists():
            logger.warning(f"MnMs2: Patient dir {patient_dir} not found. Skipping.")
            continue

        new_pid = f"{current_code:03d}"
        new_dir = out_dir / new_pid
        new_dir.mkdir(parents=True, exist_ok=True)

        # MnMs2 files: {pid}_SA_ED.nii.gz, {pid}_SA_ED_gt.nii.gz, etc.
        file_mapping = {
            f"{pid_old}_SA_ED.nii.gz": f"{new_pid}_ED.nii.gz",
            f"{pid_old}_SA_ED_gt.nii.gz": f"{new_pid}_ED_gt.nii.gz",
            f"{pid_old}_SA_ES.nii.gz": f"{new_pid}_ES.nii.gz",
            f"{pid_old}_SA_ES_gt.nii.gz": f"{new_pid}_ES_gt.nii.gz",
        }

        all_found = True
        for old_name, new_name in file_mapping.items():
            src = patient_dir / old_name
            if not src.exists():
                logger.warning(f"MnMs2: Missing {src}")
                all_found = False
                break

        if not all_found:
            shutil.rmtree(new_dir, ignore_errors=True)
            continue

        for old_name, new_name in file_mapping.items():
            shutil.copy2(patient_dir / old_name, new_dir / new_name)

        # Determine original split from MnMs2 convention
        if pid_int <= 160:
            original_split = "train"
        elif pid_int <= 200:
            original_split = "val"
        else:
            original_split = "test"

        records.append({
            "MERGING_CODE": current_code,
            "ID_MM1": np.nan,
            "ID_MM2": pid_int,
            "Vendor1": np.nan,
            "Vendor2": row.get("VENDOR", ""),
            "Pathology1": np.nan,
            "Pathology2": row.get("DISEASE", ""),
            "Age": np.nan,
            "Sex": "",
            "Height": np.nan,
            "Weight": np.nan,
            "Scanner": row.get("SCANNER", ""),
            "Field": row.get("FIELD", np.nan),
            "Original_Split": original_split,
            "Source": "MnMs2",
        })
        current_code += 1

    logger.info(f"MnMs2: Processed {len(records)} patients (codes {start_code}-{current_code - 1})")
    return pd.DataFrame(records), current_code


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MM_merge directory from raw MnMs1 and MnMs2 downloads."
    )
    parser.add_argument(
        "--mnms1_dir", type=Path, required=True,
        help="Path to raw MnMs1 OpenDataset directory (contains Training/, Validation/, Testing/ and the CSV)",
    )
    parser.add_argument(
        "--mnms2_dir", type=Path, required=True,
        help="Path to raw MnMs2 directory (contains dataset/ folder and dataset_information.csv)",
    )
    parser.add_argument(
        "--out_dir", type=Path, required=True,
        help="Output MM_merge directory",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Process MnMs1 first (codes starting from 1)
    df1, next_code = prepare_mnms1(args.mnms1_dir, args.out_dir, start_code=1)

    # Process MnMs2 (codes continue from where MnMs1 left off)
    df2, final_code = prepare_mnms2(args.mnms2_dir, args.out_dir, start_code=next_code)

    # Combine and save master CSV
    all_dfs = [d for d in [df1, df2] if not d.empty]
    if not all_dfs:
        logger.error("No data processed. Check your input paths.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns
    cols = merged_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("MERGING_CODE")))
    merged_df = merged_df[cols]

    output_csv = args.out_dir / "20241305_dataset_information.csv"
    merged_df.to_csv(output_csv, index=False)

    logger.info("=" * 60)
    logger.info(f"DONE: {len(merged_df)} patients total")
    logger.info(f"  MnMs1: {len(df1)} patients")
    logger.info(f"  MnMs2: {len(df2)} patients")
    logger.info(f"  Master CSV: {output_csv}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step: run the preprocessor:")
    logger.info(f'  python data/mnms_merged/preprocess.py \\')
    logger.info(f'    --data_dir "{args.out_dir}" \\')
    logger.info(f'    --info_csv "{output_csv}" \\')
    logger.info(f'    --out_dir "C:\\Users\\Rodo\\.cache\\cinema_datasets\\mnms_merged\\processed"')


if __name__ == "__main__":
    main()

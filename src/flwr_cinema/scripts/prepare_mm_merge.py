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
import hashlib
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_scalar_3d(volume: np.ndarray, src_path: Path) -> np.ndarray:
    """Ensure a scalar 3D array (drop singleton axes, reject unsupported shapes)."""
    arr = np.asarray(volume)

    if arr.ndim == 4:
        arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze for {src_path}, got shape {arr.shape}")

    return arr


def sanitize_volume_array(volume: np.ndarray, is_label: bool) -> np.ndarray:
    """Sanitize volume values to avoid downstream preprocessing failures."""
    if is_label:
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        return np.rint(volume).astype(np.uint8)

    finite_mask = np.isfinite(volume)
    if not finite_mask.any():
        return np.zeros_like(volume, dtype=np.float32)

    finite_values = volume[finite_mask]
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    volume = np.nan_to_num(volume, nan=vmin, posinf=vmax, neginf=vmin)
    return volume.astype(np.float32, copy=False)


def write_sanitized_nifti(src_path: Path, dst_path: Path, is_label: bool) -> None:
    """Rewrite a NIfTI file with sanitized values and orthonormal direction."""
    try:
        src_img = sitk.ReadImage(str(src_path), outputPixelType=sitk.sitkUInt8 if is_label else sitk.sitkFloat32)
        volume = sitk.GetArrayFromImage(src_img)
        spacing = src_img.GetSpacing()
        origin = src_img.GetOrigin()
    except Exception:
        nib_img = nib.load(str(src_path))
        volume = np.asanyarray(nib_img.dataobj)
        spacing = tuple(float(x) for x in nib_img.header.get_zooms()[:3])
        origin = tuple(float(x) for x in nib_img.affine[:3, 3])

    volume = ensure_scalar_3d(volume, src_path=src_path)
    volume = sanitize_volume_array(volume, is_label=is_label)
    out_img = sitk.GetImageFromArray(volume)
    out_img.SetSpacing(tuple(float(s) for s in spacing[:3]))
    out_img.SetOrigin(tuple(float(o) for o in origin[:3]))
    out_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(out_img, str(dst_path), useCompression=True)


def build_patient_signature(image_paths: list[Path]) -> str:
    """Build a stable signature from ED/ES image and mask voxel content.

    The signature ignores filenames and container/header differences so the
    same patient can match across MnMs1 and MnMs2 even when files were
    re-encoded or renamed.
    """
    hasher = hashlib.sha1()
    for image_path in image_paths:
        volume = None

        # nibabel is usually more permissive than ITK for malformed direction metadata.
        try:
            import nibabel as nib

            volume = np.asanyarray(nib.load(str(image_path)).dataobj)
        except Exception:
            volume = None

        if volume is None:
            try:
                volume = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
            except Exception:
                # Last-resort fallback for unreadable files.
                with image_path.open("rb") as f:
                    volume = f.read()

        if isinstance(volume, bytes):
            hasher.update(volume)
            continue

        volume = ensure_scalar_3d(volume, src_path=image_path)

        hasher.update(str(volume.shape).encode("utf-8"))
        hasher.update(str(volume.dtype).encode("utf-8"))
        hasher.update(np.ascontiguousarray(volume).tobytes())
    return hasher.hexdigest()


def first_non_empty(values: pd.Series):
    """Return the first value that is not NaN and not an empty string."""
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return np.nan


def consolidate_by_merging_code(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows sharing MERGING_CODE into one merged row."""
    if df.empty:
        return df

    grouped_rows = []
    columns = df.columns.tolist()
    for merging_code, group in df.groupby("MERGING_CODE", sort=True, dropna=False):
        merged_row = {"MERGING_CODE": merging_code}
        for col in columns:
            if col == "MERGING_CODE":
                continue
            merged_row[col] = first_non_empty(group[col])

        if "Source" in group.columns:
            unique_sources = sorted([str(s) for s in group["Source"].dropna().unique() if str(s).strip()])
            if len(unique_sources) > 1:
                merged_row["Source"] = "+".join(unique_sources)

        grouped_rows.append(merged_row)

    return pd.DataFrame(grouped_rows)


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

    # Rewrite frame with sanitized values and orthonormal direction.
    frame_arr = sitk.GetArrayFromImage(frame)
    frame_arr = ensure_scalar_3d(frame_arr, src_path=image_4d_path)
    frame_arr = sanitize_volume_array(frame_arr, is_label=is_label)
    out_img = sitk.GetImageFromArray(frame_arr)
    out_img.SetSpacing(tuple(float(s) for s in frame.GetSpacing()[:3]))
    out_img.SetOrigin(tuple(float(o) for o in frame.GetOrigin()[:3]))
    out_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(out_img, str(out_path), useCompression=True)


def prepare_mnms1(mnms1_dir: Path, out_dir: Path, start_code: int) -> tuple:
    """Process MnMs1 dataset: extract ED/ES frames from 4D files.

    Args:
        mnms1_dir: Path to MnMs1 OpenDataset directory.
        out_dir: Output MM_merge directory.
        start_code: Starting MERGING_CODE integer.

    Returns:
        Tuple of (DataFrame with records, next available code, signature map).
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
    signature_to_code = {}

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

            signature = build_patient_signature([
                new_dir / f"{new_pid}_ED.nii.gz",
                new_dir / f"{new_pid}_ED_gt.nii.gz",
                new_dir / f"{new_pid}_ES.nii.gz",
                new_dir / f"{new_pid}_ES_gt.nii.gz",
            ])
        except Exception as e:
            logger.warning(f"MnMs1: Error extracting frames for {pid}: {e}. Skipping.")
            shutil.rmtree(new_dir, ignore_errors=True)
            continue

        if signature in signature_to_code:
            logger.warning(
                "MnMs1: Duplicate image signature detected for patient %s. Keeping first occurrence with MERGING_CODE=%s.",
                pid,
                signature_to_code[signature],
            )
            shutil.rmtree(new_dir, ignore_errors=True)
            continue

        signature_to_code[signature] = current_code

        records.append({
            "MERGING_CODE": current_code,
            "ID_MM1": pid,
            "ID_MM2": np.nan,
            "Vendor1": row.get("VendorName", row.get("Vendor", "")),
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
    return pd.DataFrame(records), current_code, signature_to_code


def prepare_mnms2(
    mnms2_dir: Path,
    out_dir: Path,
    start_code: int,
    signature_to_code: dict[str, int] | None = None,
) -> tuple:
    """Process MnMs2 dataset: copy 3D ED/ES files with new IDs.

    Args:
        mnms2_dir: Path to MnMs2 directory.
        out_dir: Output MM_merge directory.
        start_code: Starting MERGING_CODE integer.

    Returns:
        Tuple of (DataFrame with records, next available code, updated signature map).
    """
    logger.info("Processing MnMs2 dataset...")

    csv_path = mnms2_dir / "dataset_information.csv"
    if not csv_path.exists():
        logger.error(f"MnMs2 CSV not found at {csv_path}")
        return pd.DataFrame(), start_code

    # Keep text columns as strings to avoid mixed-type parsing warnings.
    df = pd.read_csv(csv_path, dtype=str, low_memory=False).dropna(subset=["SUBJECT_CODE"])
    data_dir = mnms2_dir / "dataset"
    if not data_dir.exists():
        # Maybe files are directly in the root
        data_dir = mnms2_dir

    records = []
    current_code = start_code
    if signature_to_code is None:
        signature_to_code = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MnMs2"):
        pid_int = int(row["SUBJECT_CODE"])
        pid_old = f"{pid_int:03d}"
        patient_dir = data_dir / pid_old

        if not patient_dir.exists():
            logger.warning(f"MnMs2: Patient dir {patient_dir} not found. Skipping.")
            continue

        # MnMs2 files: {pid}_SA_ED.nii.gz, {pid}_SA_ED_gt.nii.gz, etc.
        required_files = [
            f"{pid_old}_SA_ED.nii.gz",
            f"{pid_old}_SA_ED_gt.nii.gz",
            f"{pid_old}_SA_ES.nii.gz",
            f"{pid_old}_SA_ES_gt.nii.gz",
        ]

        all_found = True
        for old_name in required_files:
            src = patient_dir / old_name
            if not src.exists():
                logger.warning(f"MnMs2: Missing {src}")
                all_found = False
                break

        if not all_found:
            continue

        signature = build_patient_signature([patient_dir / name for name in required_files])

        # Determine original split from MnMs2 convention
        if pid_int <= 160:
            original_split = "train"
        elif pid_int <= 200:
            original_split = "val"
        else:
            original_split = "test"

        if signature in signature_to_code:
            matched_code = signature_to_code[signature]
            logger.info(
                "MnMs2: Found duplicate with existing MERGING_CODE=%s for SUBJECT_CODE=%s. Merging metadata.",
                matched_code,
                pid_old,
            )
            records.append({
                "MERGING_CODE": matched_code,
                "ID_MM1": np.nan,
                "ID_MM2": pid_int,
                "Vendor1": np.nan,
                "Vendor2": row.get("Vendor", row.get("VENDOR", "")),
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
            continue

        new_pid = f"{current_code:03d}"
        new_dir = out_dir / new_pid
        new_dir.mkdir(parents=True, exist_ok=True)
        file_mapping = {
            f"{pid_old}_SA_ED.nii.gz": f"{new_pid}_ED.nii.gz",
            f"{pid_old}_SA_ED_gt.nii.gz": f"{new_pid}_ED_gt.nii.gz",
            f"{pid_old}_SA_ES.nii.gz": f"{new_pid}_ES.nii.gz",
            f"{pid_old}_SA_ES_gt.nii.gz": f"{new_pid}_ES_gt.nii.gz",
        }
        for old_name, new_name in file_mapping.items():
            src = patient_dir / old_name
            dst = new_dir / new_name
            write_sanitized_nifti(src, dst, is_label=new_name.endswith("_gt.nii.gz"))

        records.append({
            "MERGING_CODE": current_code,
            "ID_MM1": np.nan,
            "ID_MM2": pid_int,
            "Vendor1": np.nan,
            "Vendor2": row.get("Vendor", row.get("VENDOR", "")),
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
        signature_to_code[signature] = current_code
        current_code += 1

    logger.info(f"MnMs2: Processed {len(records)} patients (codes {start_code}-{current_code - 1})")
    return pd.DataFrame(records), current_code, signature_to_code


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
    df1, next_code, signature_to_code = prepare_mnms1(args.mnms1_dir, args.out_dir, start_code=1)

    # Process MnMs2 (codes continue from where MnMs1 left off)
    df2, _, _ = prepare_mnms2(
        args.mnms2_dir,
        args.out_dir,
        start_code=next_code,
        signature_to_code=signature_to_code,
    )

    # Combine and save master CSV
    all_dfs = [d for d in [df1, df2] if not d.empty]
    if not all_dfs:
        logger.error("No data processed. Check your input paths.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = consolidate_by_merging_code(merged_df)

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
    logger.info("  python data/mnms_merged/preprocess.py \\")
    logger.info(f'    --data_dir "{args.out_dir}" \\')
    logger.info(f'    --info_csv "{output_csv}" \\')
    logger.info('    --out_dir "C:\\Users\\Rodo\\.cache\\cinema_datasets\\mnms_merged\\processed"')


if __name__ == "__main__":
    main()

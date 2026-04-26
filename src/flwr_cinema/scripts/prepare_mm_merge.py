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


def load_reference_mapping(ref_csv_path: Path | None) -> tuple[dict[str, int], dict[str, int]]:
    """Load ID mappings from reference CSV if provided."""
    if not ref_csv_path or not ref_csv_path.exists():
        logger.warning("No reference CSV provided or found. Overlaps will not be detected!")
        return {}, {}
    
    df = pd.read_csv(ref_csv_path, low_memory=False)
    mm1_to_mm2 = {}
    mm1_to_merging_code = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("ID_MM1")):
            pid = str(row["ID_MM1"])
            mm1_to_merging_code[pid] = int(row["MERGING_CODE"])
            if pd.notna(row.get("ID_MM2")):
                mm1_to_mm2[pid] = int(row["ID_MM2"])
    
    logger.info("Loaded %d overlap mappings and %d total mappings from reference CSV.", len(mm1_to_mm2), len(mm1_to_merging_code))
    return mm1_to_mm2, mm1_to_merging_code


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
    try:
        if is_label:
            img = sitk.ReadImage(str(image_4d_path), outputPixelType=sitk.sitkUInt8)
        else:
            img = sitk.ReadImage(str(image_4d_path))
        
        # the 4th dimension in SimpleITK is usually the last one [x, y, z, t]
        frame = img[:, :, :, frame_index]

        # Rewrite frame with sanitized values and orthonormal direction.
        frame_arr = sitk.GetArrayFromImage(frame)
        spacing = img.GetSpacing()[:3]
        origin = img.GetOrigin()[:3]
    except Exception as e:
        logger.warning(f"MnMs1: SimpleITK read failed for {image_4d_path.name} ({e}). Falling back to nibabel.")
        import nibabel as nib
        nib_img = nib.load(str(image_4d_path))
        # nibabel shape is (x, y, z, t)
        data = np.asanyarray(nib_img.dataobj)
        frame_arr = data[..., frame_index]
        # nibabel array shape is (x,y,z), but sitk expects (z,y,x), so we transpose
        frame_arr = frame_arr.T
        
        # Get spacing from header zooms
        zooms = nib_img.header.get_zooms()
        spacing = tuple(float(s) for s in zooms[:3])
        
        # NIfTI affine gives translation (origin) in last column
        affine = nib_img.affine
        # Convert RAS (nibabel) to LPS (ITK) for origin
        origin = (float(-affine[0,3]), float(-affine[1,3]), float(affine[2,3]))

    frame_arr = ensure_scalar_3d(frame_arr, src_path=image_4d_path)
    frame_arr = sanitize_volume_array(frame_arr, is_label=is_label)
    out_img = sitk.GetImageFromArray(frame_arr)
    out_img.SetSpacing(tuple(float(s) for s in spacing))
    out_img.SetOrigin(tuple(float(o) for o in origin))
    out_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(out_img, str(out_path), useCompression=True)


def prepare_mnms1(
    mnms1_dir: Path,
    out_dir: Path,
    start_code: int,
    mm1_to_mm2: dict[str, int],
    mm1_to_merging_code: dict[str, int],
) -> tuple:
    """Process MnMs1 dataset: extract ED/ES frames from 4D files.

    Args:
        mnms1_dir: Path to MnMs1 OpenDataset directory.
        out_dir: Output MM_merge directory.
        start_code: Starting MERGING_CODE integer.
        mm1_to_mm2: Mapping of ID_MM1 to ID_MM2 for detecting overlaps.
        mm1_to_merging_code: Mapping of ID_MM1 to exact MERGING_CODE to preserve order.

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

    # Ordered by desired output sequence for MnMs1-only patients
    split_order = [
        ("Validation", "val"),
        ("Training/Labeled", "train"),
        ("Training/Unlabeled", "train"),
        ("Testing", "test"),
    ]

    # Build a map: pid -> (patient_dir, split_name, order_idx)
    pid_to_info = {}
    for order_idx, (split_subdir, split_name) in enumerate(split_order):
        split_path = mnms1_dir / split_subdir
        if not split_path.exists():
            continue
        for patient_dir in split_path.iterdir():
            if patient_dir.is_dir():
                pid_to_info[patient_dir.name] = (patient_dir, split_name, order_idx)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MnMs1"):
        pid = str(row["External code"])
        if pid not in pid_to_info:
            logger.warning(f"MnMs1: Patient {pid} directory not found. Skipping.")
            continue

        patient_dir, _split_name, _order_idx = pid_to_info[pid]
        sa_path = patient_dir / f"{pid}_sa.nii.gz"
        sa_gt_path = patient_dir / f"{pid}_sa_gt.nii.gz"

        if not sa_path.exists() or not sa_gt_path.exists():
            logger.warning(f"MnMs1: Missing NIfTI for {pid}. Skipping.")
            continue

        if pid in mm1_to_mm2:
            matched_code = mm1_to_merging_code.get(pid, mm1_to_mm2[pid])
            logger.info(
                "MnMs1: Patient %s matches existing MERGING_CODE=%s (MnMs2 overlap). Merging metadata.",
                pid, matched_code,
            )
            records.append({
                "MERGING_CODE": matched_code,
                "ID_MM1": pid,
                "ID_MM2": np.nan,
                "Vendor1": row.get("VendorName", row.get("Vendor", "")),
                "Vendor2": np.nan,
                "Scanner": np.nan,
                "Field": np.nan,
                "Pathology1": row.get("Pathology", ""),
                "Pathology2": np.nan,
                "Age": row.get("Age", np.nan),
                "Sex": row.get("Sex", ""),
                "Height": row.get("Height", np.nan),
                "Weight": row.get("Weight", np.nan),
            })
            continue

        ed_index = int(row["ED"])
        es_index = int(row["ES"])

        if pid in mm1_to_merging_code:
            this_code = mm1_to_merging_code[pid]
        else:
            this_code = current_code
            current_code += 1

        new_pid = f"{this_code:03d}"
        new_dir = out_dir / new_pid
        new_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract ED/ES frames from 4D -> 3D
            extract_and_save_frame(sa_path, ed_index, new_dir / f"{new_pid}_ED.nii.gz", is_label=False)
            extract_and_save_frame(sa_gt_path, ed_index, new_dir / f"{new_pid}_ED_gt.nii.gz", is_label=True)
            extract_and_save_frame(sa_path, es_index, new_dir / f"{new_pid}_ES.nii.gz", is_label=False)
            extract_and_save_frame(sa_gt_path, es_index, new_dir / f"{new_pid}_ES_gt.nii.gz", is_label=True)

            # Validate GT has actual segmentation (not just background)
            ed_gt_arr = sitk.GetArrayFromImage(
                sitk.ReadImage(str(new_dir / f"{new_pid}_ED_gt.nii.gz"), sitk.sitkUInt8)
            )
            es_gt_arr = sitk.GetArrayFromImage(
                sitk.ReadImage(str(new_dir / f"{new_pid}_ES_gt.nii.gz"), sitk.sitkUInt8)
            )
            if not (ed_gt_arr > 0).any() and not (es_gt_arr > 0).any():
                logger.warning(
                    "MnMs1: Patient %s has empty GT (ED+ES all background). Skipping.", pid,
                )
                shutil.rmtree(new_dir, ignore_errors=True)
                continue

        except Exception as e:
            logger.warning(f"MnMs1: Error extracting frames for {pid}: {e}. Skipping.")
            shutil.rmtree(new_dir, ignore_errors=True)
            continue

        records.append({
            "MERGING_CODE": this_code,
            "ID_MM1": pid,
            "ID_MM2": np.nan,
            "Vendor1": row.get("VendorName", row.get("Vendor", "")),
            "Vendor2": np.nan,
            "Scanner": np.nan,
            "Field": np.nan,
            "Pathology1": row.get("Pathology", ""),
            "Pathology2": np.nan,
            "Age": row.get("Age", np.nan),
            "Sex": row.get("Sex", ""),
            "Height": row.get("Height", np.nan),
            "Weight": row.get("Weight", np.nan),
        })

    logger.info(f"MnMs1: Processed {len(records)} patients")
    return pd.DataFrame(records), current_code


def prepare_mnms2(
    mnms2_dir: Path,
    out_dir: Path,
    start_code: int,
) -> tuple:
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

    # Keep text columns as strings to avoid mixed-type parsing warnings.
    df = pd.read_csv(csv_path, dtype=str, low_memory=False).dropna(subset=["SUBJECT_CODE"])
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
            "Scanner": row.get("SCANNER", ""),
            "Field": row.get("FIELD", np.nan),
            "Pathology1": np.nan,
            "Pathology2": row.get("DISEASE", ""),
            "Age": np.nan,
            "Sex": "",
            "Height": np.nan,
            "Weight": np.nan,
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
        "--out_dir",
        type=Path,
        default=Path("docs/mnms_merged"),
        help="Output MM_merge directory",
    )
    parser.add_argument(
        "--reference_csv", type=Path, default=None,
        help="Path to reference CSV (e.g., docs/20241305_dataset_information_joan.csv) to load ID mappings.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    mm1_to_mm2, mm1_to_merging_code = load_reference_mapping(args.reference_csv)

    # Process MnMs2 first (codes 1..360 = SUBJECT_CODE)
    df2, next_code = prepare_mnms2(
        args.mnms2_dir, args.out_dir, start_code=1,
    )

    # Process MnMs1 (codes continue from 361+)
    df1, _ = prepare_mnms1(
        args.mnms1_dir, args.out_dir, start_code=next_code,
        mm1_to_mm2=mm1_to_mm2,
        mm1_to_merging_code=mm1_to_merging_code,
    )

    # Combine and save master CSV
    all_dfs = [d for d in [df1, df2] if not d.empty]
    if not all_dfs:
        logger.error("No data processed. Check your input paths.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = consolidate_by_merging_code(merged_df)

    col_order = [
        "MERGING_CODE", "ID_MM1", "ID_MM2", "Vendor1", "Vendor2",
        "Scanner", "Field", "Pathology1", "Pathology2",
        "Age", "Sex", "Height", "Weight",
    ]
    merged_df = merged_df[col_order]

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

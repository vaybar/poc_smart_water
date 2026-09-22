"""
fix_water_meter_labels.py - Canonical quad label fixer for water_meter dataset.

Re-indexes the 4 corner keypoints in existing YOLO label files (water_meter/labels/**/*.txt)
so that Edge 0->1 is ALWAYS the long reading edge (TL->TR), and Edge 0->3 is ALWAYS the short edge (TL->BL).
Runs directly in Google Colab on the existing water_meter folder in ~5 seconds without needing original masks.
"""

from pathlib import Path
import numpy as np
from tqdm import tqdm
import config

def canonical_quad_sort(pts: np.ndarray) -> np.ndarray:
    """
    Sorts 4 corners of an elongated dial box canonically:
    - Edge 0 -> 1 is ALWAYS the long edge (reading direction: TL -> TR).
    - Edge 0 -> 3 is ALWAYS the short edge (thickness: TL -> BL).
    - Order is strictly TL, TR, BR, BL in clockwise direction.
    """
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts_cw = pts[np.argsort(angles)]

    edges = [pts_cw[(i + 1) % 4] - pts_cw[i] for i in range(4)]
    lengths = [np.linalg.norm(e) for e in edges]

    if (lengths[0] + lengths[2]) > (lengths[1] + lengths[3]):
        long_edge_candidates = [(0, 1), (2, 3)]
    else:
        long_edge_candidates = [(1, 2), (3, 0)]

    def edge_score(pair):
        pA, pB = pts_cw[pair[0]], pts_cw[pair[1]]
        return (pA[1] + pB[1]) / 2.0

    top_pair = min(long_edge_candidates, key=edge_score)
    idxA, idxB = top_pair
    pA, pB = pts_cw[idxA], pts_cw[idxB]

    if abs(pA[0] - pB[0]) > 1e-2:
        if pA[0] < pB[0]:
            tl_idx, tr_idx = idxA, idxB
        else:
            tl_idx, tr_idx = idxB, idxA
    else:
        if pA[1] < pB[1]:
            tl_idx, tr_idx = idxA, idxB
        else:
            tl_idx, tr_idx = idxB, idxA

    neighbors_tl = [(tl_idx - 1) % 4, (tl_idx + 1) % 4]
    bl_idx = [n for n in neighbors_tl if n != tr_idx][0]

    neighbors_tr = [(tr_idx - 1) % 4, (tr_idx + 1) % 4]
    br_idx = [n for n in neighbors_tr if n != tl_idx][0]

    return np.array([pts_cw[tl_idx], pts_cw[tr_idx], pts_cw[br_idx], pts_cw[bl_idx]], dtype=np.float32)

def fix_labels(dataset_dir: Path = config.DATASET_ROOT):
    dataset_dir = dataset_dir.resolve()
    labels_dir = dataset_dir / "labels"
    if not labels_dir.exists():
        # Fallback if run from different cwd
        if Path("water_meter/labels").exists():
            labels_dir = Path("water_meter/labels").resolve()
        elif Path("../water_meter/labels").exists():
            labels_dir = Path("../water_meter/labels").resolve()
        else:
            print(f"[ERROR] Labels directory not found at: {labels_dir}")
            return

    txt_files = sorted(list(labels_dir.glob("**/*.txt")))
    print(f"Found {len(txt_files)} label files in {labels_dir}")

    fixed_count = 0
    unchanged_count = 0

    for txt_p in tqdm(txt_files, desc="Fixing labels"):
        try:
            line = txt_p.read_text(encoding="utf-8").strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 13:
                continue

            clase = tokens[0]
            cx, cy, w, h = [float(v) for v in tokens[1:5]]
            pts = np.array([float(v) for v in tokens[5:13]], dtype=np.float32).reshape(4, 2)

            ord_pts = canonical_quad_sort(pts)

            # Check if order changed
            if not np.allclose(pts, ord_pts, atol=1e-5):
                fixed_count += 1
            else:
                unchanged_count += 1

            new_tokens = [clase, f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for kx, ky in ord_pts:
                new_tokens.extend([f"{kx:.6f}", f"{ky:.6f}"])

            txt_p.write_text(" ".join(new_tokens) + "\n", encoding="utf-8")
        except Exception:
            continue

    print(f"\n[DONE] Finished fixing labels:")
    print(f"  Fixed (re-ordered) : {fixed_count}")
    print(f"  Already canonical  : {unchanged_count}")
    print(f"  Total processed    : {fixed_count + unchanged_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix water_meter labels to canonical quad alignment")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_ROOT, help="Path to water_meter folder")
    args = parser.parse_args()
    fix_labels(args.dataset_dir)

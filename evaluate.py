# evaluate.py
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
from model.nafnet import compute_metrics
import argparse

def main(args):
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    preds = sorted(pred_dir.glob("*.tif"))
    rows = []
    for p in preds:
        gt_path = gt_dir / p.name
        if not gt_path.exists():
            print(f"GT not found for {p.name}, skipping")
            continue
        pred = io.imread(str(p)).astype(np.float32)
        gt = io.imread(str(gt_path)).astype(np.float32)
        # normalize to [0,1] if not already
        if pred.max() > 1.0:
            pred = pred / 65535.0
        if gt.max() > 1.0:
            gt = gt / 65535.0
        pval, sip, ssimv, mss = compute_metrics(gt, pred)
        rows.append({"image": p.name, "psnr": pval, "si_psnr": sip, "ssim": ssimv, "msssim": mss})
    df = pd.DataFrame(rows)
    df.to_csv("evaluation_metrics.csv", index=False)
    print(df.describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    args = parser.parse_args()
    main(args)

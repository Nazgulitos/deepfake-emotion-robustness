import pandas as pd, numpy as np, torch, yaml, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, 'scripts/exp15_modality_gated')
from utils import delong_test, bootstrap_auc_ci

# Parse NoDetector and AblationDataset from 07_ablation.py without running main()
src = open('scripts/exp15_modality_gated/07_ablation.py').read()
# Execute only up to the first top-level function definition after the classes
import re
# Split on 'def run_epoch' to get just the class definitions + imports
class_code = src.split('# ---------------------------------------------------------------------------\n# Single fold training')[0]
exec(class_code, globals())

pred    = Path('scripts/exp15_modality_gated/outputs/predictions')
abl_dir = Path('scripts/exp15_modality_gated/outputs/ablation')
te_df   = pd.read_parquet(pred / 'test_feature_matrix.parquet')
y_test  = te_df['label_int'].values
cfg     = yaml.safe_load(open('scripts/exp15_modality_gated/config.yaml'))
emo_cols  = cfg['emotion_feature_cols']
qual_cols = cfg['quality_feature_cols']
det_col   = 'detector_score'
all_feat  = [det_col] + emo_cols + qual_cols

nd_probs = []
for k in range(5):
    ckpt = torch.load(abl_dir/f'no_detector/fold_{k}/best.pt', map_location='cpu', weights_only=False)
    m = NoDetector(len(emo_cols), len(qual_cols), cfg['embed_dim'], cfg['dropout'])
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    ts = te_df.copy()
    ts[all_feat] = ckpt['scaler'].transform(te_df[all_feat])
    ds = AblationDataset(ts, det_col, emo_cols, qual_cols)
    ldr = DataLoader(ds, batch_size=64, shuffle=False)
    p = []
    with torch.no_grad():
        for d, e, q, _ in ldr:
            p.append(torch.sigmoid(m(d, e, q)['logit']).numpy())
    nd_probs.append(np.concatenate(p))

nd_ens     = np.stack(nd_probs).mean(0)
full_probs = pd.read_csv(pred / 'test_exp15_predictions.csv')['prediction'].values

auc_full = roc_auc_score(y_test, full_probs)
auc_nd   = roc_auc_score(y_test, nd_ens)
print(f'full AUC:        {auc_full:.4f}')
print(f'no_detector AUC: {auc_nd:.4f}')
print(f'delta:           {auc_full - auc_nd:+.4f}')

z, p_val = delong_test(y_test, full_probs, nd_ens)
print(f'DeLong z={z:.3f}  p={p_val:.4f}')

_, lo_f, hi_f = bootstrap_auc_ci(y_test, full_probs, n_iter=2000, seed=42)
_, lo_n, hi_n = bootstrap_auc_ci(y_test, nd_ens,     n_iter=2000, seed=42)
print(f'full CI:        [{lo_f:.4f}, {hi_f:.4f}]')
print(f'no_detector CI: [{lo_n:.4f}, {hi_n:.4f}]')
if lo_f < hi_n:
    print('CI overlap: YES  -> delta NOT statistically significant -> detector is redundant')
else:
    print('CI overlap: NO   -> statistically distinguishable')

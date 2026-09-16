"""Exclude CSI300 membership pairs absent from CSI800 snapshots (Section 5.5.3)."""
import argparse,json
from pathlib import Path
import numpy as np
from dynafuse.fusion import load,fuse,metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-features',type=Path,required=True,help='alpha158_raw/by_instrument directory with membership masks')
    p.add_argument('--master-predictions',type=Path,required=True);p.add_argument('--expert-predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reports/membership_sensitivity.json'))
    a=p.parse_args();m=load(a.master_predictions);e=load(a.expert_predictions)
    if not ((m['dates']>=20220104)&(m['dates']<=20251231)).all():raise ValueError('Expected test predictions only')
    excluded=set()
    for path in sorted(a.raw_features.glob('*.npz')):
        with np.load(path,allow_pickle=False) as raw:
            mask=raw['is_csi300'] & ~raw['is_csi800']
            excluded.update((int(day),path.stem) for day in raw['dates'][mask])
    if not excluded:raise ValueError('No membership differences found; check raw-feature input')
    keep=np.array([(int(d), i.decode("ascii") if isinstance(i,bytes) else str(i)) not in excluded for d,i in zip(m['dates'],m['instruments'])])
    if not (~keep).any():raise ValueError('No prediction keys matched excluded membership pairs')
    out={'all_period_mismatched_pairs':len(excluded),'test_excluded':int((~keep).sum()),'finite_target_test_excluded':int((~keep & np.isfinite(m['labels'])).sum()),'score_alignment':'full daily cross-section before exclusion','models':{}}
    for name,pred in [('MASTER',m),('DynaFuse',fuse(m,e))]:out['models'][name]={'before':metrics(pred),'after':metrics(pred,keep)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

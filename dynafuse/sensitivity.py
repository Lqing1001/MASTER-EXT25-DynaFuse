"""Validation-only Top-k and fusion-weight sensitivity (Table 9)."""
import argparse,csv,json
from pathlib import Path
from dynafuse.fusion import load,fuse,metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-validation',type=Path,required=True)
    p.add_argument('--top1-validation',type=Path,required=True)
    p.add_argument('--sweep-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('reports/sensitivity'))
    a=p.parse_args();m=load(a.master_validation);e=load(a.top1_validation)
    def validation_only(x):
        if not ((x['dates']>=20200102)&(x['dates']<=20211224)).all():raise ValueError('This sweep accepts validation predictions only')
    validation_only(m);rows=[]
    for k in (1,2,4,8):
        suffix='' if k==4 else f'_topk{k}'
        expert=e if k==1 else load(a.sweep_dir/f'ta_deformable{suffix}_csi300_seed0_validation_predictions.npz')
        validation_only(expert);v=metrics(fuse(m,expert));rows.append({'sweep':'topk','setting':k,**v,'V':(v['IC']+v['RankIC'])/2})
    for alpha in (0,.25,.5,.75,1):
        v=metrics(fuse(m,e,alpha));rows.append({'sweep':'alpha','setting':alpha,**v,'V':(v['IC']+v['RankIC'])/2})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'table9.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (a.output_dir/'table9.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()

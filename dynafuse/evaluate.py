"""Reproduce Tables 7-8 using fixed equal-weight fusion on complete daily cross sections."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json,csv,hashlib,argparse,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from dynafuse.statistics import hac_mean_test,moving_block_ci
ROOT=Path(__file__).resolve().parents[1]
METRICS=['IC','RankIC','ICIR','RankICIR'];MODELS=['MASTER','TOP1','DynaFuse','HOM','MX']
BOOTSTRAP_SEEDS=json.loads((ROOT/'configs/bootstrap_seeds.json').read_text())
def block_ci(x,seed):
 rng=np.random.default_rng(seed);n=len(x);means=np.empty(5000)
 for i in range(5000):
  starts=rng.choice(n-20+1,size=math.ceil(n/20),replace=True)
  ix=(starts[:,None]+np.arange(20)).ravel()[:n];means[i]=x[ix].mean()
 ci=np.quantile(means,[.025,.975]);return {'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'seed':seed,'generator':'numpy.default_rng','block_length':20,'draws':5000}
def summarize(universe, results_root, source_map=None):
 arrays={};sources=[]
 for seed in range(3):
  for model,folder,prefix in [('MASTER','master','master_full'),('TOP1','sparse_top1','ta_deformable_topk1'),('XGB','traditional_baselines','xgboost')]:
   p=Path(source_map[universe][model][str(seed)]) if source_map else results_root/universe/folder/f'{prefix}_{universe}_seed{seed}_predictions.npz'
   with np.load(p,allow_pickle=False) as data: arrays[model,seed]={k:data[k] for k in data.files}
   sources.append({'model':model,'seed':seed,'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 ref=arrays['MASTER',0]
 for a in arrays.values():
  assert np.array_equal(ref['dates'],a['dates']) and np.array_equal(ref['instruments'],a['instruments'])
  assert np.allclose(ref['labels'],a['labels'],equal_nan=True)
 order=np.argsort(ref['dates'],kind='stable');cuts=np.r_[0,np.flatnonzero(np.diff(ref['dates'][order]))+1,len(order)]
 daily={m:[[] for _ in range(3)] for m in MODELS};corr={'DynaFuse':[[] for _ in range(3)],'HOM':[[] for _ in range(3)]};dates=[]
 def z(x):return (x-x.mean())/max(x.std(),1e-12)
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  ix=order[lo:hi];y=ref['labels'][ix].astype(float);ok=np.isfinite(y);p={k:a['predictions'][ix].astype(float) for k,a in arrays.items()};assert all(np.isfinite(x).all() for x in p.values())
  for s in range(3):
   m,e,x=p['MASTER',s],p['TOP1',s],p['XGB',s];other=p['MASTER',(s+1)%3]
   corr['DynaFuse'][s].append(float(np.corrcoef(rankdata(m),rankdata(e))[0,1]));corr['HOM'][s].append(float(np.corrcoef(rankdata(m),rankdata(other))[0,1]))
   if ok.sum()<3:continue
   preds={'MASTER':m,'TOP1':e,'DynaFuse':.5*z(m)+.5*z(e),'HOM':.5*z(m)+.5*z(other),'MX':.5*z(m)+.5*z(x)}
   yr=rankdata(y[ok])
   for name,v in preds.items():daily[name][s].append([float(np.corrcoef(v[ok],y[ok])[0,1]),float(np.corrcoef(rankdata(v[ok]),yr)[0,1])])
  if ok.sum()>=3:dates.append(int(ref['dates'][ix[0]]))
 assert len(dates)==964
 daily={k:np.array(v) for k,v in daily.items()};byseed={k:np.concatenate([v.mean(1),v.mean(1)/v.std(1,ddof=0)],axis=1) for k,v in daily.items()}
 summary={m:{metric:{'mean':float(v[:,j].mean()),'sd':float(v[:,j].std(ddof=1)),'values':v[:,j].tolist()} for j,metric in enumerate(METRICS)} for m,v in byseed.items()}
 paired={}
 for control in ['MASTER','HOM','MX']:
  diffs=(daily['DynaFuse']-daily[control]).mean(0)
  for j,metric in enumerate(METRICS[:2]):
   delta=diffs[:,j];nw=hac_mean_test(delta,10)
   if control=='MX':ci=moving_block_ci(delta,np.random.RandomState(20260831),20,5000);ci.update(seed=20260831,generator='numpy.RandomState')
   else:
    ci=block_ci(delta,BOOTSTRAP_SEEDS[universe][f'DynaFuse_minus_{control}:{metric}'])
   paired[f'DynaFuse_minus_{control}:{metric}']={'NW':nw,'bootstrap':ci,'significant_positive':bool(nw['p_two_sided']<.05 and ci['ci95_low']>0),'seed_differences':(daily['DynaFuse']-daily[control]).mean(1)[:,j].tolist()}
 return {'summary':summary,'paired_tests':paired,'score_rank_correlation':{m:{'per_seed_median':np.median(v,axis=1).tolist(),'mean_of_seed_medians':float(np.median(v,axis=1).mean())} for m,v in corr.items()},'sources':sources,'n_dates':len(cuts)-1,'valid_dates':len(dates)},daily,dates

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--results-root',type=Path,default=Path('runs'))
 ap.add_argument('--source-map',type=Path,help='Optional JSON mapping universe/model/seed to prediction NPZ paths')
 ap.add_argument('--output-dir',type=Path,default=Path('reports'))
 args=ap.parse_args();dest=args.output_dir;dest.mkdir(parents=True,exist_ok=True)
 source_map=json.loads(args.source_map.read_text()) if args.source_map else None
 result={'protocol':{'seeds':[0,1,2],'fusion_weight':.5,'score_alignment':'full cross-section before finite-label filtering','HOM_pairs':[[0,1],[1,2],[2,0]],'NW_lag':10,'bootstrap_block':20,'bootstrap_draws':5000,'between_seed_sd_ddof':1,'daily_IR_ddof':0},'universes':{}}
 for u in ['csi300','csi800']:
  a,daily,dates=summarize(u,args.results_root,source_map);result['universes'][u]=a
  np.savez_compressed(dest/f'{u}_daily.npz',dates=dates,**daily)
  print(u,'done',flush=True)
 (dest/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 with (dest/'table7_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['universe','model',*[f'{m}_{s}' for m in METRICS for s in ['mean','sd']]])
  for u,a in result['universes'].items():
   for m in MODELS:w.writerow([u,m,*[a['summary'][m][k][s] for k in METRICS for s in ['mean','sd']]])
 with (dest/'table8_paired_tests.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['universe','contrast_metric','delta','NW_p','CI_low','CI_high'])
  for u,a in result['universes'].items():
   for k,v in a['paired_tests'].items():w.writerow([u,k,v['NW']['mean_difference'],v['NW']['p_two_sided'],v['bootstrap']['ci95_low'],v['bootstrap']['ci95_high']])
 with (dest/'per_seed_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['universe','seed','model',*METRICS])
  for u,a in result['universes'].items():
   for s in range(3):
    for m in MODELS:w.writerow([u,s,m,*[a['summary'][m][k]['values'][s] for k in METRICS]])

 print(dest,flush=True)
if __name__=='__main__':main()

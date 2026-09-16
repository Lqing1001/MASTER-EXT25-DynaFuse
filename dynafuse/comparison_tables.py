"""Export broad seed-0 comparisons (Tables 4-5) and component comparison (Table 6)."""
import argparse,csv,json
from pathlib import Path
from dynafuse.fusion import load,fuse,metrics

BASELINES=[('Ridge','traditional_baselines','ridge'),('Random Forest','traditional_baselines','random_forest'),('XGBoost','traditional_baselines','xgboost'),('StockMamba','neural_baselines','stockmamba'),('ACT','neural_baselines','act'),('PRISM-VQ','neural_baselines','prism_vq'),('MASTER','master','master_full')]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-root',type=Path,default=Path('runs'));p.add_argument('--paired-summary',type=Path,default=Path('reports/results.json'));p.add_argument('--baseline-map',type=Path);p.add_argument('--output-dir',type=Path,default=Path('reports'));p.add_argument('--master-validation',type=Path);p.add_argument('--expert-validation',type=Path);a=p.parse_args()
 summary=json.loads(a.paired_summary.read_text());mapping=json.loads(a.baseline_map.read_text()) if a.baseline_map else None;a.output_dir.mkdir(parents=True,exist_ok=True);rows=[]
 for u in ['csi300','csi800']:
  for display,folder,prefix in BASELINES:
   path=Path(mapping[u][prefix]) if mapping else a.results_root/u/folder/f'{prefix}_{u}_seed0.json'
   data=json.loads(path.read_text());v=data['metrics']['all_20220104_20251231']
   rows.append({'universe':u,'model':display,**{k:v[k] for k in ['IC','RankIC','ICIR','RankICIR']}})
  v=summary['universes'][u]['summary']['DynaFuse'];rows.append({'universe':u,'model':'DynaFuse',**{k:v[k]['values'][0] for k in ['IC','RankIC','ICIR','RankICIR']}})
 with (a.output_dir/'tables4_5.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 if bool(a.master_validation)!=bool(a.expert_validation):raise ValueError('Provide both validation prediction files')
 if a.master_validation:
  m=load(a.master_validation);e=load(a.expert_validation)
  if not ((m['dates']>=20200102)&(m['dates']<=20211224)).all():raise ValueError('Expected validation split')
  entries=[]
  for name,archive in [('MASTER',m),('TOP1',e),('DynaFuse',fuse(m,e))]:
   val=metrics(archive);test=summary['universes']['csi300']['summary'][name]
   entries.append({'model':name,'Val_IC':val['IC'],'Val_RankIC':val['RankIC'],'Test_IC':test['IC']['values'][0],'Test_RankIC':test['RankIC']['values'][0]})
  with (a.output_dir/'table6.csv').open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
 print(a.output_dir)
if __name__=='__main__':main()

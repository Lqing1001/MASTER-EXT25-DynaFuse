"""Run the final paper protocol serially. Existing output directories are not overwritten."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset-root',type=Path,required=True)
 p.add_argument('--output-root',type=Path,default=Path('runs'))
 p.add_argument('--universes',nargs='+',choices=['csi300','csi800'],default=['csi300','csi800'])
 p.add_argument('--seeds',nargs='+',type=int,choices=[0,1,2],default=[0,1,2])
 p.add_argument('--include-baselines',action='store_true',help='Train broad seed-0 neural/Ridge/RF baselines too')
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args(); commands=[]
 for u in a.universes:
  root=a.output_root/u
  for seed in a.seeds:
   common=['--universe',u,'--seed',str(seed)]
   for module,folder,extra in [
    ('dynafuse.train_master','master',['--epochs','40','--patience','40']),
    ('dynafuse.train_continuous','continuous_prism',['--stage1-epochs','4','--base-epochs','12','--adapter-epochs','10','--patience','12']),
    ('dynafuse.train_sparse','sparse_top1',['--epochs','10','--patience','10','--selected','1','--continuous-dir',str(root/'continuous_prism')]),
    ('baselines.xgboost','traditional_baselines',['--rounds','1200','--early-stopping','100'])]:
    commands.append([sys.executable,'-m',module,*common,*extra,'--output-dir',str(root/folder)])
  if a.include_baselines:
   for model in ['stockmamba','act','prism_vq']:
    commands.append([sys.executable,'-m','baselines.neural','--model',model,'--universe',u,'--seed','0','--epochs','12','--stage1-epochs','4','--patience','12','--output-dir',str(root/'neural_baselines')])
   for model in ['ridge','random_forest']:
    commands.append([sys.executable,'-m','baselines.traditional','--model',model,'--universe',u,'--seed','0','--output-dir',str(root/'traditional_baselines')])
 if a.dry_run:
  print(json.dumps(commands,indent=2));return
 manifest=json.loads((a.dataset_root/'manifest.json').read_text(encoding='utf-8'))
 if manifest.get('status')!='PASS' or manifest.get('temporal_audit',{}).get('training_feature_cutoff')!=20191224:
  raise ValueError('Expected a PASS strictly train-only dataset manifest with cutoff 20191224')
 if a.output_root.exists() and any(a.output_root.iterdir()):raise FileExistsError('Use a new empty output root; individual stage CLIs support deliberate reruns.')
 a.output_root.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env['MASTER_EXT_DATASET_ROOT']=str(a.dataset_root.resolve())
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
 (a.output_root/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
 for command in commands:
  print('Running:',subprocess.list2cmdline(command),flush=True)
  subprocess.run(command,env=env,check=True)
 print('Training complete. Run python -m dynafuse.evaluate --results-root',a.output_root)
if __name__=='__main__':main()

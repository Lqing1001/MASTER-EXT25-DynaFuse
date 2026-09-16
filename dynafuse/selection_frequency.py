"""Validation positional-selection frequencies for the final CSI300 seed-0 expert."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from dynafuse.runtime import UniverseStore,VALID_START,VALID_END
from dynafuse.infer import load_model

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('reports/selection_frequency.json'));a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);device='cuda:0' if torch.cuda.is_available() else 'cpu'
 model=load_model(a.run_dir,'csi300',0,'continuous',1,device);r=model.residual
 store=UniverseStore(a.dataset_root,'csi300');counts=np.zeros(8,dtype=np.int64)
 with torch.no_grad():
  for day in store.dates_between(VALID_START,VALID_END):
   x,_,_=store.batch(int(day),training=False);x=torch.from_numpy(x[:,:,:158]).to(device)
   scores=r.selector(r.input_proj(x)+r.position).squeeze(-1)
   selected=torch.topk(scores,k=1,dim=1).indices.cpu().numpy().ravel();counts+=np.bincount(selected,minlength=8)
 out={'universe':'csi300','seed':0,'split':'validation','position_order':'oldest to newest','counts':counts.tolist(),'percent':(100*counts/counts.sum()).tolist()}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

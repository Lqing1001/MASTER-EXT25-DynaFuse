"""Evaluate the published checkpoint layout on validation or test dates."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from dynafuse.runtime import UniverseStore,evaluate,metrics_by_period,write_daily_csv,VALID_START,VALID_END,TEST_START,TEST_END,summarize_daily
from dynafuse.train_master import MasterExpert
from dynafuse.train_continuous import ContinuousEncoder,MixtureRanker,TemporalAdapter
from dynafuse.train_sparse import SparseExpert

def load_residual(model,path,device):
    missing,unexpected=model.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=False)
    if unexpected or any(not k.startswith('base.') for k in missing):raise ValueError('Incompatible residual checkpoint: '+str(path))

def load_model(root,universe,seed,kind='master',selected=1,device='cpu'):
    if kind=='master':
        m=MasterExpert('full',universe).to(device)
        m.load_state_dict(torch.load(root/'master'/f'master_full_{universe}_seed{seed}_best.pt',weights_only=True,map_location=device));return m.eval()
    base=MixtureRanker(ContinuousEncoder('no_vq'),'no_vq').to(device)
    base.load_state_dict(torch.load(root/'continuous_prism'/f'no_vq_{universe}_seed{seed}_base_best.pt',weights_only=True,map_location=device))
    ta=TemporalAdapter(base).to(device)
    load_residual(ta,root/'continuous_prism'/f'no_vq_{universe}_seed{seed}_adapter_best.pt',device)
    sp=SparseExpert(ta,'deformable',selected=selected).to(device)
    suffix='' if selected==4 else f'_topk{selected}'
    folder='sparse_top1' if selected==1 else 'topk_validation_sweep'
    load_residual(sp,root/folder/f'ta_deformable{suffix}_{universe}_seed{seed}_best.pt',device)
    return sp.eval()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--universe',choices=['csi300','csi800'],required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--model',choices=['master','continuous'],default='master');p.add_argument('--selected',type=int,choices=[1,2,4,8],default=1)
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=load_model(a.run_dir,a.universe,a.seed,a.model,a.selected,device)
    store=UniverseStore(a.dataset_root,a.universe)
    start,end=(VALID_START,VALID_END) if a.split=='validation' else (TEST_START,TEST_END)
    rows,pred=evaluate(model,store,start,end,device,None)
    a.output_dir.mkdir(parents=True,exist_ok=True);prefix=f'{a.model}_{a.universe}_seed{a.seed}_{a.split}'
    np.savez_compressed(a.output_dir/f'{prefix}_predictions.npz',**pred)
    write_daily_csv(a.output_dir/f'{prefix}_daily.csv',rows)
    (a.output_dir/f'{prefix}.json').write_text(json.dumps(summarize_daily(rows),indent=2),encoding='utf-8')
if __name__=='__main__':main()

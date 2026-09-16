"""Count registered parameters and matrix FLOPs on the active inference path."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.nn.attention import sdpa_kernel,SDPBackend
from dynafuse.train_master import MasterExpert
from dynafuse.train_continuous import ContinuousEncoder,MixtureRanker,TemporalAdapter
from dynafuse.train_sparse import SparseExpert
class Count(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.counts = {}
        self.ops = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        y = func(*args, **kwargs or {})
        name = str(func)
        self.ops[name] = self.ops.get(name, 0) + 1
        c = 0
        if name in ('aten.mm.default', 'aten.bmm.default'):
            c = 2 * y.numel() * args[0].shape[-1]
        elif name == 'aten.addmm.default':
            c = 2 * y.numel() * args[1].shape[-1]
        elif name == 'aten.baddbmm.default':
            c = 2 * y.numel() * args[1].shape[-1]
        if c:
            self.counts[name] = self.counts.get(name, 0) + int(c)
        return y

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('reports/model_cost.json'));a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
 m=MasterExpert('full','csi300').eval();e=SparseExpert(TemporalAdapter(MixtureRanker(ContinuousEncoder('no_vq'),'no_vq')),'deformable',1).eval()
 out={'scope':'Matrix multiply-accumulate FLOPs, 2 per MAC. Excludes elementwise ops, bias adds, normalization, activation, softmax, sorting/TopK, score alignment and fusion. Includes GRU/linear/attention matrix products. Registered parameter counts include inactive checkpoint modules.','parameters':{'MASTER':sum(p.numel() for p in m.parameters()),'continuous':sum(p.numel() for p in e.parameters())},'counts':[]}
 for n in (300,800):
  row={'N':n}
  for name,model in [('MASTER',m),('continuous',e)]:
   torch.manual_seed(0);x=torch.randn(n,8,221);counter=Count()
   with torch.no_grad(),sdpa_kernel(SDPBackend.MATH),counter:y=model(x)
   if any('scaled_dot_product' in k or '_native_multi_head' in k or '_transformer_encoder' in k or 'gru' in k for k in counter.ops):raise RuntimeError('Uncounted fused operator encountered')
   row[name]=sum(counter.counts.values())
  row['DynaFuse_HOM_ratio']=(row['MASTER']+row['continuous'])/(2*row['MASTER']);out['counts'].append(row)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

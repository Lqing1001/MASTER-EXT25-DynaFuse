"""Numerical execution defaults for the single-thread paper protocol."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'

import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from dynafuse.fusion import zscore,fuse,metrics
from dynafuse.train_continuous import ContinuousEncoder,MixtureRanker,TemporalAdapter,stage1_loss,train_spatial,train_predictor,seed_all
from dynafuse.train_sparse import SparseExpert
from data_tools.rebuild_strict_train_only_packages import fit_stats

class ProtocolTests(unittest.TestCase):
 def test_full_cross_section_before_label_filter(self):
  m={'dates':np.ones(5,dtype=int)*20220104,'instruments':np.array(['a','b','c','d','e']),'labels':np.array([1.,2.,3.,4.,np.nan]),'predictions':np.array([1.,4.,2.,3.,100.])}
  e={**m,'predictions':np.array([3.,1.,2.,4.,-50.])}
  out=fuse(m,e)
  np.testing.assert_allclose(out['predictions'],.5*zscore(m['predictions'])+.5*zscore(e['predictions']))
  wrong=.5*zscore(m['predictions'][:4])+.5*zscore(e['predictions'][:4])
  self.assertFalse(np.allclose(out['predictions'][:4],wrong))
  bad={**e,'instruments':e['instruments'][::-1]}
  with self.assertRaises(ValueError):fuse(m,bad)
 def test_universe_specific_train_only_fit(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'raw.npz'
   np.savez(p,dates=[20190101,20190102,20200102],features=np.tile(np.array([1.,3.,9999.])[:,None],(1,158)),is_csi300=[True,False,True],is_csi800=[True,True,True])
   for u,expected in [('csi300',1.),('csi800',2.)]:
    out=Path(d)/(u+'.npz');report=fit_stats([p],'is_'+u,out)
    with np.load(out) as a:self.assertTrue(np.all(a['median']==expected))
    self.assertLessEqual(report['fit_date_max'],20191224)
 def test_sparse_top1_has_zero_selector_score_gradient(self):
  # Top-1 masked softmax has weight one. Preserve this published behavior.
  seed_all(0);sp=SparseExpert(TemporalAdapter(MixtureRanker(ContinuousEncoder('no_vq'),'no_vq')),'deformable',1)
  torch.nn.init.normal_(sp.residual.head[-1].weight)
  sp(torch.randn(12,8,221)).square().mean().backward()
  g=sp.residual.selector[-1].weight.grad
  self.assertIsNotNone(g);self.assertEqual(torch.count_nonzero(g).item(),0)
  self.assertTrue(all(p.grad is None for p in sp.base.parameters()))
 def test_four_stage_training_smoke(self):
  # Small in-memory cross sections test optimizer/checkpoint wiring, not paper performance.
  class Store:
   universe="synthetic"
   def dates_between(self,start,end):return np.array([20200102,20200103])
   def batch(self,day,training=False):
    rng=np.random.RandomState(int(day)%100000)
    return rng.normal(size=(24,8,221)).astype('float32'),rng.normal(size=24).astype('float32'),np.arange(24)
  store=Store();device=torch.device('cpu');dates=np.array([20190102,20190103]);seed_all(0)
  encoder=ContinuousEncoder('no_vq');opt=torch.optim.AdamW(encoder.parameters(),lr=1e-4)
  history=train_spatial(encoder,'no_vq',store,dates,opt,device,1,None)
  self.assertEqual(len(history),1)
  model=MixtureRanker(encoder,'no_vq')
  for label in ['ranker','TA','SP']:
   if label=='TA':model=TemporalAdapter(model)
   if label=='SP':model=SparseExpert(model,'deformable',1)
   opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
   history,epoch,state=train_predictor(model,label,store,dates,opt,device,1,1,0,None,None)
   self.assertEqual(epoch,0);self.assertTrue(state);self.assertTrue(np.isfinite(history[0]['selection_score']))

if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);unittest.main()

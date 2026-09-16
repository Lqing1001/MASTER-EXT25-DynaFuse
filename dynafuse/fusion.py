"""Fixed score alignment and daily evaluation; no fitted fusion parameters."""
import numpy as np
from scipy.stats import rankdata

def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def aligned(m,e):
    for key in ('dates','instruments'):
        if not np.array_equal(m[key],e[key]):raise ValueError('Mismatched '+key)
    if not np.allclose(m['labels'],e['labels'],equal_nan=True):raise ValueError('Mismatched labels')

def zscore(values):
    values=np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('Non-finite prediction')
    return (values-values.mean())/max(values.std(ddof=0),1e-12)

def fuse(m,e,alpha=0.5):
    aligned(m,e)
    if not 0<=alpha<=1:raise ValueError('alpha must be between zero and one')
    scores=np.empty(len(m['dates']),dtype=np.float64)
    for day in np.unique(m['dates']):
        mask=m['dates']==day
        scores[mask]=(1-alpha)*zscore(m['predictions'][mask])+alpha*zscore(e['predictions'][mask])
    return {**m,'predictions':scores}

def metrics(archive,keep=None):
    rows=[]
    for day in np.unique(archive['dates']):
        mask=archive['dates']==day
        if keep is not None:mask=mask & keep
        p=archive['predictions'][mask].astype(float);y=archive['labels'][mask].astype(float)
        ok=np.isfinite(p)&np.isfinite(y)
        if ok.sum()<3:continue
        rows.append([np.corrcoef(p[ok],y[ok])[0,1],np.corrcoef(rankdata(p[ok]),rankdata(y[ok]))[0,1]])
    v=np.asarray(rows);v=v[np.isfinite(v).all(1)]
    if not len(v):raise ValueError('No valid evaluation days')
    return dict(zip(('IC','RankIC','ICIR','RankICIR'),np.r_[v.mean(0),v.mean(0)/v.std(0)]))

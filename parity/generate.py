#!/usr/bin/env python3
"""Generate deterministic, provenance-rich umap-learn reference artifacts."""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, platform, sys
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from sklearn.utils import check_random_state
import umap
from umap.umap_ import nearest_neighbors, smooth_knn_dist, fuzzy_simplicial_set

REFERENCE_COMMIT = "cdb0eeb236f5d6759e0c3dd1fe132aef4db9e7b1"
SEED = 42

def datasets():
    r = np.random.default_rng(SEED)
    dense = r.normal(size=(32, 6)).astype(np.float32)
    clustered = np.vstack([r.normal(loc=v, scale=.18, size=(16, 4)) for v in (-2, 0, 2)]).astype(np.float32)
    noisy = np.column_stack((np.linspace(-2, 2, 48), np.sin(np.linspace(-2, 2, 48)*3), r.normal(scale=.7,size=(48,6)))).astype(np.float32)
    base = r.normal(size=(12,5)).astype(np.float32); duplicated = np.repeat(base, 3, axis=0)
    degenerate = np.zeros((24,4),np.float32); degenerate[:,0]=np.linspace(0,1,24); degenerate[::4,1]=1e-6
    sd = dense.copy(); sd[np.abs(sd)<.8]=0
    return [("dense","fast",dense),("sparse","full",sparse.csr_matrix(sd)),("clustered","full",clustered),("noisy","full",noisy),("duplicated","fast",duplicated),("degenerate","fast",degenerate)]

def matrix_json(x):
    if sparse.issparse(x):
        x=x.tocsr(); return {"rows":x.shape[0],"columns":x.shape[1],"values":x.data.tolist(),"indices":x.indices.tolist(),"indptr":x.indptr.tolist()}
    return {"rows":x.shape[0],"columns":x.shape[1],"data":x.reshape(-1).tolist()}

def dense_values(x): return (x.toarray() if sparse.issparse(x) else x).astype(np.float64).reshape(-1).tolist()

def generate(name,suite,x):
    n=x.shape[0]; k=min(10,n-1); components=2; epochs=100
    idx,dist,_=nearest_neighbors(x,k,"euclidean",{},False,SEED,False)
    sigmas,rhos=smooth_knn_dist(dist.astype(np.float32),float(k),local_connectivity=1.0)
    graph,_,_=fuzzy_simplicial_set(x,k,check_random_state(SEED),"euclidean",{},knn_indices=idx,knn_dists=dist,angular=False,set_op_mix_ratio=1.0,local_connectivity=1.0)
    coo=graph.tocoo(); order=np.lexsort((coo.col,coo.row))
    init=check_random_state(SEED).uniform(low=-10.0,high=10.0,size=(n,components)).astype(np.float32)
    model=umap.UMAP(n_neighbors=k,n_components=components,metric="euclidean",min_dist=.1,spread=1.0,n_epochs=epochs,init=init.copy(),random_state=SEED,transform_seed=SEED,n_jobs=1,force_approximation_algorithm=False).fit(x)
    dv=dense_values(x); checksum=hashlib.sha256(np.asarray(dv,dtype="<f4").tobytes()).hexdigest()
    packages={p:importlib.metadata.version(p) for p in ("umap-learn","numpy","scipy","scikit-learn","numba","pynndescent")}
    return {"schema_version":1,"name":name,"suite":suite,"provenance":{"generator":"parity/generate.py","reference_commit":REFERENCE_COMMIT,"python":platform.python_version(),"platform":platform.platform(),"packages":packages,"input_sha256":checksum,"seed":SEED},"parameters":{"n_neighbors":k,"n_components":components,"metric":"euclidean","min_dist":.1,"spread":1.0,"local_connectivity":1.0,"n_epochs":epochs},"input":matrix_json(x),"knn":{"indices":idx.reshape(-1).tolist(),"distances":dist.reshape(-1).tolist()},"smooth_knn":{"rho":rhos.tolist(),"sigma":sigmas.tolist()},"fuzzy_graph":[{"head":int(coo.row[i]),"tail":int(coo.col[i]),"weight":float(coo.data[i])} for i in order if coo.data[i]!=0],"initialization":init.reshape(-1).tolist(),"embedding":model.embedding_.reshape(-1).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("parity/fixtures"));p.add_argument("--suite",choices=("fast","full"),default="full");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    selected=[]
    for name,suite,x in datasets():
        if a.suite=="fast" and suite!="fast": continue
        artifact=generate(name,suite,x); path=a.output/f"{name}.json";path.write_text(json.dumps(artifact,indent=2,sort_keys=True)+"\n");selected.append({"name":name,"suite":suite,"file":path.name,"input_sha256":artifact["provenance"]["input_sha256"]})
    (a.output/"manifest.json").write_text(json.dumps({"schema_version":1,"reference_commit":REFERENCE_COMMIT,"fixtures":selected},indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()

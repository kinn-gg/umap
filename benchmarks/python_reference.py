#!/usr/bin/env python3
"""Machine-readable staged benchmarks for the pinned umap-learn reference."""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, statistics, sys, time, tracemalloc
from pathlib import Path
import importlib.metadata

import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.utils import check_random_state
import umap
from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors, smooth_knn_dist

try:
    import resource
except ImportError:  # Windows: the Go peakrss wrapper records runtime but not RSS.
    resource = None

SEED=42
REFERENCE_COMMIT="cdb0eeb236f5d6759e0c3dd1fe132aef4db9e7b1"

def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):return line.split(":",1)[1].strip()
        except OSError:pass
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER","")

def checksum(x):
    if sparse.issparse(x):
        x=x.tocsr(); raw=x.data.astype("<f4").tobytes()+x.indices.astype("<i8").tobytes()+x.indptr.astype("<i8").tobytes()
    else: raw=np.asarray(x,dtype="<f4").tobytes()
    return hashlib.sha256(raw).hexdigest()

def synthetic_cases():
    rng=np.random.default_rng(SEED);base=dict(rows=256,dimensions=16,neighbors=15,components=2,density=1.0,metric="euclidean")
    axes={"rows":[64,1024],"dimensions":[2,128],"neighbors":[5,50],"components":[2,8],"density":[.05,.5],"metric":["euclidean","cosine"]}
    for axis,values in axes.items():
        for value in values:
            p=base.copy();p[axis]=value;x=rng.normal(size=(p["rows"],p["dimensions"])).astype(np.float32)
            if p["density"]<1:
                x[rng.random(x.shape)>p["density"]]=0;x=sparse.csr_matrix(x)
            yield f"synthetic/{axis}/{value}",x,p

def public_cases(suite,cache):
    for name,bunch in (("iris",load_iris()),("digits",load_digits())):
        x=np.asarray(bunch.data,dtype=np.float32);yield f"public/{name}",x,dict(rows=len(x),dimensions=x.shape[1],neighbors=15,components=2,density=1.,metric="euclidean")
    rng=np.random.default_rng(SEED);x=sparse.random(1000,5000,density=.01,format="csr",random_state=rng,dtype=np.float32);yield "synthetic/sparse-text",x,dict(rows=1000,dimensions=5000,neighbors=15,components=2,density=.01,metric="cosine")
    if suite!="full":return
    for label,dataset in (("mnist","mnist_784"),("fashion-mnist","Fashion-MNIST")):
        b=fetch_openml(dataset,version=1,as_frame=False,data_home=cache,parser="auto")
        idx=np.random.default_rng(SEED).choice(len(b.data),2000,replace=False);x=np.asarray(b.data[idx],dtype=np.float32);yield f"public/{label}-2000",x,dict(rows=len(x),dimensions=x.shape[1],neighbors=15,components=2,density=1.,metric="euclidean")

def rss_bytes():
    if resource is None:return None
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=="darwin" else value*1024)

def run_once(x,p):
    timings={};start=time.perf_counter_ns();idx,dist,_=nearest_neighbors(x,p["neighbors"],p["metric"],{},False,SEED,False);timings["knn"]=time.perf_counter_ns()-start
    start=time.perf_counter_ns();sigmas,rhos=smooth_knn_dist(dist.astype(np.float32),float(p["neighbors"]),local_connectivity=1.);timings["smooth_knn"]=time.perf_counter_ns()-start
    start=time.perf_counter_ns();fuzzy_simplicial_set(x,p["neighbors"],check_random_state(SEED),p["metric"],{},knn_indices=idx,knn_dists=dist,set_op_mix_ratio=1.,local_connectivity=1.);timings["fuzzy_graph"]=time.perf_counter_ns()-start
    init=check_random_state(SEED).uniform(-10,10,size=(x.shape[0],p["components"])).astype(np.float32)
    start=time.perf_counter_ns();umap.UMAP(n_neighbors=p["neighbors"],n_components=p["components"],metric=p["metric"],n_epochs=100,init=init,random_state=SEED,n_jobs=1).fit(x);timings["fit_end_to_end"]=time.perf_counter_ns()-start
    timings["total"]=timings["fit_end_to_end"];return timings

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--suite",choices=("fast","full"),default="fast");ap.add_argument("--case",action="append",help="run only this exact case name; repeatable");ap.add_argument("--repeats",type=int,default=5);ap.add_argument("--warmups",type=int,default=1);ap.add_argument("--output",type=Path);ap.add_argument("--prepare-public",action="store_true");ap.add_argument("--cache",type=Path,default=Path("benchmarks/.data"));args=ap.parse_args()
    if args.prepare_public:
        for name,x,_ in public_cases("full",args.cache):
            if name.startswith("public/"):print(json.dumps({"name":name,"input_sha256":checksum(x)},sort_keys=True))
        return
    out=args.output.open("w") if args.output else sys.stdout;packages={p:importlib.metadata.version(p) for p in ("umap-learn","numpy","scipy","scikit-learn","numba","pynndescent")}
    metadata={"schema_version":1,"type":"metadata","platform":platform.platform(),"architecture":platform.machine(),"processor":cpu_model(),"cpu_count":os.cpu_count(),"python":platform.python_version(),"packages":packages,"reference_commit":REFERENCE_COMMIT,"seed":SEED,"warmups":args.warmups,"repeats":args.repeats,"warmup_policy":"untimed before measured repeats"};print(json.dumps(metadata,sort_keys=True),file=out,flush=True)
    for name,x,p in list(synthetic_cases())+list(public_cases(args.suite,args.cache)):
        if args.case and name not in args.case:continue
        warmup_ns=[]
        for _ in range(args.warmups):
            start=time.perf_counter_ns();run_once(x,p);warmup_ns.append(time.perf_counter_ns()-start)
        runs=[run_once(x,p) for _ in range(args.repeats)]
        gc.collect();tracemalloc.start();allocation_timings=run_once(x,p);_,allocated_peak=tracemalloc.get_traced_memory();gc.collect();retained,_=tracemalloc.get_traced_memory();tracemalloc.stop()
        totals=[r["total"] for r in runs];record={"schema_version":1,"type":"case","name":name,"parameters":p,"input_sha256":checksum(x),"warmup_ns":warmup_ns,"runs":runs,"total_mean_ns":statistics.mean(totals),"total_stdev_ns":statistics.stdev(totals) if len(totals)>1 else 0,"allocation_probe":{"timings":allocation_timings,"python_allocated_peak_bytes":allocated_peak,"python_retained_bytes":retained},"peak_rss_bytes":rss_bytes()};print(json.dumps(record,sort_keys=True),file=out,flush=True)
    if out is not sys.stdout:out.close()
if __name__=="__main__":main()

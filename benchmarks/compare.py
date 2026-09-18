#!/usr/bin/env python3
"""Validate and summarize matched Go and Python UMAP benchmark JSONL."""
import argparse,json,statistics
from pathlib import Path

def load(path):
    records=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return records[0],{r["name"]:r for r in records[1:] if r.get("type")=="case"}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--go",required=True);ap.add_argument("--python",required=True);ap.add_argument("--output",required=True);args=ap.parse_args()
    go_meta,go=load(args.go);py_meta,py=load(args.python);names=sorted(set(go)&set(py))
    if not names:raise SystemExit("no matched benchmark cases")
    rows=[]
    for name in names:
        g,p=go[name],py[name]
        if g["parameters"]!=p["parameters"]:raise SystemExit(f"parameter mismatch for {name}")
        if g["input_sha256"]!=p["input_sha256"]:raise SystemExit(f"input checksum mismatch for {name}")
        go_ns=statistics.median(r["total"] for r in g["runs"]);py_ns=statistics.median(r["total"] for r in p["runs"])
        rows.append({"name":name,"go_median_ns":go_ns,"python_median_ns":py_ns,"python_over_go":py_ns/go_ns,"go_allocated_bytes":g["allocation_probe"]["go_allocated_bytes"],"go_retained_heap_bytes":g["allocation_probe"]["go_retained_heap_bytes"],"python_tracemalloc_peak_bytes":p["allocation_probe"]["python_allocated_peak_bytes"],"python_retained_bytes":p["allocation_probe"]["python_retained_bytes"],"go_peak_rss_bytes":g.get("peak_rss_bytes"),"python_peak_rss_bytes":p.get("peak_rss_bytes")})
    result={"schema_version":1,"go_metadata":go_meta,"python_metadata":py_meta,"cases":rows}
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("| Case | Go median | Python median | Python / Go | Go allocated | Python traced peak | Go RSS | Python RSS |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        go_rss="n/a" if r["go_peak_rss_bytes"] is None else f'{r["go_peak_rss_bytes"]/1048576:.1f} MiB'
        py_rss="n/a" if r["python_peak_rss_bytes"] is None else f'{r["python_peak_rss_bytes"]/1048576:.1f} MiB'
        print(f'| {r["name"]} | {r["go_median_ns"]/1e6:.2f} ms | {r["python_median_ns"]/1e6:.2f} ms | {r["python_over_go"]:.2f}x | {r["go_allocated_bytes"]/1048576:.2f} MiB | {r["python_tracemalloc_peak_bytes"]/1048576:.2f} MiB | {go_rss} | {py_rss} |')

if __name__=="__main__":main()

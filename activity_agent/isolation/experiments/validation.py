"""Saved-result validation, bounded memory benchmark, and optional offline browser checks."""
from __future__ import annotations

import argparse
import copy
import base64
import json
import pickle
import socket
import subprocess
import sys
import time
import tracemalloc
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from activity_agent.common.storage import write_json, fingerprint
from .features import regional_frame
from .runner import _status, consolidate, _temporal, _injections
from .report import render_report, PAGES
from .paths import model_directory


def memory_benchmark(run):
    features = pd.read_parquet(run / "data/features.parquet")
    future = features.loc[features.common_eligible & features.month.isin(["202511","202512"])]
    rows = []
    for path in sorted(model_directory(run).glob("*.pkl")):
        if path.stem in ("operating_original", "IF_BASELINE_ORIGINAL"):
            continue
        with path.open("rb") as stream:
            model = pickle.load(stream)
        data = regional_frame(future) if model.regional else future
        sample = data.sample(n=min(512,len(data)),random_state=42)
        tracemalloc.start()
        start=time.perf_counter()
        model.predict(sample)
        elapsed=time.perf_counter()-start
        current,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"model_id":path.stem,"query_rows":len(sample),"python_traced_peak_bytes":peak,
                     "python_traced_current_bytes":current,"benchmark_inference_seconds":elapsed,
                     "scope":"동일 프로세스 512행 추론 중 Python 추적 할당량; 네이티브 전체/RSS/학습 메모리 아님"})
        meta=model_directory(run)/f"{path.stem}.json"
        content=json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        if hasattr(model,"scaler"):
            for record in model.scaler.diagnostics:
                record["fallback"] = bool(record["training_rows"] < 8 or ((model.scaler.mode!="N0" or model.scaler.regional) and record["iqr"]<.001))
                record["floor_applied"] = bool(model.scaler.mode=="N0" and not model.scaler.regional and record["iqr"]<.001)
            pd.DataFrame(model.scaler.diagnostics).assign(model_id=path.stem).to_csv(run/f"data/normalization_{path.stem}.csv",index=False)
            content["fitted_normalization"]={"keys":model.scaler.keys,
                "global_median":model.scaler.global_median.to_dict(),"global_scale":model.scaler.global_scale.to_dict(),
                "profiles":[{"key":list(key),"median":med.to_dict(),"scale":scale.to_dict()}
                            for key,(med,scale) in model.scaler.profiles.items()]}
        if hasattr(model,"region"):
            content["regional_stage_features"]=model.region.columns
            content["regional_stage_normalization_profiles"]=[{"key":list(key),"median":med.to_dict(),"scale":scale.to_dict()}
                for key,(med,scale) in model.region.scaler.profiles.items()]
            content["dual_weight"]=model.weight
        write_json(meta,content)
    pd.DataFrame(rows).to_csv(run/"data/memory_metrics.csv",index=False,encoding="utf-8-sig")
    _status(run,"MEMORY_PROFILING","COMPLETE",reason="고정 512행 추론의 Python 추적 할당량 측정; 전체 프로세스/네이티브·학습 peak는 미측정",models=len(rows))
    snapshot=run/"config/code_snapshot";snapshot.mkdir(exist_ok=True)
    hashes={}
    for source in Path(__file__).parent.glob("*.py"):
        (snapshot/source.name).write_bytes(source.read_bytes())
        hashes[source.name]=fingerprint(source)
    manifest_path=run/"run_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(experiment_code_sha256=hashes,code_snapshot="config/code_snapshot",git_worktree_changes_included=True)
    write_json(manifest_path,manifest)
    consolidate(run)
    return rows


def finalize_saved(run, measure_memory=False):
    """Derive diagnostics from saved scores; no operational writes or model refits."""
    run=Path(run)
    config_path=run/"config/experiment_config.json"
    scenario_cfg=json.loads(config_path.read_text(encoding="utf-8"))
    if scenario_cfg.get("scenario_version",1)<2:
        for path in sorted(model_directory(run).glob("*.pkl")):
            identity=path.stem
            anchors=run/f"data/injection_anchors_{identity}.parquet"
            if identity in ("operating_original", "IF_BASELINE_ORIGINAL") or not anchors.exists():continue
            with path.open("rb") as stream:model=pickle.load(stream)
            _injections(run,identity,model,pd.read_parquet(anchors),only_scenarios={"S4","S5"})
        scenario_cfg["scenario_version"]=2
        write_json(config_path,scenario_cfg)
        _status(run,"INJECTION_TRAJECTORY_VALIDATION","COMPLETE",reason="S4 실제 두 연속 월간 감소, S5 직전 비교 0·현재 일시 감소; 초기 임시 결과는 data/archive에 보존")
    if measure_memory:
        memory_benchmark(run)
    for path in sorted((run/"data").glob("predictions_*.parquet")):
        predicted=pd.read_parquet(path)
        _temporal(run,str(predicted.model_id.iloc[0]),predicted)
    for path in sorted((run/"data").glob("injection_pairs_*.parquet")):
        identity=path.stem.removeprefix("injection_pairs_")
        metric_path=run/f"data/injection_{identity}.csv"
        metrics=pd.read_csv(metric_path,dtype={"geo_code":str,"month":str})
        pairs=pd.read_parquet(path)
        for scenario,sequence in pairs.groupby("scenario"):
            previous=None
            for strength,current in sequence.groupby("strength",sort=True):
                values=current.set_index("row_id").injected_score
                chosen=metrics.scenario.eq(scenario)&np.isclose(metrics.strength,strength,equal_nan=False)
                for i,row in metrics.loc[chosen].iterrows():
                    subset=current
                    if row.geo_code!="ALL":
                        subset=subset.loc[subset.geo_code.eq(row.geo_code)&subset.month.eq(row.month)]
                    score=subset.set_index("row_id").injected_score
                    if previous is None:
                        value=np.nan
                    else:
                        prior=previous.reindex(score.index)
                        valid=np.isfinite(score)&np.isfinite(prior)
                        value=float((score[valid]>=prior[valid]-1e-12).mean()) if valid.any() else np.nan
                    metrics.loc[i,"severity_nondecreasing_fraction"]=value
                    inputs=[c for c in subset if c.startswith("original_") and c.removeprefix("original_") not in ("score","percentile","candidate")]
                    inputs=[c for c in inputs if "injected_"+c.removeprefix("original_") in subset]
                    if inputs:
                        modified=subset[["injected_"+c.removeprefix("original_") for c in inputs]].to_numpy()
                        base=subset[inputs].to_numpy()
                        metrics.loc[i,"model_input_changed_rate"]=float((~np.isclose(base,modified,equal_nan=True)).any(axis=1).mean())
                previous=values
        metrics.to_csv(metric_path,index=False)
    dual_path=model_directory(run)/"DUAL_IF.pkl"
    if dual_path.exists():
        with dual_path.open("rb") as stream:
            model=pickle.load(stream)
        features=pd.read_parquet(run/"data/features.parquet")
        split=json.loads((run/"config/data_split.json").read_text(encoding="utf-8"))
        common=features.loc[features.common_eligible]
        calibration=common.loc[common.month.eq("202510")&common.block_code.isin(split["calibration_blocks"])]
        evaluation=common.loc[common.month.isin(["202511","202512"])]
        rows=[]
        for weight in (.25,.5,.75):
            alternative=copy.copy(model);alternative.weight=weight
            alternative.reference=np.sort(alternative.raw_score(calibration))
            alternative.threshold=float(np.quantile(alternative.reference,.95))
            prediction=alternative.predict(evaluation)
            for (month,code),part in prediction.groupby(["month","geo_code"]):
                rows.append({"group_weight":weight,"regional_weight":1-weight,"month":month,"geo_code":code,
                             "candidate_rows":int(part.candidate.sum()),"candidate_blocks":int(part.loc[part.candidate,"block_code"].nunique()),
                             "candidate_rate":float(part.candidate.mean()),"calibration_threshold":alternative.threshold})
        pd.DataFrame(rows).to_csv(run/"data/dual_weight_sensitivity.csv",index=False,encoding="utf-8-sig")
        _status(run,"DUAL_WEIGHT_SENSITIVITY","COMPLETE",reason="0.25/0.5/0.75 고정 가중치; 동일 보정 집합에서 각각 95% 경계, 평가로 가중치 선택하지 않음")
    consolidate(run)
    config_path=run/"config/experiment_config.json"
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    cfg.update(dual_group_weights=[.25,.5,.75],scenario_version=2,
               normalization_modes={"N0":"시도×성×연령 median/IQR 및 기존 0.001 floor",
                                    "N1":"시도별 pooled median/IQR, 통합 IF",
                                    "N2":"시도별 pooled median/IQR, 시도별 IF; 보정 원점수 공통 95% 경계",
                                    "N3":"성×연령 pooled median/IQR, 통합 IF"},
               intervention_semantics="S4 g1=(1-d)*g2, g0=(1-d)*g1; 직전 창에도 같은 직전월 배율 적용. S5 직전 비교를 반사실적으로 0으로 두고 현재 비교 -d. 실제 정상 라벨 아님. S2 d=0.4이면 전체 시간 유동 -10%, 낮 -40%. S6 배제 분모는 실제 증가한 변형 쌍만 사용")
    write_json(config_path,cfg)
    freeze_code(run)


def freeze_code(run):
    run=Path(run)
    project=Path(__file__).resolve().parents[3]
    root=run/"config/code_snapshot_v2"
    sources=list((project/"activity_agent").rglob("*.py"))
    sources += [project/"pyproject.toml",project/"requirements.lock.txt",project/"config/default.json"]
    hashes={}
    for source in sources:
        relative=source.relative_to(project)
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[relative.as_posix()]=fingerprint(source)
    path=run/"run_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    manifest.update(code_snapshot="config/code_snapshot_v2",experiment_code_sha256=hashes,
                    snapshot_scope="activity_agent Python source, pyproject, dependency lock, default config; project-relative layout",
                    git_worktree_changes_included=True)
    write_json(path,manifest)


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.external_scripts=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag=="script" and attrs.get("src"):
            self.external_scripts.append(attrs["src"])


def validate_static(run):
    checks=[]
    for page in PAGES:
        path=run/f"{page}.html"
        parser=Links();parser.feed(path.read_text(encoding="utf-8"))
        missing=[link for link in parser.links if not link.startswith(("http:","https:","#","data:")) and not (run/link).exists()]
        if missing or parser.external_scripts:
            raise ValueError(f"{page}: missing={missing}, external_scripts={parser.external_scripts}")
        checks.append({"page":page,"status":"COMPLETE","local_links":len(parser.links),"external_scripts":0})
    guard=json.loads((run/"config/production_hashes.json").read_text(encoding="utf-8"))
    from activity_agent.isolation.anomaly import relocated_model_path
    changed=[path for path,sha in guard.items() if fingerprint(relocated_model_path(path))!=sha]
    if changed:
        raise ValueError(f"Production artifacts changed: {changed}")
    split=json.loads((run/"config/data_split.json").read_text(encoding="utf-8"))
    if set(split["train_blocks"]) & set(split["calibration_blocks"]):
        raise ValueError("Block leakage")
    result={"pages":checks,"production_hashes_unchanged":True,"block_split_disjoint":True}
    write_json(run/"data/static_validation.json",result)
    return result


def browser_validate(run):
    """Use installed Chrome headless via CDP; HTTP(S) page resources are blocked."""
    from websockets.sync.client import connect

    choices=[Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")]
    browser=next((p for p in choices if p.exists()),None)
    if browser is None:
        return {"status":"SKIPPED","reason":"No installed Chrome/Edge headless browser"}
    directory=run/"validation";directory.mkdir(exist_ok=True)
    downloads=directory/"downloads";downloads.mkdir(exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",0));port=probe.getsockname()[1]
    args=[str(browser),"--headless=new","--disable-gpu","--no-first-run","--no-default-browser-check",
          "--disable-background-networking","--disable-component-update","--disable-sync","--metrics-recording-only",
          "--allow-file-access-from-files",f"--remote-debugging-port={port}",f"--user-data-dir={directory/'browser-profile'}","about:blank"]
    process=subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                             creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        targets=None
        for _ in range(80):
            try:
                targets=json.loads(opener.open(f"http://127.0.0.1:{port}/json",timeout=1).read())
                if targets:break
            except OSError:
                time.sleep(.25)
        if not targets:
            raise RuntimeError("Headless browser startup failed")
        websocket=next(t["webSocketDebuggerUrl"] for t in targets if t.get("type")=="page")
        errors=[];events=[];counter=0
        with connect(websocket,proxy=None,max_size=64*1024*1024,open_timeout=15) as ws:
            def call(method,params=None):
                nonlocal counter
                counter+=1;identity=counter
                ws.send(json.dumps({"id":identity,"method":method,"params":params or {}}))
                while True:
                    message=json.loads(ws.recv(timeout=80))
                    if message.get("method")=="Runtime.exceptionThrown":
                        errors.append(message["params"])
                    if "method" in message:events.append(message)
                    if message.get("id")==identity:
                        if "error" in message:raise RuntimeError(message["error"])
                        return message.get("result",{})
            def evaluate(expression):
                answer=call("Runtime.evaluate",{"expression":expression,"returnByValue":True,"awaitPromise":True})
                if answer.get("exceptionDetails"):
                    raise RuntimeError(answer["exceptionDetails"])
                return answer.get("result",{}).get("value")
            call("Runtime.enable");call("Page.enable");call("Network.enable")
            call("Network.setBlockedURLs",{"urls":["http://*","https://*"]})
            call("Page.setDownloadBehavior",{"behavior":"allow","downloadPath":str(downloads)})
            results=[]
            for page in PAGES:
                call("Page.navigate",{"url":(run/f"{page}.html").as_uri()})
                for _ in range(100):
                    ready=evaluate("document.readyState==='complete' && document.querySelector('main')!==null")
                    if ready:break
                    time.sleep(.15)
                if not ready:
                    raise RuntimeError(f"Page did not finish loading: {page}")
                if errors:
                    raise RuntimeError(f"Browser script error on {page}: {errors[-1]}")
                results.append({"page":page,"status":"COMPLETE","plots":evaluate("document.querySelectorAll('.js-plotly-plot').length")})
                if page=="index":
                    png=call("Page.captureScreenshot",{"format":"png","captureBeyondViewport":False})["data"]
                    (directory/"overview.png").write_bytes(base64.b64decode(png))
                if page=="model_comparison":
                    evaluate("document.getElementById('modelFilter').value='IF_F4';document.getElementById('regionFilter').value='32';document.getElementById('monthFilter').value='202512';drawComparison()")
                    table=evaluate("Array.from(document.querySelector('#filteredTable table').rows).map(r=>Array.from(r.cells).map(c=>c.textContent))")
                    source=pd.read_csv(run/"data/regional_metrics.csv",dtype={"geo_code":str,"month":str})
                    expected=source.loc[source.model_id.eq("IF_F4") & source.geo_code.eq("32") & source.month.eq("202512")].iloc[0]
                    if len(table)!=2 or int(table[1][3])!=int(expected.candidate_rows):
                        raise RuntimeError(f"Comparison filter mismatch: {table}")
                if page=="candidate_explorer":
                    evaluate("document.getElementById('candidateModel').value='IF_F4';loadCandidateModel()")
                    predicted=pd.read_parquet(run/"data/predictions_IF_F4.parquet")
                    target=predicted.loc[predicted.candidate].iloc[0]
                    values={"candidateMonth":str(target.month),"candidateRegion":str(target.geo_code),"candidateSex":str(target.sex),"candidateAge":str(target.age_band),"candidateSearch":str(target.block_code)}
                    script=';'.join(f"document.getElementById({json.dumps(k)}).value={json.dumps(v)}" for k,v in values.items())
                    evaluate(script+";filterCandidates()")
                    actual=evaluate("filteredCandidates.length")
                    expected=predicted.loc[predicted.candidate & predicted.month.eq(target.month) & predicted.geo_code.eq(target.geo_code)
                        & predicted.sex.eq(target.sex) & predicted.age_band.eq(target.age_band) & predicted.block_code.str.contains(target.block_code,regex=False)]
                    if actual!=len(expected):raise RuntimeError(f"Candidate filter mismatch {actual} != {len(expected)}")
                    downloaded=downloads/"filtered_candidates.csv"
                    if downloaded.exists():downloaded.unlink()
                    evaluate("document.getElementById('exportCandidates').click()")
                    for _ in range(40):
                        downloaded=downloads/"filtered_candidates.csv"
                        if downloaded.exists():break
                        time.sleep(.2)
                    csv=pd.read_csv(downloaded,encoding="utf-8-sig")
                    if len(csv)!=actual:raise RuntimeError("Downloaded CSV differs from displayed filter")
                    png=call("Page.captureScreenshot",{"format":"png","captureBeyondViewport":False})["data"]
                    (directory/"candidate_explorer.png").write_bytes(base64.b64decode(png))
            external=[e for e in events if e.get("method")=="Network.requestWillBeSent" and e["params"]["request"]["url"].startswith(("http:","https:"))]
            result={"status":"COMPLETE","pages":results,"runtime_errors":errors,"http_requests":len(external),
                    "candidate_filters_checked":True,"comparison_filter_csv_equality":True,"csv_download_filter_equality":True,
                    "network_http_https_blocked":True}
            write_json(run/"data/browser_validation.json",result)
            return result
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill()


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser();parser.add_argument("--run-dir",required=True)
    parser.add_argument("--benchmark",action="store_true");parser.add_argument("--browser",action="store_true")
    parser.add_argument("--finalize",action="store_true")
    args=parser.parse_args();run=Path(args.run_dir).resolve()
    if args.finalize:
        finalize_saved(run,measure_memory=args.benchmark);render_report(run)
    elif args.benchmark:
        memory_benchmark(run);render_report(run)
    print(json.dumps(validate_static(run),ensure_ascii=False))
    if args.browser:
        print(json.dumps(browser_validate(run),ensure_ascii=False))


if __name__=="__main__":main()

"""Standalone Korean HTML pages with inline Plotly and offline candidate filtering."""
from __future__ import annotations

import base64
import gzip
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from activity_agent.common.storage import records, write_json
from .features import FEATURE_SETS, DEFINITIONS
from .paths import model_directory

PAGES = {"index": "실험 개요", "dataset_diagnostics": "데이터·특징 진단", "model_comparison": "모델 비교",
         "injection_experiments": "합성 이상 주입", "ablation_study": "특징량 Ablation", "regional_temporal": "지역·시간 분석",
         "candidate_explorer": "후보 탐색", "methodology": "방법·한계"}
STYLE = """
body{margin:0;background:#f3f6fa;color:#18283c;font:15px system-ui,'Malgun Gothic',sans-serif}header{background:#122b45;color:white;padding:28px 4%}h1{margin:4px 0;font-size:29px}header p{color:#cee0ef}nav{display:flex;flex-wrap:wrap;gap:8px;padding:14px 4%;background:white;position:sticky;top:0;z-index:5;box-shadow:0 2px 8px #0001}nav a{padding:7px 11px;text-decoration:none;border-radius:5px;color:#245273;background:#edf4f9}nav a.current{background:#1976ad;color:white}main{max-width:1460px;margin:auto;padding:22px}section,.card{background:white;border-radius:10px;padding:20px;margin:0 0 20px;box-shadow:0 2px 9px #1020300b}h2{font-size:21px}p,li{line-height:1.75}.notice{border-left:5px solid #efac45;background:#fff8e9;padding:12px 18px;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}.big{font-size:28px;font-weight:700}label{display:inline-block;margin:8px}select,input,button{padding:8px;border:1px solid #bfcddb;border-radius:5px;background:white;color:#18283c}button{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #dce4ed;padding:8px;text-align:left;vertical-align:top}th{background:#eaf1f8;position:sticky;top:0}td{max-width:400px;overflow-wrap:anywhere}.scroll{overflow:auto;max-height:570px}.tag{border-radius:4px;padding:3px 7px;background:#e6edf4}.COMPLETE{color:#14663d}.FAILED{color:#b42323}.SKIPPED{color:#886200}.muted{color:#556777}.download{display:inline-block;padding:8px 11px;margin:5px;background:#e5f2fa;border-radius:5px;color:#155578;text-decoration:none}footer{color:#627589;padding:25px;text-align:center}.plot{min-height:360px}@media(max-width:700px){main{padding:10px}header{padding:20px}nav{position:relative}h1{font-size:23px}}
"""


def read(run, name):
    path = run / f"data/{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"geo_code": str, "month": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def table(frame, limit=300):
    if frame.empty:
        return '<p class="muted">표시할 결과가 없습니다. 미실행·보류 상태를 확인하세요.</p>'
    return f'<p class="muted">{len(frame):,}행 중 최대 {limit}행 표시. 전체 수치는 연결된 CSV에 있습니다.</p><div class="scroll">' + frame.head(limit).to_html(index=False, escape=True, float_format=lambda x: f"{x:.5g}") + '</div>'


def plot(figure):
    return figure.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})


def section(title, content):
    return f'<section><h2>{html.escape(title)}</h2>{content}</section>'


def download(name, label=None):
    return f'<a class="download" href="data/{html.escape(name)}" download>{html.escape(label or name)}</a>'


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")


def selector(identity, values, title):
    options = '<option value="ALL">전체</option>' + ''.join(f'<option value="{html.escape(str(v))}">{html.escape(str(v))}</option>' for v in values)
    return f'<label>{title} <select id="{identity}">{options}</select></label>'


def _interactive_comparison(regional, injections):
    models = sorted(regional.model_id.unique()) if not regional.empty else []
    controls = selector("modelFilter", models, "모델") + selector("regionFilter", ["11", "32"], "지역 코드 (11 서울·32 강원)") + selector("monthFilter", ["202511", "202512"], "관측 월")
    rows = records(regional) if not regional.empty else []
    injected = records(injections) if not injections.empty else []
    return controls + '<div id="regionalChart" class="plot"></div><div id="injectionChart" class="plot"></div><div id="filteredTable" class="scroll"></div>' + f"""
<script>
const regionalData={dump(rows)}, injectedData={dump(injected)};
function selected(row){{return ['model','region','month'].every((n,i)=>{{let v=document.getElementById(n+'Filter').value;return v==='ALL'||String(row[['model_id','geo_code','month'][i]])===v;}})}}
function drawComparison(){{
 let rows=regionalData.filter(selected);let traces=[];
 for(let key of [...new Set(rows.map(r=>r.geo_code+' / '+r.month))]){{let p=rows.filter(r=>r.geo_code+' / '+r.month===key);traces.push({{type:'bar',x:p.map(r=>r.model_id),y:p.map(r=>100*r.candidate_rate),name:key,hovertemplate:'%{{x}}<br>%{{y:.2f}}%<extra>%{{fullData.name}}</extra>'}})}}
 Plotly.react('regionalChart',traces,{{title:'지역·월별 후보 행 비율',barmode:'group',yaxis:{{title:'후보 비율 (%)'}},xaxis:{{title:'모델'}},margin:{{b:130}}}});
 const region=document.getElementById('regionFilter').value, month=document.getElementById('monthFilter').value;
 let source=injectedData.filter(r=>r.nominal===true&&r.geo_code!=='ALL'&&(region==='ALL'||String(r.geo_code)===region)&&(month==='ALL'||String(r.month)===month)&&(document.getElementById('modelFilter').value==='ALL'||r.model_id===document.getElementById('modelFilter').value));
 let groups=new Map();for(let r of source){{let key=r.model_id+'|'+r.scenario,old=groups.get(key)||{{model_id:r.model_id,scenario:r.scenario,pairs:0,detected:0}};old.pairs+=r.pairs;old.detected+=r.pairs*r.detection_rate;groups.set(key,old);}}let inj=[...groups.values()].map(r=>({{...r,detection_rate:r.detected/r.pairs}}));
 let its=[];for(let key of [...new Set(inj.map(r=>r.model_id))]){{let p=inj.filter(r=>r.model_id===key);its.push({{type:'bar',x:p.map(r=>r.scenario),y:p.map(r=>100*r.detection_rate),name:key}})}}
 Plotly.react('injectionChart',its,{{title:inj.length?'명목 주입 강도에서의 합성 후보 탐지율 (쌍 수 가중)':'선택 조건의 합성 실험 없음',barmode:'group',yaxis:{{title:'합성 후보 탐지율 (%)',range:[0,100]}},xaxis:{{title:'주입 시나리오'}}}});
 let target=document.getElementById('filteredTable');target.replaceChildren();let t=document.createElement('table');let h=t.insertRow();['모델','월','지역','후보 행','고유 소지역','유효 분모','후보 비율'].forEach(x=>{{let c=document.createElement('th');c.textContent=x;h.append(c)}});rows.forEach(r=>{{let tr=t.insertRow();[r.model_id,r.month,r.province,r.candidate_rows,r.candidate_blocks,r.eligible_rows,(100*r.candidate_rate).toFixed(2)+'%'].forEach(v=>{{tr.insertCell().textContent=v}})}});target.append(t);
}}
['model','region','month'].forEach(n=>document.getElementById(n+'Filter').addEventListener('change',drawComparison));drawComparison();
</script>"""


def _candidate_explorer(run, paths):
    blobs = {}
    for path in paths:
        frame = pd.read_parquet(path)
        frame["candidate_reason"] = np.where(frame.candidate,
            "모델의 95% 점수 경계 통과 + 집단(지역 모델은 전체) 유동 10% 이상 감소 + 낮 또는 주말 감소",
            "점수 경계 또는 위축 방향 조건 미충족 (점수 결측은 보류)")
        identity = str(frame.model_id.iloc[0])
        payload = json.dumps(records(frame), ensure_ascii=False, separators=(",", ":")).encode()
        blobs[identity] = base64.b64encode(gzip.compress(payload, mtime=0)).decode()
    other = {}
    for path in paths:
        f = pd.read_parquet(path, columns=["row_id", "candidate", "model_id"])
        for row in f.loc[f.candidate].itertuples(index=False):
            other.setdefault(row.row_id, []).append(row.model_id)
    controls = selector("candidateModel", sorted(blobs), "모델").replace('<option value="ALL">전체</option>', '', 1) + selector("candidateMonth", ["202511", "202512"], "월")
    controls += selector("candidateRegion", ["11", "32"], "지역 (11 서울·32 강원)") + selector("candidateSex", ["남성", "여성", "전체"], "성별")
    controls += selector("candidateAge", ["10", "20", "30", "40", "50", "60+", "전체"], "연령")
    controls += '<label>최소 백분위 <input id="candidateScore" type="number" min="0" max="100" value="0"></label><label>후보 여부 <select id="candidateFlag"><option value="yes">후보만</option><option value="no">비후보만</option><option value="all">전체</option></select></label><label>소지역 코드 검색 <input id="candidateSearch" type="search"></label><button id="exportCandidates">필터 결과 CSV 다운로드</button>'
    return '<p>같은 행 단위의 다른 모델 후보 여부를 함께 표시합니다. 지역 전용 F6는 별도 단위이므로 집단 행과 직접 합치지 않습니다. 한 번에 한 모델을 선택하며, 기본 화면은 상위 100행입니다.</p>' + controls + '<p id="candidateCount">모델을 선택하세요.</p><button id="prevPage">이전</button><button id="nextPage">다음</button><div id="candidateTable" class="scroll"></div>' + f"""
<script>
const candidateBlobs={dump(blobs)}, otherModels={dump(other)};let candidateRows=[], filteredCandidates=[], offset=0;
const candidateColumns=['model_id','month','province','block_code','sex','age_band','raw_score','percentile','candidate','candidate_reason','activity_decline','previous_activity_decline','daytime_decline','weekend_decline','cumulative_decline','decline_acceleration','persistent_decline','relative_decline','share_change','hourly_jsd','weekly_jsd','baseline_flow','other_models'];
async function loadCandidateModel(){{let id=document.getElementById('candidateModel').value;if(id==='ALL'){{document.getElementById('candidateModel').value=Object.keys(candidateBlobs)[0];id=document.getElementById('candidateModel').value;}}let bytes=Uint8Array.from(atob(candidateBlobs[id]),c=>c.charCodeAt(0));let decoded=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));candidateRows=JSON.parse(await new Response(decoded).text());offset=0;filterCandidates();}}
function filterCandidates(){{
 const month=document.getElementById('candidateMonth').value,region=document.getElementById('candidateRegion').value,sex=document.getElementById('candidateSex').value,age=document.getElementById('candidateAge').value,min=Number(document.getElementById('candidateScore').value),flag=document.getElementById('candidateFlag').value,search=document.getElementById('candidateSearch').value;
 filteredCandidates=candidateRows.filter(r=>(month==='ALL'||r.month===month)&&(region==='ALL'||String(r.geo_code)===region)&&(sex==='ALL'||r.sex===sex)&&(age==='ALL'||r.age_band===age)&&r.percentile!==null&&r.percentile>=min&&(flag==='all'||r.candidate===(flag==='yes'))&&r.block_code.includes(search));filteredCandidates.sort((a,b)=>b.percentile-a.percentile||b.raw_score-a.raw_score);offset=0;renderCandidateTable();
}}
function renderCandidateTable(){{let target=document.getElementById('candidateTable');target.replaceChildren();let t=document.createElement('table');let h=t.insertRow();candidateColumns.forEach(k=>{{let th=document.createElement('th');th.textContent=k;h.append(th)}});for(let r of filteredCandidates.slice(offset,offset+100)){{let tr=t.insertRow();for(let k of candidateColumns){{let value=k==='other_models'?(otherModels[r.row_id]||[]).join(', '):r[k];tr.insertCell().textContent=typeof value==='number'?value.toPrecision(6):String(value??'');}}}}target.append(t);document.getElementById('candidateCount').textContent=`필터 결과 ${{filteredCandidates.length.toLocaleString()}}행 · ${{offset+1}}~${{Math.min(offset+100,filteredCandidates.length)}}행 표시 · 점수는 확률이 아닙니다.`;}}
document.getElementById('candidateModel').addEventListener('change',loadCandidateModel);
['candidateMonth','candidateRegion','candidateSex','candidateAge','candidateScore','candidateFlag','candidateSearch'].forEach(k=>document.getElementById(k).addEventListener('input',filterCandidates));
document.getElementById('prevPage').onclick=()=>{{offset=Math.max(0,offset-100);renderCandidateTable()}};document.getElementById('nextPage').onclick=()=>{{if(offset+100<filteredCandidates.length)offset+=100;renderCandidateTable()}};
document.getElementById('exportCandidates').onclick=()=>{{let quote=String.fromCharCode(34), newline=String.fromCharCode(10);let escape=v=>quote+String(v??'').split(quote).join(quote+quote)+quote;let csv=String.fromCharCode(65279)+candidateColumns.join(',')+newline+filteredCandidates.map(r=>candidateColumns.map(k=>escape(k==='other_models'?(otherModels[r.row_id]||[]).join(';'):r[k])).join(',')).join(newline);let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='filtered_candidates.csv';a.click();URL.revokeObjectURL(a.href)}};
if(candidateBlobs['IF_F0'])document.getElementById('candidateModel').value='IF_F0';loadCandidateModel().catch(e=>document.getElementById('candidateCount').textContent='브라우저의 로컬 압축 해제 지원이 필요합니다: '+e.message);
</script>"""


def render_report(run):
    run = Path(run).resolve()
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    metrics, regional, injections = read(run, "model_metrics"), read(run, "regional_metrics"), read(run, "injection_metrics")
    statuses = json.loads((run / "data/experiment_details.json").read_text(encoding="utf-8"))
    counts = pd.Series([v["status"] for v in statuses.values()]).value_counts().to_dict()
    summary = metrics.copy()
    if not injections.empty and "detection_rate" in injections:
        nominal_summary = injections.loc[injections.nominal.eq(True)&injections.geo_code.eq("ALL")&injections.month.eq("ALL")]
        for scenario,label in (("S1","S1 집단 30% 위축 후보율 (%)"),("S7","S7 공통 충격 후보율 (%)")):
            selected=nominal_summary.loc[nominal_summary.scenario.eq(scenario)].set_index("model_id").detection_rate
            summary[label]=summary.model_id.map(selected)*100
    stability_summary=read(run,"stability_metrics")
    if "top20_jaccard" in stability_summary:
        summary["seed 상위20 Jaccard 평균"]=summary.model_id.map(stability_summary.groupby("model_id").top20_jaccard.mean())
    summary.to_csv(run/"data/model_summary.csv",index=False,encoding="utf-8-sig")
    bodies = {}
    cards = '<div class="grid">' + ''.join(f'<div class="card"><div>{k}</div><div class="big">{v}</div></div>' for k, v in [("기록 실험", len(statuses)), ("완료", counts.get("COMPLETE", 0)), ("실패", counts.get("FAILED", 0)), ("보류·미실행", counts.get("SKIPPED", 0))]) + '</div>'
    bodies["index"] = cards + section("이번 실행의 범위", '<p>서울·강원 관측 소지역 · 학습 2025년 10월 · 평가 11월·12월. 운영 IF_BASELINE과 소지역 80/20 학습·보정 비교를 구분합니다. 후보 행 수는 독립 소지역 수나 고립 인원이 아닙니다.</p>')
    main_ids=["IF_BASELINE","IF_F0","IF_F4","LOF_F4","KNN_F4","ROBUST_Z_F4","DUAL_IF"]
    primary=summary.loc[summary.model_id.isin(main_ids)].copy()
    primary["order"]=primary.model_id.map({name:i for i,name in enumerate(main_ids)})
    primary=primary.sort_values("order").drop(columns="order")
    bodies["index"] += section("핵심 모델 요약 — 운영 Baseline과 보정 비교 구분", download("model_summary.csv")+download("model_metrics.csv")+table(primary))
    if "S1 집단 30% 위축 후보율 (%)" in summary:
        indexed=summary.set_index("model_id")
        if {"IF_F0","IF_F4","DUAL_IF"}.issubset(indexed.index):
            label="S1 집단 30% 위축 후보율 (%)"
            bodies["index"] += section("관찰된 주요 차이",f'<p>고정된 집단 30% 위축 주입의 후보율: 공정 비교 IF F0 {indexed.loc["IF_F0",label]:.1f}%, 신규 특징 IF F4 {indexed.loc["IF_F4",label]:.1f}%, Dual-stage {indexed.loc["DUAL_IF",label]:.1f}%. 후보율이 높은 모델이 공통 충격에도 민감할 수 있으므로 S7·증가 방향 대조·안정성과 함께 보세요. 실제 사회고립 탐지 정확도가 개선됐다는 의미가 아닙니다.</p>')
    bodies["index"] += section("전체 특징량·정규화·표본 실험 요약",table(summary))
    bodies["index"] += section("전체 실험 상태와 미완료 이유", table(pd.DataFrame(statuses.values()), limit=500))
    source = json.loads((run / "data/source_diagnostics.json").read_text(encoding="utf-8"))
    dataset = read(run, "dataset_counts")
    bodies["dataset_diagnostics"] = section("집계 수준과 재구성 검증", '<p>연령 자료는 소지역×성×연령×월, 시간·요일 자료는 소지역×월입니다. 지역 공통 특징 4개는 집단별 행에 반복됩니다. F6는 이를 중복 학습하지 않으며 소지역×월 한 행으로 학습합니다. 지역 전체 유동은 같은 연령 원천의 12개 집단 합계로만 정의합니다. 시간·요일 원천과 절대 유동 규모를 맞추지 않습니다. 제공 문서만으로 월 집계가 고유 인원인지 확정할 수 없으므로 고유 인원으로 해석하지 않습니다.</p><pre>' + html.escape(json.dumps(source, ensure_ascii=False, indent=2)) + '</pre>')
    bodies["dataset_diagnostics"] += section("유효·제외 관측", download("dataset_counts.csv") + table(dataset))
    if not dataset.empty:
        bodies["dataset_diagnostics"] += section("지역·월 관측 수", plot(px.bar(dataset.groupby(["geo_code", "month"])[["observations", "common_valid"]].sum().reset_index().melt(id_vars=["geo_code", "month"]), x="month", y="value", color="variable", facet_col="geo_code", barmode="group", labels={"value":"행 수", "month":"관측 월", "variable":"전체·공통 유효"})))
    normalization = read(run, "normalization_diagnostics")
    bodies["dataset_diagnostics"] += section("정규화 그룹 표본·IQR 진단", download("normalization_diagnostics.csv") + table(normalization))
    features = pd.read_parquet(run / "data/features.parquet")
    sample = features.loc[features.common_eligible & features.month.eq("202510")].sample(n=min(3000, int((features.common_eligible & features.month.eq("202510")).sum())), random_state=42)
    for feature in ("activity_decline", "daytime_decline", "relative_decline", "hourly_jsd"):
        bodies["dataset_diagnostics"] += section(f"원본 특징 분포: {feature}", plot(px.histogram(sample, x=feature, color="geo_code", nbins=45, barmode="overlay", labels={feature:feature+" (비율/발산)", "count":"표본 수"})))
    # Render actual train-fitted normalized features, not independently fit plotting transforms.
    model_path = model_directory(run) / "IF_F0.pkl"
    if model_path.exists():
        import pickle
        with model_path.open("rb") as stream:
            fitted = pickle.load(stream)
        x = fitted.scaler.transform(sample)
        norm = pd.DataFrame(x, columns=fitted.columns).assign(geo_code=sample.geo_code.to_numpy())
        bodies["dataset_diagnostics"] += section("학습에서 적합한 N0 정규화 후 분포", plot(px.histogram(norm, x="activity_decline", color="geo_code", barmode="overlay", nbins=45, labels={"activity_decline":"정규화된 집단 유동 감소율", "count":"행 수"})))
    corr = pd.read_csv(run / "data/feature_correlations.csv", index_col=0)
    bodies["dataset_diagnostics"] += section("학습 특징 상관관계", plot(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="학습 공통 관측의 Pearson 상관계수")))
    bodies["dataset_diagnostics"] += section("특징 정의", table(pd.DataFrame([{"feature": k, "definition": v} for k, v in DEFINITIONS.items()])))
    bodies["model_comparison"] = section("공정 비교 필터", _interactive_comparison(regional, injections))
    bodies["model_comparison"] += section("시간·모델 크기", download("model_metrics.csv") + table(metrics))
    if not metrics.empty:
        bodies["model_comparison"] += section("계산 비용", plot(px.bar(metrics, x="model_id", y=["fit_seconds", "inference_seconds"], barmode="group", labels={"value":"초", "model_id":"모델", "variable":"단계"})))
    paths = sorted((run / "data").glob("predictions_*.parquet"))
    prediction_sample, sets, top_sets = [], {}, {}
    for path in paths:
        f = pd.read_parquet(path)
        identity = f.model_id.iloc[0]
        sets[identity] = set(f.loc[f.candidate, "block_code"].astype(str) + "|" + f.loc[f.candidate, "month"])
        top = f.loc[f.candidate].sort_values("raw_score", ascending=False).drop_duplicates(["block_code", "month"])
        top = top.groupby(["geo_code", "month"]).head(20)
        top_sets[identity] = set(top.block_code.astype(str) + "|" + top.month)
        prediction_sample.append(f.sample(n=min(1500, len(f)), random_state=42)[["model_id", "geo_code", "month", "percentile"]])
    names = sorted(sets)
    matrix = [[len(sets[a] & sets[b])/len(sets[a] | sets[b]) if sets[a] | sets[b] else np.nan for b in names] for a in names]
    bodies["model_comparison"] += section("후보 소지역×월 집합 일치도", plot(go.Figure(go.Heatmap(x=names, y=names, z=matrix, zmin=0, zmax=1, colorscale="Blues"))))
    top_matrix = [[len(top_sets[a] & top_sets[b])/len(top_sets[a] | top_sets[b]) if top_sets[a] | top_sets[b] else np.nan for b in names] for a in names]
    bodies["model_comparison"] += section("지역·월별 상위 최대 20개 고유 소지역 Jaccard", plot(go.Figure(go.Heatmap(x=names,y=names,z=top_matrix,zmin=0,zmax=1,colorscale="Blues"))))
    if prediction_sample:
        sampled = pd.concat(prediction_sample, ignore_index=True)
        bodies["model_comparison"] += section("모델별 점수 분포 (보정 분포 백분위)", plot(px.box(sampled, x="model_id", y="percentile", color="geo_code", labels={"percentile":"보정 분포 대비 백분위 (확률 아님)","model_id":"모델"})))
    stability = read(run, "stability_metrics")
    threshold = read(run, "threshold_metrics")
    bodies["model_comparison"] += section("Seed 안정성", download("stability_metrics.csv") + table(stability))
    bodies["model_comparison"] += section("임계값 민감도", download("threshold_metrics.csv") + table(threshold))
    memory=read(run,"memory_metrics")
    if not memory.empty:
        bodies["model_comparison"] += section("메모리 보조 측정", '<p>고정 512행 추론의 Python 추적 할당량입니다. 모델 네이티브 메모리 전체나 학습 프로세스 peak RSS와 같지 않습니다.</p>'+download("memory_metrics.csv")+table(memory))
    if not injections.empty and "detection_rate" in injections:
        nominal = injections.loc[injections.nominal.eq(True) & injections.geo_code.eq("ALL") & injections.month.eq("ALL")].copy()
        pivot = nominal.pivot_table(index="scenario", columns="model_id", values="detection_rate") * 100
        bodies["injection_experiments"] = section("명목 강도 합성 후보 탐지율", '<p>합성 주입 반응이며 실제 사회고립 탐지 Recall·정확도가 아닙니다. 원본을 실제 정상 라벨로 간주하지 않습니다. 분모는 원래 품질 조건을 통과한 주입 쌍이며, 주입 후 보류된 쌍도 기록합니다. S1은 현재 원본 유동 배율 감소, S4는 두 연속 월간 감소 궤적, S5는 직전 비교를 합성적으로 안정화한 뒤 현재 일시 감소입니다. model_input_changed_rate가 0인 경우는 해당 모델 입력이 아니라 후보 후처리 반응만 점검하는 대조입니다.</p>' + plot(px.imshow(pivot, zmin=0, zmax=100, color_continuous_scale="Blues", labels={"color":"합성 후보 탐지율 (%)"})))
        severity = injections.loc[injections.geo_code.eq("ALL") & injections.month.eq("ALL") & injections.status.eq("COMPLETE")]
        bodies["injection_experiments"] += section("강도별 탐지 반응", plot(px.line(severity, x="strength", y="detection_rate", color="model_id", facet_col="scenario", facet_col_wrap=3, markers=True, labels={"strength":"원천 유동량 변형 강도 (비율)","detection_rate":"합성 후보 탐지율"}, height=950)))
        bodies["injection_experiments"] += section("모델 간 비교용 백분위 변화", plot(px.line(severity, x="strength", y="percentile_delta", color="model_id", facet_col="scenario", facet_col_wrap=3, labels={"strength":"주입 강도","percentile_delta":"원본 대비 보정 백분위 변화 (점)"}, height=950)))
        pairs = []
        for path in sorted((run / "data").glob("injection_pairs_*.parquet")):
            f = pd.read_parquet(path)
            pairs.append(f.loc[f.scenario.eq("S1") & f.strength.eq(.3), ["model_id", "original_percentile", "injected_percentile"]])
        if pairs:
            p = pd.concat(pairs).melt(id_vars="model_id", value_vars=["original_percentile", "injected_percentile"], var_name="상태", value_name="백분위")
            bodies["injection_experiments"] += section("S1 30% 주입 전·후 분포", plot(px.box(p, x="model_id", y="백분위", color="상태")))
        bodies["injection_experiments"] += section("전체 합성 평가 수치", download("injection_metrics.csv") + table(injections))
        negative=nominal.loc[nominal.scenario.eq("S6"),[c for c in ("model_id","pairs","actually_increasing_pairs","negative_direction_rejection","detection_rate","raw_anomaly_rate") if c in nominal]]
        bodies["injection_experiments"] += section("증가 방향 배제 점검", '<p>S6는 원본 현재 유동에 증가를 주입합니다. 원본 감소가 크면 주입 뒤에도 감소 상태일 수 있으므로, 증가 방향 배제 비율의 분모는 변형 후 실제 비교 감소율이 0 이하인 쌍입니다. 전체 S6 후보율을 오탐률로 해석하지 않습니다.</p>'+table(negative))
    else:
        bodies["injection_experiments"] = section("미실행", "<p>이 suite에서 합성 주입 결과가 없습니다. 상태 목록을 확인하세요.</p>")
    failures = read(run, "failure_cases")
    bodies["injection_experiments"] += section("주입 후에도 후보가 되지 않은 사례", download("failure_cases.csv") + table(failures, 100))
    ablation = read(run, "ablation_metrics")
    bodies["ablation_study"] = section("특징량 구성", table(pd.DataFrame([{"ID": k, "features": ", ".join(v)} for k, v in FEATURE_SETS.items()])))
    bodies["ablation_study"] += section("공통 표본 결과 — 개선·악화 모두 표시", download("ablation_metrics.csv") + table(ablation))
    populations = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "data").glob("population_*.json"))]
    bodies["ablation_study"] += section("전체 유효 표본 규모 및 별도 적합 상태", table(pd.DataFrame(populations)))
    if "model_id" in metrics:
        bodies["ablation_study"] += section("특징별 전체 유효 표본 모델 결과", table(metrics.loc[metrics.model_id.str.startswith("FULL_")]))
    if not injections.empty and "detection_rate" in injections:
        abl = injections.loc[injections.model_id.str.match(r"IF_F[0-6]$") & injections.nominal.eq(True)
                             & injections.geo_code.eq("ALL") & injections.month.eq("ALL")]
        if not abl.empty:
            bodies["ablation_study"] += section("특징량별 합성 주입 반응", plot(px.imshow(abl.pivot_table(index="scenario",columns="model_id",values="detection_rate")*100,zmin=0,zmax=100,labels={"color":"합성 후보 탐지율 (%)"})))
    if not ablation.empty:
        bodies["ablation_study"] += section("운영 Baseline과 집단 행 후보 Jaccard", plot(px.bar(ablation, x="model_id", y="operating_row_jaccard", labels={"operating_row_jaccard":"Jaccard (지역 전용 단위는 적용 불가)","model_id":"실험"})))
    temporal = read(run, "temporal_metrics")
    bodies["regional_temporal"] = section("지역·월 후보 분포", download("regional_metrics.csv") + table(regional))
    weights=read(run,"dual_weight_sensitivity")
    if not weights.empty:
        bodies["regional_temporal"] += section("Dual-stage 고정 가중치 민감도", '<p>기본 0.5를 유지하며 0.25·0.75는 진단용입니다. 각 경계도 같은 10월 보정 집합에서 고정했고 평가 결과로 가중치를 선택하지 않았습니다.</p>'+download("dual_weight_sensitivity.csv")+plot(px.line(weights,x="group_weight",y="candidate_rate",color="geo_code",facet_col="month",markers=True,labels={"group_weight":"집단 점수 가중치 (지역=1−집단)","candidate_rate":"후보 행 비율"}))+table(weights))
    if not regional.empty:
        bodies["regional_temporal"] += section("정규화·학습 범위 변경의 지역 효과", plot(px.bar(regional.loc[regional.model_id.isin(["IF_F0", "IF_F0_N1", "IF_F0_N2", "IF_F0_N3"])], x="model_id", y="candidate_rate", color="geo_code", facet_col="month", barmode="group", labels={"candidate_rate":"후보 행 비율","model_id":"정규화 실험"})))
    bodies["regional_temporal"] += section("11월 후보의 12월 후속 변화", '<p>후속 월 하나만 존재합니다. 12월 창의 동일 격자에서 10·11·12월 유동을 비교합니다. 11→12월 10% 이상 감소면 지속 감소, 11월에 10월의 90% 미만이던 유동이 12월 90% 이상으로 돌아오면 회복, 그 수준을 회복하지 못했지만 10% 이상 증가하면 반등입니다. 그 외는 정체·부분 회복, 후속 품질이 없으면 평가 불가입니다. 장기 예측 정확도가 아닙니다.</p>' + download("temporal_metrics.csv") + table(temporal))
    if not temporal.empty and "rows" in temporal:
        bodies["regional_temporal"] += section("후속 변화 유형", plot(px.bar(temporal, x="model_id", y="rows", color="type", facet_col="geo_code", labels={"rows":"11월 후보 행 수","model_id":"모델","type":"12월 변화"})))
    bodies["candidate_explorer"] = section("후보 검색·모델 판단 차이", _candidate_explorer(run, paths))
    comparisons = []
    main = {"IF_F0", "IF_F4", "LOF_F4", "KNN_F4", "ROBUST_Z_F4", "DUAL_IF"}
    for path in paths:
        f = pd.read_parquet(path)
        if str(f.model_id.iloc[0]) in main:
            comparisons.append(f[["row_id","model_id","candidate"]])
    if comparisons:
        votes = pd.concat(comparisons).pivot(index="row_id",columns="model_id",values="candidate")
        disagree = votes.loc[votes.sum(axis=1).gt(0) & votes.sum(axis=1).lt(len(votes.columns))].reset_index()
        disagree.to_csv(run/"data/model_disagreements.csv", index=False, encoding="utf-8-sig")
        bodies["candidate_explorer"] += section("같은 관측에서 모델 판단이 다른 사례", download("model_disagreements.csv") + table(disagree,100))
    bodies["methodology"] = section("해석 한계", '<ul><li>실제 사회고립 정답 라벨이 없으며 이상 점수는 고립 확률이 아닙니다.</li><li>합성 탐지율은 주입한 변화에 대한 반응이며 실제 고립 탐지 성능이 아닙니다.</li><li>원본 평가 관측을 실제 정상 표본으로 간주하지 않습니다.</li><li>서울·강원 관측 소지역에만 적용하며 개인별 고립을 추론할 수 없습니다.</li><li>10월 단일 학습 월과 11·12월 두 평가 월로 장기 계절성·지속성을 입증할 수 없습니다.</li><li>성·연령 유동, 시간대·요일 유동, 카드·공공 통계는 집계 수준과 정의가 다릅니다. 시간대·요일 원천을 집단 유동과 임의 동기화하지 않았습니다.</li><li>공통 격자는 각 3개월 창에서 제한합니다. 신규 비율도 현재 창의 동일 격자에서 계산하며, 두 창의 격자가 완전히 같다고 가정하지 않습니다.</li><li>F5는 집단, F6는 지역 관측 단위이며 행 단위 Jaccard를 서로 직접 비교하지 않습니다.</li><li>모델·특징·임계값에 따라 후보 규모가 달라집니다. 하나의 종합 우승 순위를 만들지 않았습니다.</li><li>명시되지 않은 외부 행사·공휴일 영향은 SKIPPED로 기록했습니다.</li><li>좌표계의 측지계가 확정되지 않아 지도는 만들지 않았습니다.</li><li>브라우저에서 모델별 원점수 크기를 비교하지 말고 보정 백분위·후보 비율·합성 반응을 함께 확인하세요.</li></ul>')
    bodies["methodology"] += section("재현 설정", '<pre>' + html.escape((run / "config/experiment_config.json").read_text(encoding="utf-8")) + '</pre>')
    bodies["methodology"] += section("실행 명세·원천 출처", '<pre>' + html.escape(json.dumps(manifest, ensure_ascii=False, indent=2)) + '</pre><p>원천 파일의 SHA-256, 운영 파일 보존 해시, 소지역 분할 목록은 config/ 및 data/source_diagnostics.json에 저장됩니다. 모든 숫자는 저장된 CSV·Parquet·JSON에서 읽으며 리포트 재생성은 모델을 재학습하지 않습니다.</p>')
    javascript = get_plotlyjs()
    for identity, title in PAGES.items():
        nav = ''.join(f'<a class="{"current" if k == identity else ""}" href="{k}.html">{html.escape(v)}</a>' for k,v in PAGES.items())
        document = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | 활동 위축 실험</title><style>{STYLE}</style><script>{javascript}</script></head><body><header><h1>서울·강원 활동 위축 이상 탐지 실험</h1><p>{html.escape(title)} · 실행 {html.escape(manifest["run_id"])}</p></header><nav>{nav}</nav><main><div class="notice">연구용 비지도 비교 실험입니다. 이상 점수·합성 탐지율은 실제 사회고립 확률이나 탐지 정확도가 아닙니다. 운영 Baseline을 보존하고 별도 소지역 학습·보정 분할로 비교했습니다.</div>{bodies[identity]}</main><footer>오프라인 HTML · 외부 CDN 없음 · 전체 수치는 data/ 파일에서 확인</footer></body></html>'
        (run / f"{identity}.html").write_text(document, encoding="utf-8")
    # Persist the long-form comparison, without embedding all models simultaneously in browser memory.
    first = True
    for path in paths:
        frame = pd.read_parquet(path)
        frame["candidate_reason"] = np.where(frame.candidate,"점수 경계 통과 및 현재 10% 이상 유동 위축·낮/주말 감소", "점수 또는 위축 조건 미충족/보류")
        frame.to_csv(run / "data/candidate_comparison.csv", mode="w" if first else "a", header=first, index=False, encoding="utf-8-sig" if first else "utf-8")
        first = False
    write_json(run / "data/report_validation.json", {"html_files": list(PAGES), "inline_plotly": True,
        "external_scripts": False, "candidate_models_embedded": len(paths), "report_generated_from_saved_results": True})
    return run / "index.html"

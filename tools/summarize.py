import json, sys
from pathlib import Path
names = {"01":"Honeycomb","02":"Spiral","03":"Triangular truss","04":"Motor inner","05":"Twisted shell","06":"Angle bracket"}
root = Path(__file__).resolve().parents[1]
print("| 모델 | 결과 | makespan | IoU | Coverage | Overfill | 실패층 | 최소 팔 여유 | 경고 |")
print("|---|---|---|---|---|---|---|---|---|")
for m in names:
    f = root / "runs" / m / "summary.json"
    if not f.exists():
        print(f"| {m} {names[m]} | (미실행) | | | | | | | |"); continue
    s = json.load(open(f, encoding="utf-8"))
    sch, sh, col = s.get("schedule", {}), s.get("shape", {}), s.get("collision", {})
    def g(d, *ks):
        for k in ks:
            if isinstance(d, dict) and k in d: return d[k]
        return None
    mk = g(sch, "makespan_s")
    iou = g(sh, "overall_iou", "iou"); cov = g(sh, "overall_coverage", "coverage"); ov = g(sh, "overall_overfill_ratio", "overfill_ratio")
    fl = g(sh, "failed_layer_count", "failed_layers"); tl = g(sh, "evaluated_layer_count", "layer_count")
    arm = (col.get("arm_envelope") or {}).get("minimum_safety_margin_mm")
    pct = lambda v: f"{v*100:.1f}%" if isinstance(v, (int, float)) else "-"
    print(f"| {m} {names[m]} | {s.get('status')} | {mk/3600:.2f}h | {pct(iou)} | {pct(cov)} | {pct(ov)} | {fl}/{tl} | {arm if arm is None else round(arm)}mm | {len(s.get('issues', []))} |")

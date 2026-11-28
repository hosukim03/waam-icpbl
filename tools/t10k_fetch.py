# -*- coding: utf-8 -*-
"""봉인 묶음 후보: 깨끗한 메시(clean_candidates.csv) 중 출처 표시·변경이 허용된 라이선스만 남기고(NC·ND 제외)
600개를 무작위로(시드 1128) 골라 npz 파일만 Hugging Face 에서 받는다(스크립트 실행 없음). usage: python -I t10k_fetch.py <data 폴더>"""
import sys
from pathlib import Path
import polars as pl
from huggingface_hub import hf_hub_download
sys.stdout.reconfigure(encoding="utf-8")
d = Path(sys.argv[1])
c = pl.read_csv(d / "clean_candidates.csv", infer_schema_length=20000, ignore_errors=True)
OK = ["Creative Commons - Attribution", "Creative Commons - Attribution - Share Alike", "Creative Commons - Public Domain Dedication",
      "Public Domain", "GNU - GPL", "GNU - LGPL", "BSD License"]
c = c.filter(pl.col("License").is_in(OK))
print("라이선스 통과", c.height)
pool = c.sample(n=600, seed=1128, shuffle=True)
keep = ["ID", "Thing ID", "License", "Link", "Category", "Sub-category", "Name", "Author", "num_vertices", "num_faces"]
pool.select([k for k in keep if k in pool.columns]).write_csv(d / "pool.csv")
got = 0
for i, fid in enumerate(pool["ID"].to_list()):
    try:
        hf_hub_download("Thingi10K/Thingi10K", f"npz/{fid}.npz", repo_type="dataset", local_dir=str(d))
        got += 1
    except Exception as e:  # noqa: BLE001
        print("못 받음", fid, repr(e)[:80])
    if (i + 1) % 100 == 0:
        print("진행", i + 1, flush=True)
print("받은 파일", got)

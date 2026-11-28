# -*- coding: utf-8 -*-
"""Thingi10K 메타데이터 살펴보기: 라이선스·분류별 개수, 닫힌 한 덩어리 메시 수."""
import sys
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_fmt_str_lengths(80)
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_width_chars(140)
d = sys.argv[1] + "/metadata"
s = pl.read_csv(f"{d}/input_summary.csv", infer_schema_length=20000, ignore_errors=True)
g = pl.read_csv(f"{d}/geometry_data.csv", infer_schema_length=20000, ignore_errors=True)
c = pl.read_csv(f"{d}/contextual_data.csv", infer_schema_length=20000, ignore_errors=True)
print(s["License"].value_counts().sort("count", descending=True))
print(c["Category"].value_counts().sort("count", descending=True))
m = s.join(g, left_on="ID", right_on="file_id", how="left").join(
    c.select(["Thing ID", "Category", "Sub-category", "Name", "Author"]), on="Thing ID", how="left")
T = lambda n: pl.col(n).cast(pl.Utf8).str.to_uppercase() == "TRUE"
ok = m.filter(T("Closed") & T("Edge manifold") & T("Vertex manifold") & T("Single Component")
              & T("PWN") & (pl.col("num_self_intersections") == 0) & (pl.col("solid") == 1) & T("No degenerate faces"))
print("닫힌 한 덩어리·자기 교차 없음:", ok.height)
print(ok["License"].value_counts().sort("count", descending=True))
print(ok["Category"].value_counts().sort("count", descending=True))
ok.write_csv(sys.argv[1] + "/clean_candidates.csv")

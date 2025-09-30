# 03_monthly_agg.py
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
IN = BASE / "data_interim" / "jeju_date_price.csv"
OUT = BASE / "data_interim" / "jeju_monthly_price.csv"

print("\n")
if not IN.exists():
    raise FileNotFoundError(f"입력 파일이 없습니다: {IN}\n먼저 02_date_price_table.py를 실행해 'jeju_date_price.csv'를 만들어주세요.")

# 1) 로드 & 타입 보정
df = pd.read_csv(IN, encoding="utf-8-sig")
# 안전하게 캐스팅
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["가격_만원"] = pd.to_numeric(df["가격_만원"], errors="coerce")

# 유효값만
df = df.dropna(subset=["날짜", "가격_만원"])
df = df[df["가격_만원"] > 0]

# 2) 월 기준 컬럼 생성
df["월"] = df["날짜"].dt.to_period("M").dt.to_timestamp()  # YYYY-MM-01 형태

# 3) 월별 집계(중앙값/평균/사분위/표본수)
g = df.groupby("월")["가격_만원"]
monthly = pd.DataFrame({
    "건수": g.size(),
    "중앙값_만원": g.median(),
    "평균_만원": g.mean(),
    "P25_만원": g.quantile(0.25),
    "P75_만원": g.quantile(0.75),
})
monthly = monthly.sort_index().reset_index()

# 4) 저장
monthly.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"[저장 완료] {OUT} | rows={len(monthly)}")

# 5) 첫 40행 확인
print("\n[월별 집계 첫 40행]")

print(monthly.head(40).to_string(index=False))

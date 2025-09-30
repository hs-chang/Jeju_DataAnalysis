# 06_city_unitprice_trend.py
from pathlib import Path
import re, csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "raw_data"
INT_DIR = BASE / "data_interim"
OUT_DIR = BASE / "data_processed"
REP_DIR = BASE / "reports"
for d in [INT_DIR, OUT_DIR, REP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

JEJU_TOWNS = {"한경면","한림읍","애월읍","조천읍","구좌읍","추자면","우도면","제주시"}
SEOGWI_TOWNS = {"대정읍","안덕면","남원읍","표선면","성산읍","서귀포시"}

def setup_korean_font():
    try:
        plt.rcParams["font.family"] = "AppleGothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

# ── CSV 로더(안내문/인코딩/구분자 자동) ──
def detect_header_and_sep(path: Path):
    encs = ["cp949","euc-kr","utf-8-sig","utf-16","utf-16-le","utf-16-be","utf-8"]
    header_idx, enc_best, sep_best, score_best = None, None, None, -1
    tokens = ["시군구","법정동","읍면동","아파트","연립","전용","거래","건축","층","년","월","일","도로명","해제","등기","금액","가격","주소"]
    for enc in encs:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as fh:
                for i, line in enumerate(fh):
                    s = line.strip()
                    if len(s) < 3 or s.lower().startswith("sep="):
                        continue
                    counts = {",": line.count(","), "\t": line.count("\t"), ";": line.count(";"), "|": line.count("|")}
                    tok = sum(t in line for t in tokens)
                    sep_char, sep_cnt = max(counts.items(), key=lambda kv: kv[1])
                    if (sep_cnt >= 8) and (tok >= 2):
                        score = sep_cnt + tok*3
                        if score > score_best:
                            header_idx, enc_best, sep_best, score_best = i, enc, ("," if sep_char=="," else ("\t" if sep_char=="\t" else sep_char)), score
        except Exception:
            continue
    return (enc_best or "cp949", (header_idx if header_idx is not None else 5), (sep_best or ","))

def robust_read_one(path: Path) -> pd.DataFrame:
    enc, skip, sep = detect_header_and_sep(path)
    tries = [
        dict(encoding=enc, sep=sep, engine="python", on_bad_lines="skip",
             header=0, skiprows=skip, encoding_errors="replace"),
        dict(encoding=enc, sep=sep, engine="python", on_bad_lines="skip",
             header=0, skiprows=skip, quoting=csv.QUOTE_NONE,
             escapechar="\\", quotechar='"', doublequote=False,
             encoding_errors="replace"),
    ]
    last = None
    for kw in tries:
        try:
            df = pd.read_csv(path, **kw)
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]
            df.columns = [str(c).replace("\u00A0", " ").strip() for c in df.columns]
            if df.shape[1] > 1:
                print(f"  -> OK {path.name} | enc={enc}, sep={'TAB' if sep=='\\t' else sep}, skiprows={skip}, shape={df.shape}")
                return df
        except Exception as e:
            last = e
    raise last if last else RuntimeError(f"로드 실패: {path}")

# ── 공통 유틸 ──
def _norm_key(s: str) -> str:
    s = str(s).lower().replace("\u00A0", " ")
    return re.sub(r"[\s()]", "", s)

def to_num(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("\u00A0", "", regex=False)
         .str.replace(r"[^0-9\.\-]", "", regex=True)
         .pipe(pd.to_numeric, errors="coerce")
    )

# 날짜
def build_date_series(df: pd.DataFrame) -> pd.Series | None:
    cols = { _norm_key(c): c for c in df.columns }
    y = next((cols[k] for k in cols if any(t in k for t in ["년","계약년","거래년"])), None)
    m = next((cols[k] for k in cols if any(t in k for t in ["월","계약월","거래월"])), None)
    d = next((cols[k] for k in cols if any(t in k for t in ["일","계약일","거래일"])), None)
    if y and m and d:
        ser = pd.to_datetime(dict(year=to_num(df[y]), month=to_num(df[m]), day=to_num(df[d])), errors="coerce")
        if ser.notna().mean() > 0.3:
            return ser
    ym = next((c for c in df.columns if any(tok in _norm_key(c) for tok in ["계약년월","거래년월","년월"])), None)
    if ym:
        yms = df[ym].astype(str).str.replace(r"\D","",regex=True).str.zfill(6)
        Y = pd.to_numeric(yms.str[:4], errors="coerce")
        M = pd.to_numeric(yms.str[4:6], errors="coerce")
        d2 = d or next((c for c in df.columns if any(tok in _norm_key(c) for tok in ["계약일","거래일","일"])), None)
        if d2:
            D = to_num(df[d2])
            ser = pd.to_datetime(dict(year=Y, month=M, day=D), errors="coerce")
        else:
            ser = pd.to_datetime(dict(year=Y, month=M, day=1), errors="coerce")
        if ser.notna().mean() > 0.3:
            return ser
    best, best_ratio = None, -1.0
    for c in df.columns:
        dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        ratio = dt.notna().mean()
        if ratio < 0.3:
            s2 = df[c].astype(str).str.replace(r"\D","",regex=True)
            dt2 = pd.to_datetime(s2, format="%Y%m%d", errors="coerce")
            ratio = dt2.notna().mean()
            if ratio > best_ratio:
                best_ratio, best = ratio, dt2
        else:
            if ratio > best_ratio:
                best_ratio, best = ratio, dt
    if best is not None and best_ratio >= 0.5:
        return best
    return None

# 가격(‘금액/가격’ 포함 열만) & 면적(전용면적 우선)
def find_price_col(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    cand = [c for c in cols if ("금액" in c or "가격" in c) and not any(w in c for w in ["년","월","일","일자","날짜","년월"])]
    if not cand: cand = cols
    best, best_key = None, (-1.0, 0.0)
    for c in cand:
        num = to_num(df[c]); valid = num.notna().mean()
        if valid < 0.3: continue
        yyyymm = num.between(199001, 203512) & num.mod(100).between(1,12)
        yyyymmdd = num.between(19900101, 20351231) & num.mod(100).between(1,31)
        if max(yyyymm.mean(), yyyymmdd.mean()) >= 0.3: continue
        med = num.median(skipna=True)
        score = valid + (0.3 if 300 <= med <= 1_500_000 else 0.0)
        if (score, med) > best_key:
            best_key, best = (score, med), c
    return best

def find_area_col(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    pri = [c for c in cols if ("전용" in c and "면적" in c)] or [c for c in cols if "면적" in c]
    best, best_key = None, (-1.0, 0.0)
    for c in pri:
        num = to_num(df[c]); valid = num.notna().mean()
        if valid < 0.3: continue
        med = num.median(skipna=True)
        score = valid + (0.3 if 10 <= med <= 300 else 0.0)
        if (score, med) > best_key:
            best_key, best = (score, med), c
    return best

# 시 단위(제주시/서귀포시) 판별
def extract_city(df: pd.DataFrame) -> pd.Series:
    # 주소 후보 컬럼
    prefer = ["읍면동","읍면동명","법정동","법정동명","행정동","행정동명","지번","지번주소","도로명","도로명주소","주소","시군구","시군구명"]
    cand_cols = [c for c in prefer if any(_norm_key(c)==_norm_key(x) for x in df.columns)]
    cand_cols = [next(x for x in df.columns if _norm_key(x)==_norm_key(c)) for c in cand_cols if any(_norm_key(c)==_norm_key(x) for x in df.columns)]
    if not cand_cols:
        # 그래도 없으면 전체에서 주소/동/읍/면/시가 들어간 컬럼을 사용
        cand_cols = [c for c in df.columns if any(k in c for k in ["주소","읍","면","동","시군구","시"])]
        if not cand_cols:
            return pd.Series([None]*len(df))

    def pick(row) -> str | None:
        t = " ".join(str(row[c]) for c in cand_cols if c in row and pd.notna(row[c]))
        t = t.replace("\u00A0"," ")
        if any(r in t for r in SEOGWI_TOWNS): return "서귀포시"
        if any(r in t for r in JEJU_TOWNS):   return "제주시"
        if "서귀포시" in t: return "서귀포시"
        if "제주시"   in t: return "제주시"
        return None

    return df.apply(pick, axis=1)

def main():
    setup_korean_font()
    files = sorted([p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower()==".csv"],
                   key=lambda p: int(re.search(r"(20\d{2})", p.stem).group(1)) if re.search(r"(20\d{2})", p.stem) else 0)
    if not files: raise FileNotFoundError("raw_data에 .csv 파일이 없습니다.")

    frames = []
    for f in files:
        print(f"\n=== {f.name} ===")
        src = robust_read_one(f)
        dt = build_date_series(src)
        price_col = find_price_col(src)
        area_col  = find_area_col(src)
        city_ser  = extract_city(src)

        print(f"  · 날짜 유효비율: {0.0 if dt is None else round(dt.notna().mean(),3)}")
        print(f"  · 가격열: {price_col} / 면적열: {area_col}")
        print(f"  · 도시(샘플): {list(city_ser.dropna().astype(str).head(3))}")

        if dt is None or price_col is None or area_col is None:
            print("  ! 필수 컬럼 탐지 실패 → 스킵"); continue

        df = pd.DataFrame({
            "날짜": dt,
            "가격_만원": to_num(src[price_col]),
            "전용면적_㎡": to_num(src[area_col]),
            "도시": city_ser
        })
        df = df.dropna(subset=["날짜","가격_만원","전용면적_㎡","도시"])
        df = df[(df["가격_만원"] > 0) & (df["전용면적_㎡"] > 0)]
        df["만원_per_㎡"] = df["가격_만원"] / df["전용면적_㎡"]
        # 비현실적 단가 제거(1~2000 만원/㎡)
        df = df[(df["만원_per_㎡"] > 1) & (df["만원_per_㎡"] < 2000)]
        frames.append(df)

    if not frames:
        raise ValueError("유효한 데이터가 없습니다. (날짜/가격/면적/도시 탐지 실패)")

    core = pd.concat(frames, ignore_index=True).sort_values("날짜").reset_index(drop=True)

    # 월별 평균 (요청: 평균값 기준)
    core["월"] = core["날짜"].dt.to_period("M").dt.to_timestamp()
    monthly = core.groupby(["월","도시"])["만원_per_㎡"].mean().reset_index().rename(columns={"만원_per_㎡":"평균_만원_per_㎡"})
    cnt = core.groupby(["월","도시"]).size().rename("건수").reset_index()
    monthly = monthly.merge(cnt, on=["월","도시"], how="left").sort_values(["월","도시"])

    out_csv = OUT_DIR / "jeju_vs_seogwipo_unitprice_monthly.csv"
    monthly.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[저장] {out_csv} | rows={len(monthly)}")

    # 피벗 + 짧은 보간(≤2개월만 채움)
    piv = monthly.pivot(index="월", columns="도시", values="평균_만원_per_㎡").sort_index()
    full_months = pd.date_range(piv.index.min(), piv.index.max(), freq="MS")
    piv = piv.reindex(full_months)
    piv_interp = piv.interpolate(method="time", limit=2, limit_area="inside", axis=0)

    # Plot
    plt.figure(figsize=(10,5))
    if "제주시" in piv_interp.columns:
        plt.plot(piv_interp.index, piv_interp["제주시"], linewidth=2, label="제주시")
    if "서귀포시" in piv_interp.columns:
        plt.plot(piv_interp.index, piv_interp["서귀포시"], linewidth=2, label="서귀포시")
    plt.title("제주시 vs 서귀포시 — 면적당 평균가격(만원/㎡) 추이")
    plt.xlabel("월"); plt.ylabel("만원/㎡")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    plt.legend()
    plt.tight_layout()
    out_png = REP_DIR / "04_jeju_vs_seogwipo_unitprice.png"
    plt.savefig(out_png, dpi=150)
    print(f"[저장] {out_png}")

    print("\n[미리보기 12행]")
    print(monthly.head(12).to_string(index=False))

if __name__ == "__main__":
    main()

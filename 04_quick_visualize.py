# 04_quick_visualize.py — 내용기반 날짜/가격 탐지(가격은 '금액/가격' 컬럼에서만) + 시각화
from pathlib import Path
import re, csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE    = Path(__file__).resolve().parent
RAW_DIR = BASE / "raw_data"
INT_DIR = BASE / "data_interim"
REP_DIR = BASE / "reports"
INT_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

# ────────── 폰트(맥) ──────────
def setup_korean_font():
    try:
        plt.rcParams["font.family"] = "AppleGothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

# ────────── 로더(안내문/인코딩/구분자 자동) ──────────
def detect_header_and_sep(path: Path):
    encs = ["cp949","euc-kr","utf-8-sig","utf-16","utf-16-le","utf-16-be","utf-8"]
    header_idx, enc_best, sep_best, score_best = None, None, None, -1
    tokens = ["시군구","법정동","읍면동","아파트","연립","전용","거래","건축","층","년","월","일","도로명","해제","등기","금액","가격"]
    for enc in encs:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as fh:
                for i, line in enumerate(fh):
                    s = line.strip()
                    if len(s) < 3 or s.lower().startswith("sep="):
                        continue
                    counts = {",": line.count(","), "\t": line.count("\t"), ";": line.count(";"), "|": line.count("|")}
                    token_score = sum(t in line for t in tokens)
                    sep_char, sep_cnt = max(counts.items(), key=lambda kv: kv[1])
                    if (sep_cnt >= 8) and (token_score >= 2):
                        score = sep_cnt + token_score*3
                        if score > score_best:
                            score_best = score
                            header_idx, enc_best = i, enc
                            sep_best = "," if sep_char == "," else ("\t" if sep_char == "\t" else sep_char)
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

# ────────── 공통 유틸 ──────────
def _norm_key(s: str) -> str:
    s = str(s).lower().replace("\u00A0", " ")
    return re.sub(r"[\s()]", "", s)

def to_numeric_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("\u00A0", "", regex=False)
         .str.replace(r"[^0-9\.\-]", "", regex=True)
         .pipe(pd.to_numeric, errors="coerce")
    )

# 날짜처럼 보이는 수치(YYYYMM / YYYYMMDD) 판별 → 가격 후보에서 제외
def looks_like_yyyymm(num: pd.Series) -> float:
    # 199001 ~ 203512 이내 & 마지막 두 자리가 1..12
    m = num.between(199001, 203512) & num.mod(100).between(1, 12)
    return m.mean()

def looks_like_yyyymmdd(num: pd.Series) -> float:
    m = num.between(19900101, 20351231) & num.mod(100).between(1, 31)
    return m.mean()

# ────────── 날짜 열 구성 ──────────
def try_build_from_ymd(df: pd.DataFrame) -> pd.Series | None:
    norm = { _norm_key(c): c for c in df.columns }
    y_key = next((k for k in norm if any(t in k for t in ["년","계약년","거래년"])), None)
    m_key = next((k for k in norm if any(t in k for t in ["월","계약월","거래월"])), None)
    d_key = next((k for k in norm if any(t in k for t in ["일","계약일","거래일"])), None)
    if y_key and m_key and d_key:
        y = pd.to_numeric(df[norm[y_key]], errors="coerce")
        m = pd.to_numeric(df[norm[m_key]], errors="coerce")
        d = pd.to_numeric(df[norm[d_key]], errors="coerce")
        dt = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
        if dt.notna().mean() > 0.3:
            return dt
    return None

def guess_date_series(df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    # 1) 년/월/일
    dt = try_build_from_ymd(df)
    if dt is not None:
        return dt, "Y-M-D 조합"
    # 2) 계약년월 + 일
    ym_col = None
    for c in df.columns:
        if any(tok in _norm_key(c) for tok in ["계약년월","거래년월","년월"]):
            ym_col = c; break
    if ym_col:
        ym = df[ym_col].astype(str).str.replace(r"\D","",regex=True).str.zfill(6)
        y = pd.to_numeric(ym.str[:4], errors="coerce")
        m = pd.to_numeric(ym.str[4:6], errors="coerce")
        d_col = next((c for c in df.columns if any(tok in _norm_key(c) for tok in ["계약일","거래일","일"])), None)
        if d_col:
            d = pd.to_numeric(df[d_col], errors="coerce")
            dt2 = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
        else:
            dt2 = pd.to_datetime(dict(year=y, month=m, day=1), errors="coerce")
        if dt2.notna().mean() > 0.3:
            return dt2, f"{ym_col} + 일"
    # 3) 단일열 파싱(최고 유효비율 선택)
    best_dt, best_ratio, best_col = None, -1.0, None
    for c in df.columns:
        s = df[c]
        dt3 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        ratio = dt3.notna().mean()
        if ratio < 0.3:
            s2 = s.astype(str).str.replace(r"\D","",regex=True)
            dt3b = pd.to_datetime(s2, format="%Y%m%d", errors="coerce")
            ratio = dt3b.notna().mean()
            if ratio > 0.3:
                dt3 = dt3b
        if ratio > best_ratio:
            best_ratio, best_dt, best_col = ratio, dt3, c
    if best_dt is not None and best_ratio >= 0.5:
        mask = (best_dt.dt.year >= 2010) & (best_dt.dt.year <= 2035)
        if mask.mean() >= 0.5:
            return best_dt, f"단일열 파싱: {best_col}"
    return None, "탐지 실패"

# ────────── 가격 열 탐지(핵심 수정) ──────────
def find_price_column(df: pd.DataFrame) -> tuple[str | None, str]:
    """
    1) 헤더 이름 우선: '거래금액', '매매금액', '...금액', '...가격' 포함 컬럼 중에서 선택
       - 날짜/일자/년월 등의 단어가 들어간 컬럼은 제외
    2) 숫자 유효비율이 가장 높고, 중앙값이 '만원' 스케일(예: 5,000~1,000,000)인 것을 선호
    3) YYYYMM/ YYYYMMDD 패턴(날짜처럼 보이는 값)이 많은 컬럼은 강제 제외
    """
    cols = list(df.columns)
    norm_map = {c: _norm_key(c) for c in cols}

    # 후보군: '금액' 또는 '가격' 포함 & 날짜 관련 토큰 미포함
    def is_price_name(nk: str) -> bool:
        has_price = ("금액" in nk) or ("가격" in nk)
        has_date_word = any(tok in nk for tok in ["년월","년","월","일","일자","날짜"])
        return has_price and not has_date_word

    named_candidates = [c for c in cols if is_price_name(norm_map[c])]

    def score_col(c: str):
        num = to_numeric_series(df[c])
        if num.notna().mean() < 0.3:
            return (-1, 0, c)
        # 날짜 패턴 비중 계산
        yyyymm_ratio   = looks_like_yyyymm(num)
        yyyymmdd_ratio = looks_like_yyyymmdd(num)
        if max(yyyymm_ratio, yyyymmdd_ratio) >= 0.5:
            return (-1, 0, c)  # 강제 탈락
        med = num.median(skipna=True)
        plausible = 500 <= med <= 1_500_000  # 만원 기준: 0.5만~150만(=0.5억~150억)
        bonus = 0.3 if plausible else 0.0
        return (num.notna().mean() + bonus, med, c)

    chosen, reason = None, ""
    candidates = named_candidates if named_candidates else cols  # 1차: 이름 후보, 없으면 전체 탐색
    scored = sorted((score_col(c) for c in candidates), reverse=True)
    for valid_ratio, med, c in scored:
        if valid_ratio <= 0:  # 탈락
            continue
        chosen = c
        reason = f"열이름 기반({'이름후보' if c in named_candidates else '전체탐색'}) + 유효비율/중앙값 점수"
        break
    return chosen, reason

# ────────── raw_data → (날짜, 가격_만원) ──────────
def build_from_raw() -> pd.DataFrame:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"raw_data 폴더 없음: {RAW_DIR}")
    files = [p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower()==".csv"]
    if not files:
        raise FileNotFoundError("raw_data에 .csv 파일이 없습니다.")

    def year_key(p: Path) -> int:
        m = re.search(r"(20\d{2})", p.stem)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=year_key)
    print("[raw_data 로딩] →", [p.name for p in files])

    frames = []
    for f in files:
        print(f"\n=== {f.name} ===")
        df = robust_read_one(f)

        # 날짜
        date_ser, date_src = guess_date_series(df)
        print(f"  · 날짜열 탐지: {date_src} / 유효비율={0.0 if date_ser is None else round(date_ser.notna().mean(),3)}")
        if date_ser is None:
            print("  · 날짜 탐지 실패 → 스킵")
            continue

        # 가격 (핵심)
        price_col, why = find_price_column(df)
        if price_col is None:
            print("  · 가격열 탐지 실패 → 스킵")
            continue
        price_num = to_numeric_series(df[price_col])
        # 날짜형 패턴이 아닌지 최종 방어
        if max(looks_like_yyyymm(price_num), looks_like_yyyymmdd(price_num)) >= 0.5:
            print(f"  · '{price_col}'이 날짜형 패턴(YYYYMM/DDD)으로 판단 → 스킵")
            continue

        mini = pd.DataFrame({"날짜": date_ser, "가격_만원": price_num})
        mini = mini.dropna(subset=["날짜","가격_만원"])
        mini = mini[mini["가격_만원"] > 0]
        print(f"  · 선택된 가격열: {price_col} ({why}) / 유효행={len(mini)} / 중앙값≈{0 if mini.empty else int(mini['가격_만원'].median()):,}만원")
        if not mini.empty:
            frames.append(mini)

    if not frames:
        raise ValueError("유효한 날짜·가격 쌍을 구성하지 못했습니다. (컬럼명이 매우 이례적일 수 있음)")
    out = pd.concat(frames, ignore_index=True).sort_values("날짜").reset_index(drop=True)
    tmp_path = INT_DIR / "jeju_date_price_auto.csv"
    out.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    print(f"\n[중간저장] {tmp_path} | rows={len(out)}")
    return out

# ────────── 메인(집계+그리기) ──────────
def main():
    setup_korean_font()
    df = build_from_raw()

    # 월별 집계
    df["월"] = df["날짜"].dt.to_period("M").dt.to_timestamp()
    g = df.groupby("월")["가격_만원"]
    monthly = pd.DataFrame({
        "건수": g.size(),
        "중앙값_만원": g.median(),
        "평균_만원": g.mean(),
        "P25_만원": g.quantile(0.25),
        "P75_만원": g.quantile(0.75),
    }).reset_index().sort_values("월")
    if monthly.empty:
        raise ValueError("월별 집계가 비었습니다.")

    # Plot 1
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(monthly["월"], monthly["중앙값_만원"], linewidth=2)
    plt.title("제주도 아파트 월별 거래가격 중앙값(만원)")
    plt.xlabel("월"); plt.ylabel("가격(만원)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    last = monthly.iloc[-1]
    plt.annotate(f"{int(last['중앙값_만원']):,}만원",
                 xy=(last["월"], last["중앙값_만원"]),
                 xytext=(10, 10), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", lw=1))
    plt.tight_layout()
    out1 = REP_DIR / "01_monthly_median.png"
    plt.savefig(out1, dpi=150)
    print(f"[저장] {out1}")

    # Plot 2
    fig2 = plt.figure(figsize=(10,5))
    plt.scatter(df["날짜"], df["가격_만원"], s=8, alpha=0.35)
    roll = df.set_index("날짜")["가격_만원"].sort_index().rolling("30D").median()
    plt.plot(roll.index, roll.values, linewidth=2)
    plt.title("일별 거래(점) & 30일 롤링 중앙값(선)")
    plt.xlabel("날짜"); plt.ylabel("가격(만원)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    plt.tight_layout()
    out2 = REP_DIR / "02_daily_scatter_rolling.png"
    plt.savefig(out2, dpi=150)
    print(f"[저장] {out2}")

    print("\n[월별 집계 상위 12행]")
    print(monthly.head(12).to_string(index=False))

if __name__ == "__main__":
    main()

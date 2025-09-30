# 02a_build_date_price_clean.py
from pathlib import Path
import pandas as pd
import re, csv

BASE = Path(__file__).resolve().parent
RAW  = BASE / "raw_data"
OUTD = BASE / "data_interim"
OUTD.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 0) 파일명 정리(.scv→.csv, 앞뒤 공백/작은따옴표 제거)
def sanitize_filenames(raw_dir: Path) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(f"폴더 없음: {raw_dir}")
    for p in list(raw_dir.iterdir()):
        if not p.is_file():
            continue
        new_name = p.name.strip()
        if new_name.startswith("'") and new_name.endswith("'"):
            new_name = new_name[1:-1].strip()
        if new_name.lower().endswith(".scv"):
            new_name = new_name[:-4] + ".csv"
        if new_name != p.name:
            p.rename(p.with_name(new_name))

# ──────────────────────────────────────────────────────────────────────────────
# 1) 헤더/구분자/인코딩 추정(정부 CSV 안내문 여러 줄 스킵)
def detect_header_and_sep(path: Path):
    encs = ["cp949","euc-kr","utf-8-sig","utf-16","utf-16-le","utf-16-be","utf-8"]
    header_idx, enc_best, sep_best, score_best = None, None, None, -1
    tokens = ["시군구","법정동","읍면동","아파트","연립","전용","거래","건축","층","년","월","일","도로명","해제","등기"]
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
                    header_like = (sep_cnt >= 8) and (token_score >= 2)
                    if header_like:
                        score = sep_cnt + token_score*3
                        if score > score_best:
                            score_best = score
                            header_idx, enc_best = i, enc
                            sep_best = "," if sep_char == "," else ("\t" if sep_char == "\t" else sep_char)
        except Exception:
            continue
    # 실패 시 보수적 기본값
    return (enc_best or "cp949", (header_idx if header_idx is not None else 5), (sep_best or ","))

# ──────────────────────────────────────────────────────────────────────────────
# 2) 단일 파일 robust 로드
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
            # 'Unnamed: ...' 같은 잔여열 제거
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]
            if df.shape[1] > 1:
                print(f"  -> OK {path.name} | enc={enc}, sep={'TAB' if sep=='\\t' else sep}, skiprows={skip}, shape={df.shape}")
                return df
        except Exception as e:
            last = e
    raise last if last else RuntimeError(f"로드 실패: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) 컬럼 표준화 & 유연한 컬럼 탐지
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # NBSP(0xA0) → 공백, 앞뒤 공백 제거
    df.columns = [str(c).replace("\u00A0", " ").strip() for c in df.columns]
    return df

def _norm_key(s: str) -> str:
    s = str(s).lower().replace("\u00A0", " ")
    # 공백·괄호 제거
    s = re.sub(r"[\s()]+", "", s)
    return s

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    candidates 예: ["거래금액(만원)","거래금액","매매금액(만원)","매매금액","가격(만원)","가격"]
    - 공백/괄호/NBSP 제거한 키로 완전 일치 우선, 없으면 부분 포함 매칭
    """
    keys = {_norm_key(c): c for c in df.columns}
    for cand in candidates:
        tok = _norm_key(cand)
        if tok in keys:      # 완전 일치
            return keys[tok]
        for k, orig in keys.items():   # 부분 포함
            if tok in k:
                return orig
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 4) 취소/무효 거래 제거(보수적으로)
def drop_canceled(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # 해제여부가 있으면 이것만 우선 기준(가장 신뢰)
    if "해제여부" in d.columns:
        v = d["해제여부"].astype(str).str.strip().str.upper()
        canceled = v.isin({"O","Y","1","TRUE","T"}) | v.str.contains("해제", na=False)
        return d[~canceled]
    # 해제여부가 없고 '해제사유발생일'만 있을 때: 값이 비어있지 않으면 취소로 간주
    if "해제사유발생일" in d.columns:
        s = d["해제사유발생일"].astype(str).str.strip()
        keep = (s == "") | s.str.lower().isin({"nan","nat"}) | d["해제사유발생일"].isna()
        return d[keep]
    return d

# ──────────────────────────────────────────────────────────────────────────────
# 5) 날짜/가격 파생
def build_date_column(df: pd.DataFrame) -> pd.Series:
    to_num = lambda s: pd.to_numeric(s, errors="coerce")

    # 1) '년/월/일' 조합
    y_col = find_col(df, ["년","계약년","거래년"])
    m_col = find_col(df, ["월","계약월","거래월"])
    d_col = find_col(df, ["일","계약일","거래일"])
    if y_col and m_col and d_col:
        y = to_num(df[y_col]); m = to_num(df[m_col]); d = to_num(df[d_col])
        return pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")

    # 2) '계약년월'(YYYYMM) + (계약일/거래일/일)
    ym_col = find_col(df, ["계약년월","년월","거래년월"])
    if ym_col:
        ym = df[ym_col].astype(str).str.replace(r"\D","",regex=True).str.zfill(6)
        y = pd.to_numeric(ym.str[:4], errors="coerce")
        m = pd.to_numeric(ym.str[4:6], errors="coerce")
        d2 = d_col or find_col(df, ["계약일","거래일","일"])
        if d2:
            d = pd.to_numeric(df[d2], errors="coerce")
            return pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
        return pd.to_datetime(dict(year=y, month=m, day=1), errors="coerce")

    # 3) 완전 날짜 문자열(YYYY-MM-DD 등)
    s_col = find_col(df, ["계약일자","거래일자","계약일","거래일"])
    if s_col:
        return pd.to_datetime(df[s_col], errors="coerce")

    # 4) 실패 시 NaT
    return pd.to_datetime(pd.Series([pd.NaT]*len(df)))

def clean_price_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace("\u00A0", "", regex=False)
         .pipe(pd.to_numeric, errors="coerce")
    )

# ──────────────────────────────────────────────────────────────────────────────
# 6) 메인
def main():
    sanitize_filenames(RAW)
    files = [p for p in RAW.iterdir() if p.is_file() and p.suffix.lower()==".csv"]
    if not files:
        listing = [repr(p.name) for p in RAW.iterdir()]
        raise FileNotFoundError(f"raw_data에 .csv 없음. 현재 목록: {listing}")

    # 2023→2024→2025 정렬
    def year_key(p: Path) -> int:
        m = re.search(r"(20\\d{2})", p.stem)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=year_key)
    print("[발견된 파일들]", [p.name for p in files])

    frames = []
    for f in files:
        print(f"\n=== {f.name} ===")
        df = robust_read_one(f)
        df = normalize_columns(df)
        df = drop_canceled(df)

        # 가격 컬럼 찾기(괄호/공백/NBSP 무시 + 부분 포함 허용)
        price_col = find_col(df, ["거래금액(만원)","거래금액","매매금액(만원)","매매금액","가격(만원)","가격"])
        if price_col is None:
            raise KeyError(f"{f.name}: 가격 컬럼 탐지 실패. 실제 컬럼: {list(df.columns)}")

        date_ser = build_date_column(df)
        price_ser = clean_price_series(df[price_col])

        # 디버그(선택): 유효 개수 찍기
        print(f"   · 날짜 유효: {int(date_ser.notna().sum())} / 가격 유효: {int(price_ser.notna().sum())}")

        mini = pd.DataFrame({"날짜": date_ser, "가격_만원": price_ser}).dropna(subset=["날짜","가격_만원"])
        mini = mini[mini["가격_만원"] > 0]
        frames.append(mini)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values("날짜").reset_index(drop=True)

    # 원본 2컬럼 저장
    raw_path = OUTD / "jeju_date_price.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"[저장] 원본 2컬럼: {raw_path} | rows={len(raw)}")

    # ✅ 유효 데이터 0건이면, 트리밍 없이 저장하고 종료
    if raw.empty:
        clean_path = OUTD / "jeju_date_price_clean.csv"
        raw.to_csv(clean_path, index=False, encoding="utf-8-sig")
        print(f"[경고] 유효한 날짜/가격이 0건입니다. 취소거래 기준이 과했거나 컬럼명이 예상과 다릅니다.")
        print(f"       트리밍 없이 그대로 저장 → {clean_path}")
        print("\n[첫 40행]")
        print(raw.head(40).to_string(index=False))
        raise SystemExit(0)

    # ── 가벼운 이상치 트리밍(월별 P1~P99, 표본<10인 월은 보존)
    tmp = raw.copy()
    tmp["월"] = tmp["날짜"].dt.to_period("M").dt.to_timestamp()
    # 분위수 계산 (표본 적은 월은 NaN 나올 수 있음)
    q = tmp.groupby("월")["가격_만원"].quantile([0.01, 0.99]).unstack()
    if not q.empty:
        q = q.rename(columns={0.01:"q01", 0.99:"q99"})
    n = tmp.groupby("월").size().rename("건수")
    tmp = tmp.merge(q, on="월", how="left").merge(n, on="월", how="left")

    # keep 규칙: 건수<10 이면 무조건 keep, 아니면 q01~q99 사이만
    keep = (tmp["건수"] < 10) | (
        (tmp["가격_만원"] >= tmp["q01"]) & (tmp["가격_만원"] <= tmp["q99"])
    )
    clean = tmp[keep].drop(columns=[c for c in ["월","q01","q99","건수"] if c in tmp.columns])

    clean_path = OUTD / "jeju_date_price_clean.csv"
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    removed = len(raw) - len(clean)
    print(f"[저장] 클린 2컬럼: {clean_path} | rows={len(clean)} (제거 {removed}건)")

    # 첫 40행 확인
    print("\n[클린 표 첫 40행]")
    print(clean.head(40).to_string(index=False))

if __name__ == "__main__":
    main()

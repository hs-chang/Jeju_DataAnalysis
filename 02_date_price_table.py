# 02_date_price_table.py
from pathlib import Path
import pandas as pd
import re, csv

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw_data"
OUTDIR = BASE / "data_interim"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# (1) 파일명 정리(.scv→.csv, 앞뒤 공백/따옴표 제거)
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

# (2) 헤더/구분자/인코딩 추정 (정부 CSV 안내문 포함 대비)
def detect_header_and_sep(path: Path):
    enc_candidates = ["cp949", "euc-kr", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-8"]
    header_idx_best, enc_best, sep_best, score_best = None, None, None, -1
    expected_tokens = ["시군구","법정동","읍면동","아파트","연립","전용","거래","건축","층","년","월","일","도로명","해제","등기"]

    for enc in enc_candidates:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as fh:
                for i, line in enumerate(fh):
                    s = line.strip()
                    if len(s) < 3:
                        continue
                    counts = {",": line.count(","), "\t": line.count("\t"), ";": line.count(";"), "|": line.count("|")}
                    token_score = sum(tok in line for tok in expected_tokens)
                    sep_char, sep_cnt = max(counts.items(), key=lambda kv: kv[1])
                    header_like = (sep_cnt >= 8) and (token_score >= 2)
                    if s.lower().startswith("sep="):
                        header_like = False
                    if header_like:
                        score = sep_cnt + token_score * 3
                        if score > score_best:
                            score_best = score
                            header_idx_best = i
                            enc_best = enc
                            sep_best = "," if sep_char == "," else ("\t" if sep_char == "\t" else sep_char)
        except Exception:
            continue
    if header_idx_best is None:
        return ("cp949", 5, ",")
    return (enc_best or "cp949", header_idx_best, sep_best or ",")

# (3) 파일 하나를 튼튼하게 읽기
def robust_read_one(path: Path) -> pd.DataFrame:
    enc, header_idx, sep = detect_header_and_sep(path)
    try_order = [
        dict(encoding=enc, sep=sep, engine="python", on_bad_lines="skip",
             header=0, skiprows=header_idx, encoding_errors="replace"),
        dict(encoding=enc, sep=sep, engine="python", on_bad_lines="skip",
             header=0, skiprows=header_idx, quoting=csv.QUOTE_NONE,
             escapechar="\\", quotechar='"', doublequote=False,
             encoding_errors="replace"),
    ]
    last_err = None
    for kwargs in try_order:
        try:
            df = pd.read_csv(path, **kwargs)
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]
            if df.shape[1] > 1:
                print(f"  -> OK: {path.name} | enc={enc}, sep={'TAB' if sep=='\\t' else sep}, skiprows={header_idx}, shape={df.shape}")
                return df
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f"로드 실패: {path}")

# (4) 컬럼 표준화 헬퍼
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 공백 제거 + 괄호는 유지 (예: 전용면적(㎡))
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_price_column(cols):
    # 우선순위로 흔히 쓰는 이름 검색
    priority = [
        "거래금액(만원)", "거래금액", "매매금액(만원)", "매매금액",
        "가격(만원)", "가격"
    ]
    for p in priority:
        for c in cols:
            if p == c:
                return c
    # 부분일치 백업
    for c in cols:
        if ("거래금액" in c) or ("매매금액" in c) or ("가격" in c):
            return c
    return None

def build_date_column(df: pd.DataFrame) -> pd.Series:
    cols = set(df.columns)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    # A) '년' '월' '일'이 모두 있는 경우: 그대로 합치기
    if {"년","월","일"}.issubset(cols):
        y = to_num(df["년"]); m = to_num(df["월"]); d = to_num(df["일"])
        return pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")

    # B) '계약년월'(YYYYMM) + (계약일/거래일/일) 조합
    if "계약년월" in cols:
        ym = df["계약년월"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
        y = pd.to_numeric(ym.str[:4], errors="coerce")
        m = pd.to_numeric(ym.str[4:6], errors="coerce")

        # 우선순위: 계약일 > 거래일 > 일
        day_col = next((c for c in ["계약일","거래일","일"] if c in cols), None)
        if day_col is not None:
            d = pd.to_numeric(df[day_col], errors="coerce")
            # day 값이 없는 행은 NaT로 처리되어 후단 dropna에서 제거됨
            return pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
        else:
            # 진짜 '일' 정보가 전혀 없을 때만 1일로 보정
            return pd.to_datetime(dict(year=y, month=m, day=1), errors="coerce")

    # C) '계약일자' 같은 완전 날짜 문자열이 있으면 그대로 파싱
    for c in ["계약일자","거래일자","계약일","거래일"]:
        if c in cols:
            return pd.to_datetime(df[c], errors="coerce")

    # 그 외는 NaT → 후단 dropna로 제거
    return pd.to_datetime(pd.Series([pd.NaT]*len(df)))


def clean_price_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace("\u00A0", "", regex=False)  # &nbsp;
         .pipe(pd.to_numeric, errors="coerce")
    )

# ──────────────────────────────────────────────────────────────────────────────
def main():
    sanitize_filenames(RAW)
    files = [p for p in RAW.iterdir() if p.is_file() and p.suffix.lower()==".csv"]
    if not files:
        listing = [repr(p.name) for p in RAW.iterdir()]
        raise FileNotFoundError(f"raw_data에 .csv가 없습니다. 현재 목록: {listing}")

    # 2023→2024→2025 정렬
    def year_key(p: Path) -> int:
        m = re.search(r"(20\d{2})", p.stem)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=year_key)
    print("[발견된 파일들]", [p.name for p in files])

    # 읽기 + 표준화
    frames = []
    for f in files:
        print(f"\n=== {f.name} ===")
        df = robust_read_one(f)
        df = normalize_columns(df)

        price_col = pick_price_column(df.columns)
        if price_col is None:
            raise KeyError(f"{f.name}: 가격 컬럼(거래금액/매매금액/가격) 탐지 실패. 실제 컬럼들: {list(df.columns)}")

        date_ser = build_date_column(df)
        price_ser = clean_price_series(df[price_col])

        mini = pd.DataFrame({
            "날짜": date_ser,
            "가격_만원": price_ser
        })
        # 유효한 날짜/가격만
        mini = mini.dropna(subset=["날짜", "가격_만원"])
        frames.append(mini)

    # 병합 → 정렬 → 저장
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("날짜").reset_index(drop=True)

    # 저장
    out_csv = OUTDIR / "jeju_date_price.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] {out_csv} | rows={len(out)}")

    # 첫 40행 출력
    print("\n[첫 40행]")
    print(out.head(40).to_string(index=False))

if __name__ == "__main__":
    main()

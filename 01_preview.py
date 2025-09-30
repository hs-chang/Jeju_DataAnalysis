
from pathlib import Path
import pandas as pd
import re, csv

# ──────────────────────────────────────────────────────────────────────────────
# 설정: 현재 파일 위치 기준으로 raw_data 폴더 스캔 (수정 불필요)
BASE = Path(__file__).resolve().parent
RAW = BASE / "raw_data"

# ──────────────────────────────────────────────────────────────────────────────
# 유틸: 파일명 정리(.scv→.csv, 앞뒤 공백/따옴표 제거)
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
# 헤더 라인/구분자/인코딩 추정
def detect_header_and_sep(path: Path):
    """
    파일 앞부분을 훑어서 '진짜 헤더가 시작되는 줄 index', '인코딩', '구분자'를 추정.
    - 정부 CSV에 자주 있는 안내문(헤더 전 여러 줄)을 자동 스킵
    - 쉼표/탭/세미콜론/파이프 지원
    """
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
                    # 구분자 카운트
                    counts = {
                        ",": line.count(","),
                        "\t": line.count("\t"),
                        ";": line.count(";"),
                        "|": line.count("|"),
                    }
                    # 헤더에 흔한 토큰 포함 개수
                    token_score = sum(tok in line for tok in expected_tokens)
                    # 쉼표/탭 등 구분자 개수 중 최댓값
                    sep_char, sep_cnt = max(counts.items(), key=lambda kv: kv[1])
                    # 헤더일 법한 조건: 구분자가 충분히 많고, 토큰도 2개 이상
                    # (파일마다 상이하므로 살짝 느슨하게)
                    header_like = (sep_cnt >= 8) and (token_score >= 2)
                    # 혹시 첫 줄 "sep=," 같은 프리앰블이 있으면 강하게 가산
                    if s.lower().startswith("sep="):
                        header_like = False  # 프리앰블은 헤더로 보지 않음

                    if header_like:
                        # 점수: 구분자 개수 + 토큰 점수*3 (토큰 가중치)
                        score = sep_cnt + token_score * 3
                        if score > score_best:
                            score_best = score
                            header_idx_best = i
                            enc_best = enc
                            sep_best = "," if sep_char == "," else ("\t" if sep_char == "\t" else sep_char)
        except Exception:
            continue

    # 실패하면 보수적 기본값(앞 0~5줄은 안내문일 가능성)
    if header_idx_best is None:
        # cp949 가정, 쉼표 가정, 5줄 스킵
        return ("cp949", 5, ",")
    return (enc_best or "cp949", header_idx_best, sep_best or ",")

# ──────────────────────────────────────────────────────────────────────────────
# 하나의 파일을 튼튼하게 읽기
def robust_read_one(path: Path) -> pd.DataFrame:
    enc, header_idx, sep = detect_header_and_sep(path)
    # 1차 시도: 일반적인 CSV 파싱
    try_order = [
        dict(encoding=enc, sep=sep, engine="python", on_bad_lines="skip",
             header=0, skiprows=header_idx, encoding_errors="replace"),
        # 따옴표 깨짐 대비(QUOTE_NONE)
        dict(encoding=enc, sep=sep, engine="python", on_bad_lines="skip",
             header=0, skiprows=header_idx, quoting=csv.QUOTE_NONE,
             escapechar="\\", quotechar='"', doublequote=False,
             encoding_errors="replace"),
    ]
    last_err = None
    for kwargs in try_order:
        try:
            df = pd.read_csv(path, **kwargs)
            # 'Unnamed:...' 열 제거
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]
            if df.shape[1] > 1:
                print(f"  -> OK: enc={enc}, sep={'TAB' if sep=='\\t' else sep}, skiprows={header_idx}, shape={df.shape}")
                return df
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f"로드 실패: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# 메인
def main():
    sanitize_filenames(RAW)
    files = [p for p in RAW.iterdir() if p.is_file() and p.suffix.lower()==".csv"]
    if not files:
        listing = [repr(p.name) for p in RAW.iterdir()]
        raise FileNotFoundError(f"raw_data에 .csv 없음. 현재 목록: {listing}")

    # 연도 키로 정렬(파일명에 20xx가 없으면 0)
    def year_key(p: Path) -> int:
        m = re.search(r"(20\d{2})", p.stem)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=year_key)

    print("[발견된 파일들]", [p.name for p in files])

    dfs = []
    for f in files:
        print(f"\n=== {f.name} ===")
        df = robust_read_one(f)
        # 상위 5행/컬럼 표시
        print("[Columns]", list(df.columns))
        print(df.head(5).to_string(index=False))
        dfs.append(df.assign(_source=f.name))

    # 간단 검증: 컬럼 합집합/누락 확인
    all_cols = sorted(set().union(*[set(d.columns) for d in dfs]))
    print("\n[Union of columns]", all_cols)
    for d in dfs:
        miss = [c for c in all_cols if c not in d.columns]
        if miss:
            print(f"- {d._source.iloc[0]} missing:", miss)

    # (다음 스텝에서 병합/리네이밍 진행 예정)
    print("\n미리보기 완료. 다음 스텝에서 컬럼 표준화 & 3개 파일 병합으로 진행합니다.")

if __name__ == "__main__":
    main()

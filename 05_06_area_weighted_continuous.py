from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# ========= 경로 =========
BASE = Path(__file__).resolve().parent
DATA = BASE / "data_processed" / "jeju_modeling_dataset.csv"
PLOT_DIR = BASE / "reports" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 옵션 =========
ROLLING_DAYS = 14       # 연속 곡선 스무딩(일 단위 이동평균). 0이면 스무딩 없음
SUBREGION_TOPN = 6      # 거래수 상위 N개 행정구역
INTERP = True           # 비거래일 보간(표시용)

# ========= 한글 폰트 설정(축/제목 깨짐 방지) =========
def set_korean_font():
    candidates = ["AppleGothic", "NanumGothic", "Malgun Gothic"]
    found = None
    installed = set(f.name for f in font_manager.fontManager.ttflist)
    for c in candidates:
        if c in installed:
            found = c
            break
    if found is None:
        # 최소한 minus 깨짐 방지
        rcParams["axes.unicode_minus"] = False
        print("[알림] 시스템에 한글 폰트(AppleGothic/NanumGothic/Malgun)가 없어 기본 폰트를 사용합니다.")
        print("      macOS라면 'AppleGothic'가 기본적으로 있으므로 VSCode 터미널 재시작을 시도해보세요.")
        return
    rcParams["font.family"] = found
    rcParams["axes.unicode_minus"] = False
    # DPI 조금 올려 저장 화질 개선
    rcParams["figure.dpi"] = 110

set_korean_font()

# ========= 데이터 로드 =========
def load_df() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"모델링용 파일을 찾지 못했습니다: {DATA}")
    df = pd.read_csv(DATA, encoding="utf-8-sig", parse_dates=["날짜"])
    need = ["날짜", "도시", "행정구역", "전용면적_㎡", "가격_만원"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"필수 컬럼 누락: {miss}")
    # 위생
    df = df[(df["전용면적_㎡"] > 0) & (df["가격_만원"] > 0)].copy()
    return df

# ========= 일 단위 면적가중 단가 집계 =========
def daily_weighted(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    면적 가중 ㎡당가격(만원/㎡) = (그 날/그룹 총 가격(만원)) / (그 날/그룹 총 면적(㎡))
    """
    g = df.groupby(group_cols + ["날짜"], as_index=False)
    out = g.apply(lambda x: pd.Series({
        "거래수": len(x),
        "총면적_㎡": x["전용면적_㎡"].sum(),
        "총가격_만원": x["가격_만원"].sum(),
        "면적가중_만원_per_㎡": x["가격_만원"].sum() / x["전용면적_㎡"].sum()
    })).reset_index(drop=True)
    return out

def reindex_and_smooth(df_grp: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """
    그룹별로 날짜를 연속(일 단위) 축으로 만들고, 필요시 보간/이동평균 적용
    """
    pieces = []
    for keys, sub in df_grp.groupby(key_cols):
        sub = sub.sort_values("날짜")
        full_days = pd.date_range(sub["날짜"].min(), sub["날짜"].max(), freq="D")
        sub = sub.set_index("날짜").reindex(full_days)
        sub.index.name = "날짜"
        if INTERP:
            sub["면적가중_만원_per_㎡"] = sub["면적가중_만원_per_㎡"].interpolate("time")
        if ROLLING_DAYS and ROLLING_DAYS > 1:
            sub["면적가중_roll"] = sub["면적가중_만원_per_㎡"].rolling(ROLLING_DAYS, min_periods=1).mean()
        # 키 복원
        if isinstance(keys, tuple):
            for k, v in zip(key_cols, keys):
                sub[k] = v
        else:
            sub[key_cols[0]] = keys
        pieces.append(sub.reset_index())
    return pd.concat(pieces, ignore_index=True)

# ========= 플롯 =========
def plot_lines(df_long: pd.DataFrame, group_col: str, y_col: str, title: str, out_path: Path):
    plt.figure(figsize=(12, 6.5))
    for name, sub in df_long.groupby(group_col):
        sub = sub.sort_values("날짜")
        plt.plot(sub["날짜"], sub[y_col], label=name)
    plt.title(title)
    plt.xlabel("날짜")
    plt.ylabel("만원/㎡")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[저장] {out_path}")

# ========= 비교(도시/행정구역) =========
def compare_cities(df: pd.DataFrame):
    df2 = df[df["도시"].isin(["제주시", "서귀포시"])].copy()
    agg = daily_weighted(df2, ["도시"])
    long = reindex_and_smooth(agg, ["도시"])
    ycol = "면적가중_roll" if (ROLLING_DAYS and "면적가중_roll" in long) else "면적가중_만원_per_㎡"
    title = f"제주시 vs 서귀포시 — 연속(일 단위) 면적가중 ㎡당가격{f' (이동평균 {ROLLING_DAYS}일)' if ROLLING_DAYS else ''}"
    plot_lines(long, "도시", ycol, title, PLOT_DIR / "city_area_weighted_continuous.png")

def pick_top_subregions(df: pd.DataFrame, topn=6) -> list[str]:
    return (df.groupby("행정구역")["가격_만원"].count()
            .sort_values(ascending=False).head(topn).index.tolist())

def compare_subregions(df: pd.DataFrame, selected=None):
    if selected is None:
        selected = pick_top_subregions(df, SUBREGION_TOPN)
    df2 = df[df["행정구역"].isin(selected)].copy()
    agg = daily_weighted(df2, ["행정구역"])
    long = reindex_and_smooth(agg, ["행정구역"])
    ycol = "면적가중_roll" if (ROLLING_DAYS and "면적가중_roll" in long) else "면적가중_만원_per_㎡"
    title = f"행정구역별 — 연속(일 단위) 면적가중 ㎡당가격{f' (이동평균 {ROLLING_DAYS}일)' if ROLLING_DAYS else ''}"
    plot_lines(long, "행정구역", ycol, title, PLOT_DIR / "subregion_area_weighted_continuous.png")

def main():
    df = load_df()
    compare_cities(df)
    compare_subregions(df, selected=None)  # 특정 목록을 보고 싶으면 리스트 전달

if __name__ == "__main__":
    main()

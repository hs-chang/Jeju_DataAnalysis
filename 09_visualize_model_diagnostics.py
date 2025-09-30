# 09_visualize_model_diagnostics.py
# 모델 성능 지표/잔차/예측-실측 대각선, (있으면) 특성중요도까지 시각화
# 실행: python 09_visualize_model_diagnostics.py

import os, re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_data"
INTERIM = ROOT / "data_interim"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for d in (INTERIM, MODELS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

# ---------- 폰트(한글) ----------
def set_korean_font():
    try:
        import platform
        sysname = platform.system().lower()
        if "darwin" in sysname:
            plt.rcParams["font.family"] = "AppleGothic"
        elif "windows" in sysname:
            plt.rcParams["font.family"] = "Malgun Gothic"
        else:
            plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
set_korean_font()

# ---------- 지표 ----------
def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # squared=False 미사용
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE%": mape}

# ---------- 공통 유틸 ----------
def coerce_number(s: pd.Series) -> pd.Series:
    # 숫자/부호/소수점만 남기기 → to_numeric
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^\d\.\-]+", "", regex=True),
        errors="coerce"
    )

def parse_city_region(sigungu: str):
    if not isinstance(sigungu, str):
        return None, None
    s = sigungu.strip()
    if s.startswith("제주시"):
        r = s.replace("제주시", "", 1).strip()
        return "제주시", (r if r else "제주시")
    if s.startswith("서귀포시"):
        r = s.replace("서귀포시", "", 1).strip()
        return "서귀포시", (r if r else "서귀포시")
    tok = s.split()[0] if s else None
    if tok in ("제주시", "서귀포시"):
        r = s.replace(tok, "", 1).strip()
        return tok, (r if r else tok)
    return None, s or None

# ---------- raw → 최소 정제(비상용) ----------
def robust_read_csv(path: Path) -> pd.DataFrame:
    encs = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]
    for enc in encs:
        try:
            df = pd.read_csv(path, encoding=enc, skiprows=15)
            if df.shape[1] == 1:
                df = pd.read_csv(path, encoding=enc, skiprows=15, sep=",", engine="python")
            return df
        except Exception:
            continue
    return pd.read_csv(path, encoding="cp949", errors="ignore", skiprows=15)

def build_from_raw() -> pd.DataFrame:
    csvs = sorted([p for p in RAW.glob("*.csv")])
    if not csvs:
        raise FileNotFoundError(f"raw_data에 CSV가 없습니다: {RAW}")
    print("[raw_data 내 CSV 전체]", [p.name for p in csvs])

    frames = []
    for p in csvs:
        df = robust_read_csv(p)
        print(f"  - {p.name} | shape={df.shape}")
        df.columns = [c.strip() for c in df.columns]

        col_money = next((c for c in df.columns if "거래금액" in c), None)
        col_area  = next((c for c in df.columns if ("전용" in c and "면적" in c) or c=="전용면적"), None)
        col_yymm  = next((c for c in df.columns if "계약년월" in c), None)
        col_dd    = next((c for c in df.columns if c=="계약일"), None)
        col_year  = next((c for c in df.columns if "건축년도" in c), None)
        col_cmpx  = next((c for c in df.columns if "단지명" in c), None)
        col_floor = next((c for c in df.columns if c=="층"), None)
        col_addr  = next((c for c in df.columns if "시군구" in c), None)

        use_cols = [c for c in [col_money, col_area, col_yymm, col_dd, col_year, col_cmpx, col_floor, col_addr] if c]
        sub = df[use_cols].copy()

        if col_money:
            sub[col_money] = coerce_number(sub[col_money])
        if col_area:
            sub[col_area]  = coerce_number(sub[col_area])
        if col_year and col_year in sub:
            sub[col_year]  = coerce_number(sub[col_year])

        if col_yymm:
            sub["년"] = coerce_number(sub[col_yymm].astype(str).str.slice(0, 4))
            sub["월"] = coerce_number(sub[col_yymm].astype(str).str.slice(4, 6))
        else:
            sub["년"] = np.nan; sub["월"] = np.nan
        sub["일"] = coerce_number(sub[col_dd]) if (col_dd and col_dd in sub) else 1
        sub["날짜"] = pd.to_datetime(dict(year=sub["년"], month=sub["월"], day=sub["일"]), errors="coerce")

        if col_addr and col_addr in sub:
            parsed = sub[col_addr].apply(parse_city_region)
            sub["도시"] = parsed.apply(lambda t: t[0])
            sub["행정구역"] = parsed.apply(lambda t: t[1])
        else:
            sub["도시"] = None; sub["행정구역"] = None

        if col_year and col_year in sub:
            sub["경과년수"] = sub["년"] - sub[col_year]
        else:
            sub["경과년수"] = np.nan

        sub = sub.rename(columns={
            col_money: "가격_만원",
            col_area:  "전용면적_㎡",
            col_year:  "건축년도",
            col_cmpx:  "단지명",
            col_floor: "층",
        })

        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)

    for c in ["도시", "행정구역", "단지명"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    out["가격_만원"]  = pd.to_numeric(out.get("가격_만원"), errors="coerce")
    out["전용면적_㎡"] = pd.to_numeric(out.get("전용면적_㎡"), errors="coerce")
    if "거래유형" not in out.columns:
        out["거래유형"] = "미상"

    path = INTERIM / "jeju_ready_for_train.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[저장] {path} (rows={len(out):,})")
    return out

# ---------- 정제 데이터 로드(+견고 파싱) ----------
def load_df() -> pd.DataFrame:
    cand = [
        INTERIM / "jeju_ready_for_train.csv",
        INTERIM / "jeju_clean.csv",
        INTERIM / "jeju_date_price_enriched.csv",
    ]
    for p in cand:
        if p.exists():
            print(f"[로드] {p} (rows={sum(1 for _ in open(p, 'r', encoding='utf-8', errors='ignore'))-1:,})")
            df = pd.read_csv(p, encoding="utf-8", engine="python")
            df.columns = [c.strip() for c in df.columns]

            # 가격 컬럼 확보
            price_col = None
            for c in ["가격_만원", "거래금액(만원)", "거래금액", "거래금액_만원"]:
                if c in df.columns:
                    price_col = c; break
            if price_col is None:
                # 텍스트 검색
                price_col = next((c for c in df.columns if "거래금액" in c or "가격" in c), None)

            if price_col and price_col != "가격_만원":
                df["가격_만원"] = coerce_number(df[price_col])

            if "가격_만원" in df.columns:
                df["가격_만원"] = coerce_number(df["가격_만원"])

            # 면적 컬럼 확보
            area_col = None
            for c in ["전용면적_㎡", "전용면적(㎡)", "전용면적"]:
                if c in df.columns:
                    area_col = c; break
            if area_col and area_col != "전용면적_㎡":
                df["전용면적_㎡"] = coerce_number(df[area_col])
            if "전용면적_㎡" in df.columns:
                df["전용면적_㎡"] = coerce_number(df["전용면적_㎡"])

            # 도시/행정구역 보정
            if "도시" in df.columns:
                df["도시"] = df["도시"].astype(str).str.strip()
            if "행정구역" in df.columns:
                df["행정구역"] = df["행정구역"].astype(str).str.strip()

            # 없는 경우 시군구에서 생성 시도
            if ("도시" not in df.columns) or ("행정구역" not in df.columns):
                if "시군구" in df.columns:
                    pr = df["시군구"].apply(parse_city_region)
                    df["도시"] = pr.apply(lambda t: t[0])
                    df["행정구역"] = pr.apply(lambda t: t[1])

            # 거래유형 없으면 '미상'
            if "거래유형" not in df.columns:
                print("[알림] 선택 컬럼 '거래유형' 없음 → '미상'으로 생성")
                df["거래유형"] = "미상"

            # 날짜 파생 보정
            if "계약년월" in df.columns and (("년" not in df.columns) or ("월" not in df.columns)):
                df["년"] = coerce_number(df["계약년월"].astype(str).str.slice(0,4))
                df["월"] = coerce_number(df["계약년월"].astype(str).str.slice(4,6))

            return df

    print("[알림] 정제 CSV를 찾지 못해 raw_data에서 즉석 생성합니다.")
    return build_from_raw()

# ---------- 모델/피처 ----------
def load_models():
    lin_path = MODELS / "linreg.joblib"
    rf_path  = MODELS / "rf.joblib"
    if not lin_path.exists() or not rf_path.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다.\n - {lin_path}\n - {rf_path}\n"
            f"먼저 08_train_full.py 를 실행해 모델을 저장해 주세요."
        )
    lin = joblib.load(lin_path)
    rf  = joblib.load(rf_path)
    print("[모델 로드] linreg.joblib")
    print("[모델 로드] rf.joblib")
    return lin, rf

EXPECTED_COLS = [
    "전용면적_㎡", "층", "건축년도", "경과년수",
    "도시", "행정구역", "단지명", "년", "월", "거래유형",
    "만원_per_㎡",
]

def pick_feature_cols(model, df: pd.DataFrame):
    cols = list(getattr(model, "feature_names_in_", [])) or [c for c in EXPECTED_COLS if c in df.columns]
    # 범주형 결측 채움
    for cat in ["도시", "행정구역", "단지명", "거래유형"]:
        if cat in df.columns:
            df[cat] = df[cat].fillna("미상").astype(str).str.strip()
    return cols

# ---------- 플롯 ----------
def save_parity(y_true, y_pred, title, save_path: Path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.35)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linewidth=2)
    m = metrics(y_true, y_pred)
    plt.title(f"{title}\nR²={m['R2']:.3f} | MAE={m['MAE']:.0f}만원 | RMSE={m['RMSE']:.0f}만원 | MAPE={m['MAPE%']:.1f}%")
    plt.xlabel("실거래가 (만원)"); plt.ylabel("예측가 (만원)")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[저장] {save_path}")

def save_residuals(y_true, y_pred, title, save_path: Path):
    res = y_true - y_pred
    plt.figure(figsize=(7,4))
    plt.scatter(y_pred, res, s=10, alpha=0.35)
    plt.axhline(0, linewidth=2)
    plt.title(f"{title} — 잔차(실측-예측) vs 예측")
    plt.xlabel("예측가 (만원)"); plt.ylabel("잔차 (만원)")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[저장] {save_path}")

def save_error_hist(y_true, y_pred, title, save_path: Path):
    err = y_pred - y_true
    plt.figure(figsize=(7,4))
    plt.hist(err, bins=40)
    plt.title(f"{title} — 예측오차 분포 (예측-실측, 만원)")
    plt.xlabel("오차(만원)"); plt.ylabel("빈도")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[저장] {save_path}")

def try_plot_perm_importance():
    cand = [
        REPORTS / "perm_importance_rf.csv",
        REPORTS / "perm_importance_ridge.csv",
    ]
    for p in cand:
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8")
            req = {"feature", "importance_mean", "importance_std"}
            if not req.issubset(df.columns): continue
            df = df.sort_values("importance_mean", ascending=True).tail(20)
            plt.figure(figsize=(8, max(4, 0.35*len(df))))
            plt.barh(df["feature"], df["importance_mean"], xerr=df["importance_std"])
            plt.title(f"Permutation Importance (상위 {len(df)}) — {p.name.split('_')[-1].split('.')[0].upper()}")
            plt.xlabel("평균 중요도 (MAE 증가량)")
            plt.tight_layout()
            out = REPORTS / f"perm_importance_top_{len(df)}_{p.stem}.png"
            plt.savefig(out, dpi=150); plt.close()
            print(f"[저장] {out}")
            return True
    print("[안내] 저장된 permutation importance CSV를 찾지 못했습니다. (건너뜀)")
    return False

# ---------- 메인 ----------
def main():
    df = load_df()

    # 가격/면적이 전부 NaN이면 대체 시도
    if "가격_만원" not in df.columns or df["가격_만원"].notna().sum() == 0:
        # 만원_per_㎡ * 전용면적_㎡로 복원 시도
        if "만원_per_㎡" in df.columns and "전용면적_㎡" in df.columns:
            df["가격_만원"] = coerce_number(df["만원_per_㎡"]) * coerce_number(df["전용면적_㎡"])
        else:
            # raw에서 재생성
            print("[경고] 가격 컬럼이 전부 NaN → raw_data에서 최소 정제본 재생성")
            df = build_from_raw()

    if "전용면적_㎡" not in df.columns or df["전용면적_㎡"].notna().sum() == 0:
        # 전용면적 후보에서 다시 파싱
        for c in ["전용면적(㎡)", "전용면적"]:
            if c in df.columns:
                df["전용면적_㎡"] = coerce_number(df[c])
                break

    # 수치형 보정
    df["가격_만원"]  = coerce_number(df.get("가격_만원"))
    df["전용면적_㎡"] = coerce_number(df.get("전용면적_㎡"))

    # 핵심 결측 제거
    data = df.dropna(subset=["가격_만원", "전용면적_㎡"]).copy()
    print("[체크] 핵심 결측 제거 후 행수:", len(data))

    # 도시 필터(가능할 때만)
    if "도시" in data.columns and data["도시"].notna().any():
        mask_city = data["도시"].isin(["제주시", "서귀포시"])
        if mask_city.any():
            data = data.loc[mask_city].copy()
            print("[체크] 도시 필터 후 행수:", len(data))
        else:
            print("[안내] '제주시/서귀포시'가 없어 도시 필터 건너뜀")
    else:
        print("[안내] '도시' 컬럼이 없거나 전부 결측 → 도시 필터 건너뜀")

    if data.empty:
        raise ValueError("유효한 학습/평가 데이터가 0건입니다. CSV/전처리를 확인하세요.")

    # 타겟/입력
    y = data["가격_만원"].astype(float).values

    lin, rf = load_models()
    feat_cols = pick_feature_cols(rf, data)  # 동일 파이프라인 가정
    if not feat_cols:
        raise ValueError("입력 피처 컬럼을 찾지 못했습니다. EXPECTED_COLS를 확인하세요.")

    X = data[feat_cols].copy()

    # 범주형 결측 채움(이중 안전)
    for cat in ["도시", "행정구역", "단지명", "거래유형"]:
        if cat in X.columns:
            X[cat] = X[cat].fillna("미상").astype(str).str.strip()

    # 수치형 결측 드랍
    num_cols = [c for c in ["전용면적_㎡", "층", "건축년도", "경과년수", "년", "월"] if c in X.columns]
    keep = X[num_cols].apply(lambda s: s.notna()).all(axis=1) if num_cols else pd.Series(True, index=X.index)
    keep &= pd.Series(~np.isnan(y), index=X.index)
    X, y = X.loc[keep], y[keep]
    print("[체크] 최종 학습/평가 대상 행수:", len(X))

    # 분할
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # 예측
    y_pred_lin = lin.predict(X_te)
    y_pred_rf  = rf.predict(X_te)

    m_lin = metrics(y_te, y_pred_lin)
    m_rf  = metrics(y_te, y_pred_rf)

    print("\n== Ridge / Test 결과 ==")
    print(f"   R2: {m_lin['R2']:.3f}")
    print(f"  MAE: {m_lin['MAE']:.3f} 만원")
    print(f" RMSE: {m_lin['RMSE']:.3f} 만원")
    print(f"MAPE%: {m_lin['MAPE%']:.3f} %")

    print("\n== RandomForest / Test 결과 ==")
    print(f"   R2: {m_rf['R2']:.3f}")
    print(f"  MAE: {m_rf['MAE']:.3f} 만원")
    print(f" RMSE: {m_rf['RMSE']:.3f} 만원")
    print(f"MAPE%: {m_rf['MAPE%']:.3f} %")

    # 시각화
    save_parity(y_te, y_pred_lin, "Ridge — 실거래가 vs 예측가", REPORTS / "diag_parity_ridge.png")
    save_parity(y_te, y_pred_rf,  "RandomForest — 실거래가 vs 예측가", REPORTS / "diag_parity_rf.png")

    save_residuals(y_te, y_pred_lin, "Ridge", REPORTS / "diag_residuals_ridge.png")
    save_residuals(y_te, y_pred_rf,  "RandomForest", REPORTS / "diag_residuals_rf.png")

    save_error_hist(y_te, y_pred_lin, "Ridge", REPORTS / "diag_error_hist_ridge.png")
    save_error_hist(y_te, y_pred_rf,  "RandomForest", REPORTS / "diag_error_hist_rf.png")

    try_plot_perm_importance()
    print("\n[완료] 리포트 폴더에 이미지/결과가 저장되었습니다.")

if __name__ == "__main__":
    main()

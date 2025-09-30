# 08_train_full.py

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")

# ---------- 경로 ----------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data_processed" / "jeju_modeling_dataset.csv"
MODEL_DIR = BASE / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = BASE / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 유틸 ----------
def rmse(y_true, y_pred) -> float:
    # sklearn 1.6 + py3.13 호환: squared=False 대신 직접 루트
    return float(mean_squared_error(y_true, y_pred) ** 0.5)

def safe_mape(y_true, y_pred, eps: float = 1e-6, clip_min: float = 100.0) -> float:
    y = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    mask = y >= clip_min
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y[mask] - p[mask]) / (y[mask] + eps))) * 100)

def feature_names_from_ct(ct: ColumnTransformer) -> list[str]:
    """
    ColumnTransformer에서 최종 특성명 추출 (num/onehot 구분 반영)
    """
    out = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            fn = list(trans.get_feature_names_out(cols))
        else:
            # 파이프라인이면 마지막 스텝에서 추출 시도
            if hasattr(trans, "steps"):
                last = trans.steps[-1][1]
                if hasattr(last, "get_feature_names_out"):
                    fn = list(last.get_feature_names_out(cols))
                else:
                    fn = list(cols)
            else:
                fn = list(cols)
        # 'num__', 'cat__' 접두어 정리
        fn = [f.replace("num__", "").replace("cat__", "") for f in fn]
        out.extend(fn)
    return out

# ---------- 데이터 ----------
def load_df() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA}")
    df = pd.read_csv(DATA, encoding="utf-8-sig", parse_dates=["날짜"])

    # 필수/선택 컬럼 정의
    need = ["날짜","도시","행정구역","단지명","전용면적_㎡","층","건축년도","가격_만원"]
    opt  = ["거래유형"]

    miss_need = [c for c in need if c not in df.columns]
    if miss_need:
        raise ValueError(f"필수 컬럼 누락: {miss_need}")

    # 선택 컬럼이 없으면 기본값으로 생성
    for c in opt:
        if c not in df.columns:
            df[c] = "미상"
            print(f"[알림] 선택 컬럼 '{c}' 없음 → '미상'으로 생성")

    # 기본 위생: 숫자형 강제 변환
    for c in ["전용면적_㎡","층","건축년도","가격_만원"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 유효 행만
    df = df[(df["가격_만원"] > 0) & (df["전용면적_㎡"] > 0)].copy()
    df = df.drop_duplicates()

    # 파생
    df["년"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    # 건축년도 결측 대비
    build_year_median = np.nanmedian(df["건축년도"].values)
    df["경과년수"] = np.maximum(0, df["년"] - np.where(np.isnan(df["건축년도"]), build_year_median, df["건축년도"]))

    return df


def split_by_time(df: pd.DataFrame, cut: str = "2025-01-01"):
    cut_ts = pd.Timestamp(cut)
    train = df[df["날짜"] < cut_ts].copy()
    test  = df[df["날짜"] >= cut_ts].copy()
    if len(train)==0 or len(test)==0:
        raise ValueError(f"시간 분할에 실패 (cut={cut}). train={len(train)}, test={len(test)}")
    return train, test

# ---------- 전처리 ----------
def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
    ])
    pre = ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols)
    ])
    return pre

# ---------- 모델 ----------
def build_models(pre: ColumnTransformer):
    ridge = Pipeline([
        ("preprocess", pre),
        ("model", TransformedTargetRegressor(
            regressor=Ridge(alpha=3.0, random_state=42),
            func=np.log1p, inverse_func=np.expm1
        ))
    ])
    rf = Pipeline([
        ("preprocess", pre),
        ("model", TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                max_features="sqrt", random_state=42, n_jobs=1  # py3.13 joblib 이슈
            ),
            func=np.log1p, inverse_func=np.expm1
        ))
    ])
    return ridge, rf

def small_gridsearch(name: str, pipe: Pipeline, X_tr, y_tr):
    # 간단 튜닝(시간순 교차검증)
    tscv = TimeSeriesSplit(n_splits=4)
    if name == "Ridge":
        param_grid = {
            "model__regressor__alpha": [0.5, 1.0, 3.0, 10.0]
        }
        scoring = "neg_mean_absolute_error"
    else:  # RF
        param_grid = {
            "model__regressor__n_estimators": [200, 400],
            "model__regressor__min_samples_leaf": [1, 2, 4],
            "model__regressor__max_depth": [None, 12, 20]
        }
        scoring = "neg_mean_absolute_error"
    gs = GridSearchCV(
        pipe, param_grid=param_grid, cv=tscv,
        scoring=scoring, n_jobs=1, verbose=1
    )
    gs.fit(X_tr, y_tr)
    print(f"[{name}] best params:", gs.best_params_, "| best score(MAE↓):", -gs.best_score_)
    return gs.best_estimator_

# ---------- 평가 ----------
def evaluate(name: str, pipe: Pipeline, X_te, y_te) -> dict:
    pred = pipe.predict(X_te)  # TTR로 원 스케일 결과
    res = {
        "model": name,
        "R2": r2_score(y_te, pred),
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": rmse(y_te, pred),
        "MAPE%": safe_mape(y_te, pred)
    }
    print(f"\n== {name} / Test 결과 ==")
    for k in ["R2","MAE","RMSE","MAPE%"]:
        v = res[k]
        unit = "" if k=="R2" else (" %" if k=="MAPE%" else " 만원")
        print(f"{k:>5}: {v:,.3f}{unit}")
    return res, pred

def group_errors(df_test: pd.DataFrame, pred: np.ndarray):
    out = df_test.copy()
    out["예측가_만원"] = pred
    out["오차_만원"] = out["예측가_만원"] - out["가격_만원"]
    out["절대오차_만원"] = np.abs(out["오차_만원"])

    def summarize(g):
        return pd.Series({
            "n": len(g),
            "MAE": mean_absolute_error(g["가격_만원"], g["예측가_만원"]),
            "RMSE": rmse(g["가격_만원"], g["예측가_만원"])
        })

    # 도시별
    city = out.groupby("도시").apply(summarize).reset_index()
    # 행정구역별 상위 10
    top_admin = (out.groupby("행정구역").apply(summarize)
                  .sort_values("n", ascending=False).head(10).reset_index())
    # 면적 버킷
    bins = [0, 40, 60, 85, 102, 135, 9999]
    labels = ["~40", "40~60", "60~85", "85~102", "102~135", "135~"]
    out["면적구간"] = pd.cut(out["전용면적_㎡"], bins=bins, labels=labels, right=False)
    size = out.groupby("면적구간").apply(summarize).reset_index()

    print("\n[도시별 오차 요약]\n", city.to_string(index=False))
    print("\n[행정구역별(Top10) 오차 요약]\n", top_admin.to_string(index=False))
    print("\n[면적구간별 오차 요약]\n", size.to_string(index=False))

# ---------- 중요도 ----------
def run_permutation_importance(name: str, model, X: pd.DataFrame, y: pd.Series, topk: int = 20):
    """
    Pipeline(전처리+모델)에 대해 permutation importance 실행.
    ※ 중요: 피처 이름은 전처리 후가 아니라 X의 원본 컬럼을 그대로 사용해야 길이가 맞음.
    """
    from sklearn.inspection import permutation_importance

    # macOS + py3.13에서 loky 이슈 회피용으로 n_jobs=1 권장
    pi = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=42,
        n_jobs=1,
        scoring="neg_mean_absolute_error"
    )

    cols = list(X.columns)  # ✅ permutation에 실제 들어간 원본 컬럼명
    n = min(len(cols), pi.importances_mean.shape[0])

    imp = (
        pd.DataFrame({
            "feature": cols[:n],
            "importance_mean": pi.importances_mean[:n],
            "importance_std": pi.importances_std[:n]
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    top = imp.head(topk)
    print(f"\n=== 중요 특성 Top {len(top)} — {name} ===")
    print(top.to_string(index=False))

    outdir = Path(__file__).resolve().parent / "reports"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"perm_importance_{name.lower()}.csv"
    imp.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {out}")


# ---------- 메인 ----------
def main():
    np.random.seed(42)
    df = load_df()
    train, test = split_by_time(df, cut="2025-01-01")

    y_tr = train["가격_만원"].values
    y_te = test["가격_만원"].values
    X_tr = train.drop(columns=["가격_만원"])
    X_te = test.drop(columns=["가격_만원"])

    num_cols = ["전용면적_㎡","층","건축년도","경과년수","년","월"]

    # '거래유형'은 있으면 포함, 없으면 제외
    base_cat = ["도시","행정구역","단지명"]
    cat_cols = base_cat + (["거래유형"] if "거래유형" in df.columns else [])

    pre = build_preprocessor(num_cols, cat_cols)

    ridge, rf = build_models(pre)

    # --- (옵션) 소형 그리드서치로 파라미터 경정렬 ---
    print("\n[튜닝] Ridge …")
    ridge = small_gridsearch("Ridge", ridge, X_tr, y_tr)
    print("\n[튜닝] RandomForest …")
    rf = small_gridsearch("RF", rf, X_tr, y_tr)

    # --- 최종 학습 (Train 전체) ---
    ridge.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)

    # --- 평가 (Test) ---
    res_ridge, pred_ridge = evaluate("Ridge", ridge, X_te, y_te)
    res_rf,    pred_rf    = evaluate("RandomForest", rf,    X_te, y_te)

    # --- 그룹별 오차 진단 ---
    print("\n[오차 진단: Ridge]")
    group_errors(test, pred_ridge)
    print("\n[오차 진단: RF]")
    group_errors(test, pred_rf)

    # --- Permutation Importance (Test셋 기준) ---
    run_permutation_importance("Ridge", ridge, X_te, y_te, topk=20)
    run_permutation_importance("RF", rf, X_te, y_te, topk=20)

    # --- 결과/모델 저장 ---
    preds_path = MODEL_DIR / "predictions_test.csv"
    pd.DataFrame({
        "날짜": test["날짜"].values,
        "도시": test["도시"].values,
        "행정구역": test["행정구역"].values,
        "단지명": test["단지명"].values,
        "전용면적_㎡": test["전용면적_㎡"].values,
        "실거래가_만원": y_te,
        "예측가_Ridge_만원": pred_ridge,
        "예측가_RF_만원": pred_rf
    }).to_csv(preds_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {preds_path}")

    ridge_path = MODEL_DIR / "ridge_ttr.joblib"
    rf_path = MODEL_DIR / "rf_ttr.joblib"
    joblib.dump(ridge, ridge_path)
    joblib.dump(rf, rf_path)
    print(f"[저장] {ridge_path}, {rf_path}")

if __name__ == "__main__":
    main()

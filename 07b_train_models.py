# 07b_train_models.py — 선형회귀 & 랜덤포레스트 학습/평가 + Permutation Importance(원본 특성 기준)
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
from joblib import parallel_backend


BASE = Path(__file__).resolve().parent
DATA = BASE / "data_processed" / "jeju_modeling_dataset.csv"
REP  = BASE / "reports"
MOD  = BASE / "models"
REP.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_RATIO = 0.2  # 최근 20%를 테스트로 사용(시간 누출 방지)

def rmse(y_true, y_pred):
    # squared=False 호환 이슈 회피: 버전 무관하게 동작
    return np.sqrt(mean_squared_error(y_true, y_pred))

def time_order_split(df, test_ratio=0.2):
    """날짜 기준 정렬 후 뒤쪽 비율을 테스트로 분리"""
    df = df.sort_values("날짜").reset_index(drop=True)
    n = len(df)
    cut = int(np.floor(n * (1 - test_ratio)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def build_preprocessors(numeric_features, categorical_features):
    # 선형회귀: 수치 표준화 + 원핫
    num_linear = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_common = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preproc_linear = ColumnTransformer(
        transformers=[
            ("num", num_linear, numeric_features),
            ("cat", cat_common, categorical_features),
        ]
    )

    # 랜덤포레스트: 표준화 불필요 → 결측만 처리 + 원핫
    num_rf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preproc_rf = ColumnTransformer(
        transformers=[
            ("num", num_rf, numeric_features),
            ("cat", cat_common, categorical_features),
        ]
    )
    return preproc_linear, preproc_rf

def fit_and_eval(model_name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {
        "model": model_name,
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": rmse(y_te, pred),
        "R2": r2_score(y_te, pred),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
    return pipe, pred, metrics

def save_perm_importance(pipeline, X_te, y_te, tag):
    """
    Permutation Importance — 원본 입력 특성 기준(원-핫 확장 전 칼럼들).
    파이프라인에 대해 X_te(DataFrame)를 넣으면 원본 칼럼 단위로 섞어 평가합니다.
    """
    with parallel_backend("threading"):
        r = permutation_importance(
            pipeline, X_te, y_te,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
        )
    r = permutation_importance(
        pipeline, X_te, y_te,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    names = list(X_te.columns)
    k = min(len(names), len(r.importances_mean))
    df_imp = pd.DataFrame({
        "feature": names[:k],
        "importance_mean": r.importances_mean[:k],
        "importance_std": r.importances_std[:k],
    }).sort_values("importance_mean", ascending=False)
    out_path = REP / f"perm_importance_{tag}.csv"
    df_imp.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n=== 중요 특성 Top 15 — {tag} ===")
    print(df_imp.head(15).to_string(index=False))
    print(f"[저장] {out_path}")

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"모델링용 데이터가 없습니다: {DATA}\n먼저 07a_build_model_dataset.py를 실행해 주세요.")

    df = pd.read_csv(DATA, encoding="utf-8-sig", parse_dates=["날짜"])

    # 타깃과 특성 정의
    target = "가격_만원"
    numeric_features = ["전용면적_㎡", "층", "건축년도", "경과년수", "년", "월"]
    categorical_features = ["도시", "행정구역", "단지명"]

    # 필수 컬럼 확인
    needed = ["날짜", target] + numeric_features + categorical_features
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 위생: 핵심 결측/0 제거
    df = df.dropna(subset=["날짜", target, "전용면적_㎡", "도시"])
    df = df[(df["전용면적_㎡"] > 0) & (df[target] > 0)]

    # 시간순 분할
    train_df, test_df = time_order_split(df, TEST_RATIO)
    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]
    X_test  = test_df[numeric_features + categorical_features]
    y_test  = test_df[target]

    # 전처리기
    preproc_linear, preproc_rf = build_preprocessors(numeric_features, categorical_features)

    # 1) 선형회귀
    pipe_lr = Pipeline(steps=[
        ("preproc", preproc_linear),
        ("model", LinearRegression())
    ])
    pipe_lr, pred_lr, m_lr = fit_and_eval("LinearRegression", pipe_lr, X_train, y_train, X_test, y_test)

    # 2) 랜덤포레스트
    pipe_rf = Pipeline(steps=[
        ("preproc", preproc_rf),
        ("model", RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=1
        ))
    ])
    pipe_rf, pred_rf, m_rf = fit_and_eval("RandomForest", pipe_rf, X_train, y_train, X_test, y_test)

    # 성능 저장/표시
    metrics_df = pd.DataFrame([m_lr, m_rf]).sort_values("RMSE")
    metrics_csv = REP / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print("\n=== 성능 지표(테스트) ===")
    print(metrics_df.to_string(index=False))
    print(f"[저장] {metrics_csv}")

    # 중요도 저장
    save_perm_importance(pipe_lr, X_test, y_test, "linreg")
    save_perm_importance(pipe_rf, X_test, y_test, "rf")

    # 모델 저장
    joblib.dump(pipe_lr, MOD / "linreg.joblib")
    joblib.dump(pipe_rf, MOD / "rf.joblib")
    print(f"\n[저장] {MOD/'linreg.joblib'}, {MOD/'rf.joblib'}")

if __name__ == "__main__":
    main()

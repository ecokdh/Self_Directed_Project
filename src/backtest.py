import pandas as pd


def rolling_backtest(
    data: pd.DataFrame,
    model_fn,
    predict_fn,
    n_windows: int = 8,
    horizon: int = 24,
    train_mode: str = "expanding",
    sliding_train_len: int = 8760,
    step_size: int = None,
) -> pd.DataFrame:

    if train_mode not in ("expanding", "sliding"):
        raise ValueError(f"unknown train_mode: {train_mode}")

    if step_size is None:
        step_size = horizon

    n = len(data)
    last_origin = n - horizon
    origins = [last_origin - step_size * (n_windows - 1 - i) for i in range(n_windows)]

    min_train_len = sliding_train_len if train_mode == "sliding" else 1
    if origins[0] - min_train_len < 0:
        raise ValueError(
            f"origin이 너무 이릅니다 (첫 origin={origins[0]}, 필요한 최소 학습 길이={min_train_len}). "
            "n_windows를 줄이거나 horizon/sliding_train_len/step_size를 조정하십시오."
        )

    chunks = []
    for w, origin in enumerate(origins):
        if train_mode == "expanding":
            train_df = data.iloc[:origin]
        else:
            train_df = data.iloc[origin - sliding_train_len: origin]
        test_df = data.iloc[origin: origin + horizon]

        model = model_fn(train_df)
        pred_df = predict_fn(model, horizon)

        for bid in data.columns:
            chunks.append(pd.DataFrame({
                "window": w,
                "building": bid,
                "timestamp": test_df.index,
                "y_true": test_df[bid].to_numpy(),
                "y_pred": pred_df[bid].to_numpy(),
            }))

    return pd.concat(chunks, ignore_index=True)

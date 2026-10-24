"""
Kaggle: House Prices - Advanced Regression Techniques
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Boilerplate baseline, following D2L 5.7:
preprocessing (standardize, fill NA with 0, one-hot), log labels, MLP,
k-fold cross-validation, train on all data, write submission.csv.

Knobs to play with are in CONFIG.

Usage:
    python Kaggle_House_Prices.py                                  # raw Kaggle data
    python Kaggle_House_Prices.py --train tr.parquet --test te.parquet   # your own frames

--train / --test take dataframes saved from a notebook (.parquet, .pkl or .csv).
train needs Id + SalePrice columns, test needs Id. They still go through
preprocess(), so they can be cleaned-but-not-encoded or already fully numeric.
Prefer .parquet / .pkl over .csv: csv loses dtypes (e.g. MSSubClass as str -> int).

From a notebook you can also skip files: main(train_df, test_df).
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parent / "data"
# Same train.csv / test.csv as the Kaggle competition, mirrored by the D2L book.
# Only downloaded when missing from DATA_DIR (e.g. on a fresh clone).
URLS = {
    "train.csv": "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv",
    "test.csv": "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv",
}

CONFIG = {
    "k": 5,
    "epochs": 100,
    "lr": 0.01,
    "weight_decay": 0.0,
    "batch_size": 64,
    "hidden": [],        # [] = linear regression; e.g. [256] adds a hidden layer
    "dropout": 0.0,
}


def load_raw():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if not (DATA_DIR / name).exists():
            urlretrieve(url, DATA_DIR / name)
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


def read_frame(path):
    """Load a dataframe saved from a notebook; format picked by file extension."""
    path = Path(path)
    readers = {".parquet": pd.read_parquet, ".pkl": pd.read_pickle,
               ".pickle": pd.read_pickle, ".csv": pd.read_csv}
    if path.suffix not in readers:
        raise ValueError(f"{path}: expected one of {', '.join(readers)}")
    return readers[path.suffix](path)


def check_columns(train, test):
    for name, df, needed in (("train", train, ["Id", "SalePrice"]), ("test", test, ["Id"])):
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{name} dataframe is missing column(s): {missing}")


def build_features(train, test):
    """Standardize numbers, fill NA with 0 (= the mean), one-hot categoricals.
    Returns train rows then test rows in one frame, so column names stay available."""
    features = pd.concat([train.drop(columns=["Id", "SalePrice"]),
                          test.drop(columns=["Id"])])

    numeric = features.select_dtypes(include="number").columns
    features[numeric] = features[numeric].apply(lambda x: (x - x.mean()) / x.std())
    features[numeric] = features[numeric].fillna(0)
    return pd.get_dummies(features, dummy_na=True).astype("float32")


def preprocess(train, test):
    features = build_features(train, test)

    n_train = len(train)
    X_train = torch.tensor(features[:n_train].values)
    X_test = torch.tensor(features[n_train:].values)
    y_train = torch.log(torch.tensor(train["SalePrice"].values, dtype=torch.float32)).reshape(-1, 1)

    # pd.DataFrame(X_train.numpy()).to_csv(Path(__file__).with_name("X_train.csv"))
    # pd.DataFrame(y_train.numpy()).to_csv(Path(__file__).with_name("y_train.csv"))
    return X_train, y_train, X_test


def build_model(n_inputs):
    layers, width = [], n_inputs
    for h in CONFIG["hidden"]:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(CONFIG["dropout"])]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def log_rmse(model, X, y_log):
    """RMSE between log prices (y is already log, the model predicts log)."""
    model.eval()
    with torch.no_grad():
        return torch.sqrt(((model(X) - y_log) ** 2).mean()).item()


def train(X, y, X_val=None, y_val=None):
    model = build_model(X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])
    loader = DataLoader(TensorDataset(X, y), batch_size=CONFIG["batch_size"], shuffle=True)
    history = {"train": [], "val": []}

    for _ in range(CONFIG["epochs"]):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(Xb), yb).backward()
            optimizer.step()

        history["train"].append(log_rmse(model, X, y))
        if X_val is not None:
            history["val"].append(log_rmse(model, X_val, y_val))

    train_err = history["train"][-1]
    val_err = history["val"][-1] if X_val is not None else None
    return model, train_err, val_err, history


def k_fold(X, y):
    k = CONFIG["k"]
    fold = len(X) // k
    train_errs, val_errs, histories = [], [], []
    for i in range(k):
        val_idx = torch.arange(i * fold, (i + 1) * fold)
        mask = torch.ones(len(X), dtype=torch.bool)
        mask[val_idx] = False
        _, tr, va, history = train(X[mask], y[mask], X[val_idx], y[val_idx])
        train_errs.append(tr)
        val_errs.append(va)
        histories.append(history)
        best = min(range(len(history["val"])), key=history["val"].__getitem__)
        print(f"fold {i + 1}: train log rmse {tr:.4f}, val log rmse {va:.4f} "
              f"(lowest val {history['val'][best]:.4f} at epoch {best + 1})")
    print(f"average: train {sum(train_errs) / k:.4f}, val {sum(val_errs) / k:.4f}")
    return histories


def plot_learning_curves(histories):
    """Thin lines: each fold. Thick lines: average over folds."""
    epochs = range(1, len(histories[0]["train"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, color in (("train", "tab:blue"), ("val", "tab:orange")):
        curves = torch.tensor([h[name] for h in histories])
        for curve in curves:
            ax.plot(epochs, curve, color=color, alpha=0.25, linewidth=1)
        ax.plot(epochs, curves.mean(dim=0), color=color, linewidth=2.5,
                label=f"{name} (average over folds)")

    ax.set(title="Learning curves", xlabel="epoch", ylabel="log RMSE", yscale="log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    final_train = sum(h["train"][-1] for h in histories) / len(histories)
    final_val = sum(h["val"][-1] for h in histories) / len(histories)
    info = "\n".join([f"{key}: {value}" for key, value in CONFIG.items()]
                     + ["", f"final train: {final_train:.4f}", f"final val:   {final_val:.4f}"])
    ax.text(0.98, 0.80, info, transform=ax.transAxes, ha="right", va="top",
            multialignment="left", family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    return fig


def main(train_df=None, test_df=None):
    """Pass both frames to train on your own processed data; omit both for raw Kaggle data."""
    torch.manual_seed(0)
    if train_df is None and test_df is None:
        train_df, test_df = load_raw()
    elif train_df is None or test_df is None:
        raise ValueError("pass both train_df and test_df, or neither")
    check_columns(train_df, test_df)
    X_train, y_train, X_test = preprocess(train_df, test_df)
    print(f"train {tuple(X_train.shape)}, test {tuple(X_test.shape)}, config {CONFIG}")

    histories = k_fold(X_train, y_train)

    model, train_err, _, _ = train(X_train, y_train)
    print(f"final model on all training data: train log rmse {train_err:.4f}")

    model.eval()
    with torch.no_grad():
        prices = torch.exp(model(X_test)).squeeze(1).numpy()
    out = Path(__file__).with_name("submission.csv")
    pd.DataFrame({"Id": test_df["Id"], "SalePrice": prices}).to_csv(out, index=False)
    print(f"wrote {out}")

    plot_learning_curves(histories)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle house prices: k-fold + submission")
    parser.add_argument("--train", help="processed train dataframe (.parquet/.pkl/.csv), with Id + SalePrice")
    parser.add_argument("--test", help="processed test dataframe (.parquet/.pkl/.csv), with Id")
    args = parser.parse_args()
    if bool(args.train) != bool(args.test):
        parser.error("--train and --test must be given together")

    if args.train:
        main(read_frame(args.train), read_frame(args.test))
    else:
        main()

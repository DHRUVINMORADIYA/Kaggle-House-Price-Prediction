"""
Trace how every weight and the bias move during training, and which training
rows (by Id) pushed them.

Trains the linear model (CONFIG["hidden"] = []) once on all training rows with
plain SGD (no momentum, no weight decay). For plain SGD, one update is exactly

    update = -lr * mean_i(g_i)    over the rows i in the batch

so row i's push on the parameters in that step is -lr * g_i / batch_size, and the
pushes add up to the real update. g_i comes from torch.func (vmap over grad), and
the script checks that the pushes sum to the real update on every step.

Usage:
    python trace_training.py                                    # raw Kaggle data
    python trace_training.py --train tr.parquet --test te.parquet   # notebook frames
    python trace_training.py --epochs 50 --lr 0.005 --top 20

Outputs (in trace_output/ next to this script):
    trajectories.png      lines: bias + the --top weights with the largest final |value|
    heatmap.png           every parameter (rows) x every update (columns)
    top_rows.png          the Ids that pushed the parameters hardest overall
    learning_curves.png   the usual k-fold log RMSE plot from Kaggle_House_Prices.py (its CONFIG, Adam)
    row_influence.csv     per Id: total |push|, net push on bias, final residual
    param_top_rows.csv    per parameter: the 10 Ids with the largest total |push|
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Kaggle_House_Prices import (  # noqa: E402
    CONFIG, build_features, check_columns, k_fold, load_raw, plot_learning_curves, read_frame)

OUT_DIR = Path(__file__).with_name("trace_output")


def trace(X, y, epochs, lr, batch_size, seed=0):
    """Train with plain SGD and record parameters + per-row pushes at every update.

    Returns
        trajectory  (n_updates + 1, n_params)  parameter vector before training and after each update
        abs_push    (n_rows, n_params)         sum over training of |push| from each row
        net_push    (n_rows, n_params)         signed sum of pushes from each row
    """
    torch.manual_seed(seed)
    model = nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y, torch.arange(len(X))),
                        batch_size=batch_size, shuffle=True)

    def row_loss(params, x, target):
        pred = functional_call(model, params, (x.unsqueeze(0),))
        return ((pred - target) ** 2).mean()

    per_row_grad = vmap(grad(row_loss), in_dims=(None, 0, 0))

    def flat_params():
        # order: weights (one per feature), then bias -> matches names in main()
        return torch.cat([model.weight.detach().flatten(), model.bias.detach().flatten()])

    trajectory = [flat_params()]
    n_params = trajectory[0].numel()
    abs_push = torch.zeros(len(X), n_params)
    net_push = torch.zeros(len(X), n_params)

    for _ in range(epochs):
        for Xb, yb, idx in loader:
            params = {k: v.detach() for k, v in model.named_parameters()}
            g = per_row_grad(params, Xb, yb)
            g_flat = torch.cat([g["weight"].reshape(len(idx), -1), g["bias"].reshape(len(idx), -1)], dim=1)
            push = -lr * g_flat / len(idx)  # each row's share of this update

            before = trajectory[-1]
            optimizer.zero_grad()
            nn.functional.mse_loss(model(Xb), yb).backward()
            optimizer.step()
            after = flat_params()

            # the pushes must add up to the real update, otherwise attribution is wrong
            assert torch.allclose(push.sum(dim=0), after - before, atol=1e-5), "attribution mismatch"

            abs_push.index_add_(0, idx, push.abs())
            net_push.index_add_(0, idx, push)
            trajectory.append(after)

    return model, torch.stack(trajectory), abs_push, net_push


def plot_trajectories(trajectory, names, top, updates_per_epoch):
    final = trajectory[-1, :-1].abs()
    chosen = final.argsort(descending=True)[:top].tolist() + [len(names) - 1]  # top weights + bias

    fig, (ax_w, ax_b) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    for i in chosen[:-1]:
        ax_w.plot(trajectory[:, i], linewidth=1.2, label=names[i])
    ax_w.set(title=f"Top {top} weights by final |value|", ylabel="value")
    ax_w.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax_b.plot(trajectory[:, -1], color="black")
    ax_b.set(title="bias", xlabel=f"update ({updates_per_epoch} per epoch)", ylabel="value")
    for ax in (ax_w, ax_b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trajectories.png", dpi=130)


def plot_heatmap(trajectory, names, updates_per_epoch):
    # weights only; the bias has its own panel in trajectories.png
    weights = trajectory[:, :-1].T
    order = weights[:, -1].argsort(descending=True)  # sort rows by final value
    limit = weights.abs().max().item()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(weights[order], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                   interpolation="nearest")
    ax.set(title="Every weight over training (rows sorted by final value; bias excluded)",
           xlabel=f"update ({updates_per_epoch} per epoch)", ylabel="weight")
    shown = order[:: max(1, len(order) // 40)]
    ax.set_yticks(range(0, len(order), max(1, len(order) // 40)))
    ax.set_yticklabels([names[i] for i in shown], fontsize=6)
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap.png", dpi=130)


def plot_top_rows(influence, n=20):
    top = influence.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["Id"].astype(str), top["total_abs_push"])
    ax.set(title=f"Top {n} training rows by total |push| on all parameters",
           xlabel="sum of |push| over training", ylabel="Id")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_rows.png", dpi=130)


def main(train_df=None, test_df=None, epochs=CONFIG["epochs"], lr=0.01,
         batch_size=CONFIG["batch_size"], top=15):
    if train_df is None and test_df is None:
        train_df, test_df = load_raw()
    elif train_df is None or test_df is None:
        raise ValueError("pass both train_df and test_df, or neither")
    check_columns(train_df, test_df)

    features = build_features(train_df, test_df)
    n_train = len(train_df)
    X = torch.tensor(features[:n_train].values)
    y = torch.log(torch.tensor(train_df["SalePrice"].values, dtype=torch.float32)).reshape(-1, 1)
    names = list(features.columns) + ["bias"]
    ids = train_df["Id"].to_numpy()  # by position: rows dropped in a notebook stay aligned

    model, trajectory, abs_push, net_push = trace(X, y, epochs, lr, batch_size)
    updates_per_epoch = (len(trajectory) - 1) // epochs
    with torch.no_grad():
        residual = (model(X) - y).squeeze(1)
        final_rmse = residual.pow(2).mean().sqrt().item()
    print(f"{len(names)} parameters, {len(trajectory) - 1} updates, final train log rmse {final_rmse:.4f}")

    OUT_DIR.mkdir(exist_ok=True)
    influence = pd.DataFrame({
        "Id": ids,
        "total_abs_push": abs_push.sum(dim=1).numpy(),
        "net_push_on_bias": net_push[:, -1].numpy(),
        "final_residual_log": residual.numpy(),  # prediction - actual, in log price
    }).sort_values("total_abs_push", ascending=False)
    influence.to_csv(OUT_DIR / "row_influence.csv", index=False)

    top_ids = abs_push.argsort(dim=0, descending=True)[:10]  # (10, n_params)
    pd.DataFrame([
        {"parameter": name, "final_value": trajectory[-1, j].item(), "rank": r + 1,
         "Id": ids[top_ids[r, j]], "total_abs_push": abs_push[top_ids[r, j], j].item(),
         "net_push": net_push[top_ids[r, j], j].item()}
        for j, name in enumerate(names) for r in range(10)
    ]).to_csv(OUT_DIR / "param_top_rows.csv", index=False)

    plot_trajectories(trajectory, names, top, updates_per_epoch)
    plot_heatmap(trajectory, names, updates_per_epoch)
    plot_top_rows(influence)

    # the usual k-fold learning curves from Kaggle_House_Prices.py (its CONFIG, Adam), same data
    print("k-fold with Kaggle_House_Prices CONFIG:")
    torch.manual_seed(0)  # same seed as Kaggle_House_Prices.main -> identical curves
    plot_learning_curves(k_fold(X, y)).savefig(OUT_DIR / "learning_curves.png", dpi=130)
    print(f"wrote outputs to {OUT_DIR}")
    print("top 10 rows by total |push|:")
    print(influence.head(10).to_string(index=False))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace parameter movement and per-row pushes")
    parser.add_argument("--train", help="processed train dataframe (.parquet/.pkl/.csv), with Id + SalePrice")
    parser.add_argument("--test", help="processed test dataframe (.parquet/.pkl/.csv), with Id")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=0.01, help="plain SGD learning rate")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--top", type=int, default=15, help="weights shown in trajectories.png")
    args = parser.parse_args()
    if bool(args.train) != bool(args.test):
        parser.error("--train and --test must be given together")

    frames = (read_frame(args.train), read_frame(args.test)) if args.train else (None, None)
    main(*frames, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, top=args.top)

"""
plot the dataset and the regression line
price = θ0 + θ1 * km

usage:
    python3 visualizer.py              show the window
    python3 visualizer.py out.png      save the image

θ0, θ1 are read from theta.json (made by train.py)
if theta.json does not exist, θ0 = θ1 = 0
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from ft_math import ft_predict
from predict import load_theta
from train import load, extract_columns

# palette (validated: scripts/validate_palette.js, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SUB = "#52514e"
GRID = "#e8e7e3"
ERROR = "#b8b7b0"
POINT = "#2a78d6"
LINE = "#eb6834"


def format_equation(theta0: float, theta1: float) -> str:
    """
    return "price = θ0 − |θ1| × km" as a string
    """
    sign = "−" if theta1 < 0 else "+"
    return f"price = {theta0:,.2f} {sign} {abs(theta1):.5f} × km"


def style_axes(ax: Axes) -> None:
    """
    recessive grid and axes
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SUB, length=0, labelsize=10, pad=6)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def draw(
        km: np.ndarray,
        price: np.ndarray,
        theta0: float,
        theta1: float
        ) -> Figure:
    """
    return a figure with the data, the errors and the regression line
    """
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    style_axes(ax)

    # 回帰直線(データ範囲より少し広く)
    x_line = np.array([0.0, km.max() * 1.05])
    y_line = ft_predict(theta0, theta1, x_line)

    # 誤差(予測と実際のずれ)。cost はこの長さの2乗の平均
    predicted = ft_predict(theta0, theta1, km)
    ax.vlines(km, predicted, price, color=ERROR, linewidth=1.0, zorder=1)
    ax.plot([], [], color=ERROR, linewidth=1.0,
            label="Error (actual − predicted)")

    ax.plot(x_line, y_line, color=LINE, linewidth=2.0, zorder=2,
            solid_capstyle="round", label="Regression line")
    ax.scatter(km, price, s=70, color=POINT, edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="Data")

    # y 軸の範囲は、点と直線の両方が見えるように
    lo = min(price.min(), y_line.min())
    hi = max(price.max(), y_line.max())
    pad = (hi - lo) * 0.08
    ax.set_xlim(0, x_line[1])
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Mileage (km)", color=INK_SUB, fontsize=11, labelpad=10)
    ax.set_ylabel("Price", color=INK_SUB, fontsize=11, labelpad=10)

    # 凡例は、データ → 直線 → 誤差の順
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for n in
             ("Data", "Regression line", "Error (actual − predicted)")]
    legend = ax.legend([handles[i] for i in order],
                       [labels[i] for i in order],
                       frameon=False, loc="upper right", fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK_SUB)

    fig.text(0.06, 0.95, "Car price by mileage", color=INK, fontsize=16,
             fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.895, format_equation(theta0, theta1), color=INK_SUB,
             fontsize=11, ha="left", va="top")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.14)
    return fig


def has_display() -> bool:
    """
    return True if a window can be shown
    """
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> None:
    """
    main
    """
    if len(sys.argv) > 2:
        raise ValueError("usage: visualizer.py [output.png]")
    output = sys.argv[1] if len(sys.argv) == 2 else None

    df = load("data.csv")
    km, price = extract_columns(df)
    theta0, theta1 = load_theta("theta.json")
    if not os.path.isfile("theta.json"):
        print("theta.json not found: theta0 = theta1 = 0", file=sys.stderr)

    if output is None and not has_display():
        output = "regression.png"
        print(f"no display: saving to {output}")

    if output is not None:
        matplotlib.use("Agg")
    fig = draw(km, price, theta0, theta1)

    if output is None:
        plt.show()
        return
    try:
        fig.savefig(output, dpi=200, facecolor=SURFACE)
    except (OSError, ValueError) as e:
        raise OSError(f"cannot save {output}") from e
    print(f"saved: {output}")


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

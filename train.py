"""
mileage = 走行距離
price = θ0 + θ1 * km(mileage)

標準偏差(standard deviation)
std = sqrt( (1/m) * Σ(x_i - mean)² )

standardization:標準化(z-score, 標準得点)
z = (x - mean) / std
"""

import pandas as pd
import sys
import os
import numpy as np
from ft_math import ft_mean, ft_std, ft_standardize, ft_gradient_descent
import json


class BadPathError(Exception):
    pass


def _check_path(path: str) -> None:
    """
    check path
    """
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not path:
        raise ValueError("path is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")
    if not path.endswith(".csv"):
        raise ValueError(f"{path} is not CSV file")


def load(path: str) -> pd.DataFrame:
    """
    return PD_Dataflame
    if error exit(1)
    """
    try:
        _check_path(path)
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset of dimensions {df.shape}")
    return df


def extract_columns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    return (km, price) as float arrays
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a DataFrame")
    if df.shape[1] < 2:
        raise ValueError("dataset needs 2 columns (km, price)")
    if df.shape[0] < 2:
        raise ValueError("dataset needs at least 2 rows")

    # print(df.to_numpy().dtype)
    try:
        data = df.to_numpy(dtype=float)
    except ValueError as e:
        raise ValueError("dataset must contain only numbers") from e
    # print(data)
    if not np.isfinite(data).all():
        raise ValueError("dataset contains NaN or inf")

    km = data[:, 0]
    price = data[:, 1]
    # print(mileage)

    return km, price


def standardize_feature(
        x: np.ndarray
        ) -> tuple[np.ndarray, float, float]:
    """
    return (standardized x, mean, std)
    """
    # 既存パッケージ使用
    # x_mean, x_std = x.mean(), x.std()
    # x_n = (x - x_mean) / x_std

    x_mean, x_std = ft_mean(x), ft_std(x, ddof=0)
    if x_std == 0:
        raise ValueError("cannot standardize: all values are the same")
    return ft_standardize(x, x_mean, x_std), x_mean, x_std


def check_number(value: float, name: str) -> None:
    """
    raise an error if value is not a finite int or float
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def unstandardize(
        t0: float,
        t1: float,
        x_mean: float,
        x_std: float
        ) -> tuple[float, float]:
    """
    return unstandardized θ0 and θ1
    """
    check_number(t0, "t0")
    check_number(t1, "t1")
    check_number(x_mean, "x_mean")
    check_number(x_std, "x_std")
    if x_std == 0:
        raise ValueError("x_std must not be 0")
    theta1 = t1 / x_std
    theta0 = t0 - theta1 * x_mean

    return (theta0, theta1)


def save_theta(path: str, theta0: float, theta1: float) -> None:
    """
    save theta0 and theta1 to a json file
    """
    if not isinstance(path, str):
        raise TypeError("path must be a str")
    if not path:
        raise ValueError("path is empty")
    if not path.endswith(".json"):
        raise ValueError(f"{path} is not JSON file")
    check_number(theta0, "theta0")
    check_number(theta1, "theta1")

    res = {
        "theta0": theta0,
        "theta1": theta1,
    }
    try:
        with open(path, "w") as f:
            json.dump(res, f)
    except OSError as e:
        raise OSError(f"cannot write {path}") from e


def main() -> None:
    """
    main
    """
    df = load("data.csv")
    # print(df.head())

    # 準備--------------------------------------------
    km, price = extract_columns(df)
    km_n, km_mean, km_std = standardize_feature(km)

    # print(f"平均走行距離: {km_mean}")
    # print(f"標準偏差: {km_std}")
    # print(f"標準得点: {km_n}")

    # 勾配降下法-----------------------------------------
    # print(ft_cost(0.0, 0.0, km_n, price))
    # 20880519.291666668

    # print(err)
    t0, t1, history = ft_gradient_descent(km_n, price)

    print(f"before: {history[0]}")
    print(f"after:  {history[-1]}")
    print(f"epochs: {len(history) - 1}")
    print(f"t0: {t0}, t1: {t1}  (standardized scale)")
    # t0=6331 kmが平均値の場合の予測価格。
    # t1=-1106 標準偏差１つ分増えると、価格がt1下がる

    # numpy_ver
    # b, a = np.polyfit(km, price, 1)
    # print(f"numpy:  t0: {a}, t1: {b}")
    # print(f"ours:   t0: {t0 - t1 * km_mean / km_std}, t1: {t1 / km_std}")
    theta0, theta1 = unstandardize(t0, t1, km_mean, km_std)
    # print(f"theta0: {theta0}")
    # print(f"theta1: {theta1}")

    save_theta("theta.json", theta0, theta1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

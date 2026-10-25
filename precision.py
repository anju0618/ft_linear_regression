"""
calculate the precision of the linear regression
price = θ0 + θ1 * km

MSE:平均二乗誤差(Mean Squared Error)
    予測と実際のズレを 2 乗して平均。大きく外れたものを重く見る
MSE  = (1 / m) * Σ(予測 - 実際)²

RMSE:二乗平均平方根誤差(Root Mean Squared Error)
    MSE の平方根。単位が price と同じ(円)なので「平均でこのくらいズレる」と読める
RMSE = sqrt(MSE)

MAE:平均絶対誤差(Mean Absolute Error)
    ズレの絶対値の平均。外れ値の影響が RMSE より小さい
MAE  = (1 / m) * Σ|予測 - 実際|

R²:決定係数(coefficient of determination)
    「いつも price の平均を答える」予測と比べて、直線がどれだけ良いかを表す
    分子 = 直線の誤差、分母 = 平均で答えたときの誤差
    1 なら完璧、0 なら平均で答えるのと同じ、負なら平均より悪い
R²   = 1 - Σ(実際 - 予測)² / Σ(実際 - 平均)²

θ0, θ1 are read from theta.json
"""

import sys
import numpy as np
from ft_math import ft_mean, ft_predict
from predict import load_theta
from train import load, extract_columns


def ft_mse(predict: np.ndarray, y: np.ndarray) -> float:
    """
    return (1 / m) * Σ(予測 - 実際)²
    """
    return float(((predict - y) ** 2).sum() / len(y))


def ft_rmse(predict: np.ndarray, y: np.ndarray) -> float:
    """
    return sqrt(MSE) (same unit as price)
    """
    return float(np.sqrt(ft_mse(predict, y)))


def ft_mae(predict: np.ndarray, y: np.ndarray) -> float:
    """
    return (1 / m) * Σ|予測 - 実際|
    """
    return float(np.abs(predict - y).sum() / len(y))


def ft_r2(predict: np.ndarray, y: np.ndarray) -> float:
    """
    return coefficient of determination
    1 is perfect, 0 is as bad as always predicting the mean
    """
    rss = ((y - predict) ** 2).sum()
    tss = ((y - ft_mean(y)) ** 2).sum()
    if tss == 0:
        raise ValueError("price has no variance, R² is undefined")
    return float(1 - rss / tss)


def main() -> None:
    """
    main
    """
    theta0, theta1 = load_theta("theta.json")
    km, price = extract_columns(load("data.csv"))
    predict = ft_predict(theta0, theta1, km)

    print(f"theta0: {theta0}")
    print(f"theta1: {theta1}")
    print(f"MSE   : {ft_mse(predict, price):.2f}")
    print(f"RMSE  : {ft_rmse(predict, price):.2f}")
    print(f"MAE   : {ft_mae(predict, price):.2f}")
    print(f"R²    : {ft_r2(predict, price):.4f}")


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

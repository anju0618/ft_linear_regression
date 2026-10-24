# ft_linear_regression

42 project: predict car price from mileage with linear regression (gradient descent).

## Model

```
price = θ0 + θ1 * km
```

- θ0: 切片(走行距離 0 のときの価格)
- θ1: 傾き(1 km あたりの価格の変化。中古車なので負になる)

## Usage

```
make install   # 依存関係をインストール (uv sync)
make train     # 学習して theta.json に θ0, θ1 を保存
make predict   # 走行距離を入力して価格を予測 (theta.json がなければ θ0 = θ1 = 0)
make vision    # データと回帰直線を表示 (画像で保存: python3 visualizer.py out.png)
make lint      # flake8 + mypy
make clean     # キャッシュ削除
make fclean    # clean + .venv 削除
```

## ft_math

自作の統計・回帰関数をまとめたパッケージ。

```python
from ft_math import ft_mean, ft_std, ft_standardize, ft_predict, ft_cost
```

### stats

| 関数 | 内容 | 式 |
|---|---|---|
| `ft_mean(x)` | 平均 | `Σx / m` |
| `ft_std(x, ddof=0)` | 標準偏差。ばらつきの大きさ。`ddof=0` は母標準偏差、`1` は標本標準偏差 | `sqrt(Σ(x - mean)² / (m - ddof))` |
| `ft_standardize(x, mean, std)` | 標準化(z-score)。平均 0、標準偏差 1 にそろえる | `(x - mean) / std` |

km は数十万、price は数千と桁が違うため、勾配降下の前に km を標準化する。

### regression

| 関数 | 内容 | 式 |
|---|---|---|
| `ft_predict(t0, t1, x)` | 直線で価格を予測する | `t0 + t1 * x` |
| `ft_cost(t0, t1, x, y)` | 予測と実際の差の 2 乗の平均(MSE の 1/2)。小さいほど直線がデータに合っている | `Σ(予測 - 実際)² / 2m` |

### Example

```python
import numpy as np
from ft_math import ft_mean, ft_std, ft_standardize, ft_predict, ft_cost

km = np.array([240000.0, 139800.0, 150500.0])
price = np.array([3650.0, 3800.0, 4400.0])

km_mean, km_std = ft_mean(km), ft_std(km)
km_n = ft_standardize(km, km_mean, km_std)

print(ft_predict(0.0, 0.0, km_n))        # [0. 0. 0.]
print(ft_cost(0.0, 0.0, km_n, price))    # θ が 0 のときのコスト
```

## Resources

- [統計学の時間](https://bellcurve.jp/statistics/course/)
  - [6-2. 標準偏差](https://bellcurve.jp/statistics/course/5924.html)
  - [6-3. 標準偏差の使い方](https://bellcurve.jp/statistics/course/5927.html)
- [勾配降下法とは](https://www.ibm.com/jp-ja/think/topics/gradient-descent)
- [勾配降下法について](https://qiita.com/YudaiSadakuni/items/ece07b04c685a64eaa01)

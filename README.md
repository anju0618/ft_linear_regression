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
make precision # 精度を表示 (MSE, RMSE, MAE, R²)
make vision    # データと回帰直線を表示 (画像で保存: python3 visualizer.py out.png)
make lint      # flake8 + mypy
make clean     # キャッシュと生成物 (theta.json) を削除
make fclean    # clean + .venv + data.csv 削除
```

## precision

`make precision` で `theta.json` と `data.csv` から精度を計算する。

| 指標 | 内容 | 式 |
|---|---|---|
| MSE(平均二乗誤差) | ズレを 2 乗して平均。大きな外れを重く見る | `Σ(予測 - 実際)² / m` |
| RMSE(二乗平均平方根誤差) | MSE の平方根。単位が price と同じ(円)で「平均でこのくらいズレる」と読める | `sqrt(MSE)` |
| MAE(平均絶対誤差) | ズレの絶対値の平均。外れ値の影響が RMSE より小さい | `Σ|予測 - 実際| / m` |
| R²(決定係数) | 平均で答える予測と比べて、直線がどれだけ良いか | `1 - Σ(実際 - 予測)² / Σ(実際 - 平均)²` |

### R² の読み方

```
R² = 1 - A / B
A = Σ(実際 - 予測)²   直線で予測したときのズレの 2 乗和
B = Σ(実際 - 平均)²   いつも price の平均を答えたときのズレの 2 乗和
```

A / B は「直線の誤差が、平均で答えた誤差の何割か」。R² はそれを 1 から引いた値。

| R² | 意味 |
|---|---|
| 1 | 直線が全部ぴったり当たる |
| 0 | 平均で答えるのと同程度 |
| 負 | 平均で答えるより悪い |

例: R² = 0.73 なら、price のばらつきの約 73% を km で説明できている。
MSE などは単位(円)付きで、良し悪しは price の大きさ次第。R² は単位がなく、データが変わっても比べられる。

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

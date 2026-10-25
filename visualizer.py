"""
plot the dataset and the regression line
price = θ0 + θ1 * km

usage:
    python3 visualizer.py              show the window
    python3 visualizer.py out.png      save the image

θ0, θ1 are read from theta.json (made by train.py)
if theta.json does not exist, θ0 = θ1 = 0
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from ft_math import ft_predict
from predict import load_theta
from train import load, extract_columns


def main() -> None:
    """
    main
    """
    if len(sys.argv) > 2:
        raise ValueError("usage: visualizer.py [output.png]")

    km, price = extract_columns(load("data.csv"))
    theta0, theta1 = load_theta("theta.json")

    x_line = np.array([0.0, km.max() * 1.05])
    plt.scatter(km, price, label="Data")
    plt.plot(x_line, ft_predict(theta0, theta1, x_line), color="red",
             label="Regression line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.legend()

    if len(sys.argv) == 2:
        plt.savefig(sys.argv[1])
        print(f"saved: {sys.argv[1]}")
    else:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

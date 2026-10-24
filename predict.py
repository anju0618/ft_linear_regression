"""
predict the price of a car from its mileage
estimatePrice(mileage) = θ0 + θ1 * mileage

θ0, θ1 are read from theta.json
if theta.json does not exist, θ0 = θ1 = 0
"""

import json
import os
import sys
import numpy as np
from ft_math import ft_predict


def load_theta(path: str) -> tuple[float, float]:
    """
    return (theta0, theta1) read from a json file
    return (0.0, 0.0) if the file does not exist
    """
    if not isinstance(path, str):
        raise TypeError("path must be a str")
    if not path:
        raise ValueError("path is empty")
    if not path.endswith(".json"):
        raise ValueError(f"{path} is not JSON file")
    if not os.path.isfile(path):
        return 0.0, 0.0

    try:
        with open(path) as f:
            data = json.load(f)
        theta0, theta1 = data["theta0"], data["theta1"]
    except OSError as e:
        raise OSError(f"cannot read {path}") from e
    except ValueError as e:
        raise ValueError(f"{path} is not valid JSON") from e
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path} needs theta0 and theta1") from e

    for name, value in (("theta0", theta0), ("theta1", theta1)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be an int or float")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    return float(theta0), float(theta1)


def ask_mileage() -> float:
    """
    ask the mileage and return it as a float
    raise an error if it is not a finite number >= 0
    """
    s = input("mileage (km): ").strip()
    try:
        km = float(s)
    except ValueError as e:
        raise ValueError(f"mileage must be a number: {s!r}") from e
    if not np.isfinite(km):
        raise ValueError(f"mileage must be finite: {s!r}")
    if km < 0:
        raise ValueError(f"mileage must not be negative: {s!r}")

    return km


def main() -> None:
    """
    main
    """
    theta0, theta1 = load_theta("theta.json")
    km = ask_mileage()

    price = float(ft_predict(theta0, theta1, np.array([km]))[0])
    if not np.isfinite(price):
        raise ValueError("estimated price is not finite")
    print(f"estimated price: {price:.2f}")


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

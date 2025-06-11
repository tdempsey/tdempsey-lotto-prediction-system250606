"""Flask app replicating PHP lot_cover functionality up to count building."""

from __future__ import annotations

import csv
import os
from typing import List

from flask import Flask, jsonify, render_template

from lotto250611.scaffolding_count import build_number_counts


app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Render the index page."""
    return render_template("index.html")


def load_draws(path: str) -> List[List[int]]:
    """Load lottery draws from a CSV file.

    The CSV file is expected to have a column named ``numbers`` where each row
    contains comma-separated integers.

    Args:
        path: Path to the CSV file containing draw data.

    Returns:
        List of draws, where each draw is a list of integers.
    """
    draws: List[List[int]] = []
    if not os.path.exists(path):
        return draws

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            numbers = [int(n) for n in row.get("numbers", "").split(",") if n]
            if numbers:
                draws.append(numbers)
    return draws


@app.route("/counts")
def counts() -> dict:
    """Endpoint returning number counts for the first 1000 draws."""
    data_path = os.path.join(os.path.dirname(__file__), "data.csv")
    draws = load_draws(data_path)[:1000]
    number_counts = build_number_counts(draws)
    return jsonify(number_counts)


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)

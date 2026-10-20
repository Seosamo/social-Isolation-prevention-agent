"""Formatting shared by task-specific evidence views."""
import pandas as pd


def percentage(value):
    return "계산 불가" if pd.isna(value) else f"{value * 100:.1f}%"

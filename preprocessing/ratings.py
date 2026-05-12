import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("../data")
RATE_PATH = Path("data/company_rate.csv")
CLIENTS_PATH = DATA_DIR / "client_dataset.json"


def company_popularity(df: pd.DataFrame) -> pd.Series:
    return df.groupby("client_name").size().rename("company_popularity")


def staff_turnover(df: pd.DataFrame) -> pd.Series:
    mask = df["grade_proof"].eq("подтверждён")
    return df.loc[mask].groupby("client_name").size().rename("staff_turnover")


def competition_ratio(df: pd.DataFrame) -> pd.DataFrame:
    total = df.groupby(["client_name", "position"]).size().rename("total_applicants")
    confirmed = (
        df.loc[df["grade_proof"].eq("подтверждён")]
        .groupby(["client_name", "position"])
        .size()
        .rename("confirmed_applicants")
    )

    result = total.to_frame().join(confirmed, how="left")
    result["confirmed_applicants"] = result["confirmed_applicants"].fillna(0)
    result["competition_ratio"] = result["total_applicants"].div(
        result["confirmed_applicants"].replace(0, np.nan)
    )
    return result


def company_rates(df: pd.DataFrame, rate_path: Path = RATE_PATH) -> pd.DataFrame:
    rates = pd.read_csv(rate_path)
    return df.merge(rates, on="client_name", how="inner")


def load_clients(path: Path = CLIENTS_PATH) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


if __name__ == "__main__":
    df = load_clients()
    df = company_rates(df)
    print(df.head(10))

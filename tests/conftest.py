import json
from pathlib import Path

import pandas as pd
import pytest

from activity_agent.common.config import Config
from activity_agent.data.ingest import AGE_COLUMNS, DAY_COLUMNS, TIME_COLUMNS


@pytest.fixture
def config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    values = json.loads((root / "config/default.json").read_text())
    # Project output may be absolute; all fixtures must remain inside tmp_path.
    values["output_dir"] = "artifacts"
    values["isolation"]["model_id"] = "IF_BASELINE"
    values["isolation"].pop("model_path", None)
    values["isolation"].pop("flow_cutoffs_path", None)
    (tmp_path / "config").mkdir()
    (tmp_path / "references").mkdir()
    for name in ("industries.csv", "block_regions.csv", "resources.csv"):
        (tmp_path / "references" / name).write_bytes((root / "references" / name).read_bytes())
    path = tmp_path / "config/default.json"
    path.write_text(json.dumps(values))
    return Config.load(path)


@pytest.fixture
def source_data(config):
    data = config.path("data_dir")
    data.mkdir()
    blocks = {"b1": [100, 100, 50, 25, 10, 5], "b2": [100] * 6}
    for offset, month in enumerate(["202507", "202508", "202509", "202510", "202511", "202512"]):
        for family, columns in [("age", AGE_COLUMNS), ("time", TIME_COLUMNS), ("wkdy", DAY_COLUMNS)]:
            rows = []
            for b, values in blocks.items():
                for x in (100.0, 200.0):
                    rows.append([month, b, x, 500.0, *([values[offset]] * len(columns))])
            frame = pd.DataFrame(rows, columns=["STD_YM", "BLOCK_CD", "X_COORD", "Y_COORD", *columns])
            if month == "202512" and family in ("time", "wkdy"):
                frame = pd.concat([frame, frame])
            frame.to_csv(data / f"flow_{family}_pop_{month}.csv", sep="|", index=False)
    card1, card2 = [], []
    for offset in range(6):
        for region, values in [("서울 강남구", blocks["b1"]), ("강원 춘천시", blocks["b2"])]:
            for age in ("10 대", "50 대", "60 대", "70 대"):
                for industry in ("영화/공연", "약국"):
                    row = [f"2025{7+offset:02d}01", "12_17", region, industry, "남성", age, values[offset]*100, values[offset]]
                    card1.append(row)
                    card2.append([row[0], region, industry, "서울", "남성", age, row[-2], row[-1]])
    card1.append(["20250701", "12_17", "서울 강남구", "가구", "법인", "법인", 99999, 999])
    pd.DataFrame(card1, columns=["TA_YMD", "TIME_GB", "MCT_SGG_CD", "MCT_RY_CD", "SEX_CCD", "AGE_CCD", "TS_AT", "USE_CNT"]).to_csv(data / "신한카드_데이터1.txt", sep="\t", index=False, encoding="cp949")
    pd.DataFrame(card2, columns=["TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "CLN_SGG_CD", "SEX_CCD", "AGE_CCD", "TS_AT", "USE_CNT"]).to_csv(data / "신한카드_데이터2.txt", sep="\t", index=False, encoding="cp949")
    return config


@pytest.fixture
def complete_pipeline(source_data):
    from activity_agent.data.ingest import preprocess
    from activity_agent.activity.features import build_features
    from activity_agent.activity.detection import detect
    from activity_agent.activity.labels import label
    preprocess(source_data)
    build_features(source_data)
    detect(source_data)
    label(source_data)
    return source_data

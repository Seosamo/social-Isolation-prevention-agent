import pandas as pd
from shapely.geometry import box

from activity_agent.isolation.spatial import assign_coordinates


def test_candidate_crs_consensus_requires_a_unique_matching_province():
    x, y = 956633.991787, 1947479.147389
    points = pd.DataFrame({"x": [x, x + 10000], "y": [y, y]})
    polygon = box(x - 1000, y - 1000, x + 1000, y + 1000)
    result = assign_coordinates(points, {"11": polygon}, 5179)
    assert result.geo_code.tolist() == ["11", ""]
    tight = assign_coordinates(points.iloc[:1], {"11": box(x - 1, y - 1, x + 1, y + 1)}, 5179)
    assert tight.sido_5179.iloc[0] == "11"
    assert tight.sido_5178.iloc[0] == ""
    assert tight.geo_code.iloc[0] == ""
    ambiguous = assign_coordinates(points.iloc[:1], {"11": polygon, "31": polygon}, 5179)
    assert ambiguous.geo_code.iloc[0] == ""

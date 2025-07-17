# ライブラリ
from dataclasses import dataclass
from typing import Any, Callable

import geopandas as gpd
import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pulp
import streamlit as st
from geopandas import GeoDataFrame
from networkx import MultiDiGraph
from shapely.geometry import Point


@dataclass
class MapInfo:
    gdf: GeoDataFrame
    graph: MultiDiGraph
    roads: GeoDataFrame
    rivers: GeoDataFrame
    evacuation_sites: GeoDataFrame
    evacuee_gdf: GeoDataFrame

    def plot_map(self) -> tuple[plt.Figure, plt.Axes]:
        # データを可視化
        fig, ax = plt.subplots(figsize=(13, 13))
        # 国立市のバウンディングボックスを取得
        minx, miny, maxx, maxy = self.gdf.total_bounds
        rangex = abs(maxx - minx)
        rangey = abs(maxy - miny)

        # 表示範囲を国立市のバウンディングボックスにズーム
        ax.set_xlim(minx - rangex * 0.01, maxx + rangex * 0.01)
        ax.set_ylim(miny - rangey * 0.01, maxy + rangey * 0.01)
        self.gdf.plot(
            ax=ax, facecolor="none", edgecolor="black", linewidth=1, label="国立市境界"
        )  # 国立市の境界
        self.rivers.plot(ax=ax, color="blue", linewidth=2, label="河川")  # 河川
        self.roads.plot(ax=ax, color="gray", linewidth=1, alpha=0.6, label="道路")  # 道路

        return fig, ax


@st.cache_data
def generate_data(
    n: int, m: int, capacities: list[int], total_population: int, max_distance: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    人工データを生成する関数。

    Args:
    n (int): 避難所の候補地の数。
    m (int): 避難者グループの数。
    capacities (list[int]): 各避難所の収容人数上限。
    total_population (int): 避難者グループの総人口。
    max_distance (int): 最大距離（道路ネットワークの計算用）。
    seed (int): 乱数のシード。

    Returns:
    tuple[MapInfo, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        地理情報、避難所の座標（形状: (n, 2)）、避難者グループの座標（形状: (m, 2)）、避難者グループの人口（形状: (m,)）、各避難所の収容人数上限（形状: (n,)）、避難者グループから避難所までの距離行列（形状: (n, m)）。
    """
    np.random.seed(seed)

    # 国立市の行政境界を取得
    place_name = "国立市, 東京都, 日本"
    gdf = ox.geocode_to_gdf(place_name)

    # 道路ネットワークを取得（徒歩移動を考慮）
    graph = ox.graph_from_place(place_name, network_type="walk")
    roads = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    # 川（河川）を取得
    river_tags = {"waterway": ["river", "stream", "canal"]}
    rivers = ox.features_from_place(place_name, river_tags)

    # 避難所候補の取得（学校、公民館、体育館、市役所）
    tags = {
        "amenity": ["school", "townhall", "community_centre"],  # 学校、公民館、市役所
        # "leisure": "sports_centre",  # 体育館
    }
    evacuation_sites = ox.features_from_place(place_name, tags)
    evacuation_sites = evacuation_sites[["geometry"]].dropna()
    evacuation_sites = evacuation_sites[
        evacuation_sites.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])
    ]
    evacuation_sites["geometry"] = evacuation_sites["geometry"].centroid
    evacuation_sites["latitude"] = evacuation_sites.geometry.y
    evacuation_sites["longitude"] = evacuation_sites.geometry.x
    evacuation_sites = evacuation_sites.sample(n=min(n, len(evacuation_sites)), random_state=seed)

    shelter_coords = np.array([[g.x, g.y] for g in evacuation_sites.geometry])
    evacuation_sites["capacity"] = capacities

    # 避難者グループの位置をランダムに生成（国立市の範囲内）
    evacuee_points = []
    while len(evacuee_points) < m:
        p = Point(
            np.random.uniform(gdf.total_bounds[0], gdf.total_bounds[2]),
            np.random.uniform(gdf.total_bounds[1], gdf.total_bounds[3]),
        )
        if gdf.contains(p).any():
            evacuee_points.append(p)

    evacuee_coords = np.array([[p.x, p.y] for p in evacuee_points])
    # 避難者グループの人口をランダムに生成
    # 人口の合計が total_population になるように割合をランダムに生成
    population_ratios = np.random.rand(m)
    population_ratios /= population_ratios.sum()
    evacuee_populations = (population_ratios * total_population).astype(int)
    evacuee_groups = pd.DataFrame(
        {
            "latitude": evacuee_coords[:, 1],
            "longitude": evacuee_coords[:, 0],
            "people": evacuee_populations,
        }
    )
    evacuee_gdf = gpd.GeoDataFrame(
        evacuee_groups,
        geometry=gpd.points_from_xy(evacuee_groups.longitude, evacuee_groups.latitude),
    )

    map_info = MapInfo(
        gdf, graph, roads, rivers, evacuation_sites=evacuation_sites, evacuee_gdf=evacuee_gdf
    )

    # 避難者グループと避難所の距離行列を計算（道路ネットワークを利用）
    distance_matrix = np.full((m, n), 1e7)

    for i, evacuee in enumerate(evacuee_coords):
        evacuee_node = ox.distance.nearest_nodes(graph, evacuee[0], evacuee[1])
        for j, shelter in enumerate(shelter_coords):
            shelter_node = ox.distance.nearest_nodes(graph, shelter[0], shelter[1])
            try:
                dist = nx.shortest_path_length(graph, evacuee_node, shelter_node, weight="length")
                if dist <= max_distance:
                    distance_matrix[i, j] = dist
            except nx.NetworkXNoPath:
                continue  # 経路がない場合はそのまま無限大（∞）

    return (
        map_info,
        shelter_coords,
        evacuee_coords,
        evacuee_populations,
        capacities,
        distance_matrix.T,
    )


DESC_OPTIMIZE_SHELTER_INSTALLATION = """
    ### 避難所配置問題の定式化

    #### 概要:
    全員が避難できるという条件のもとで，避難所の設置数を最小化する

    #### 定数:
    - $$ n $$: 避難所の候補地の数。
    - $$ m $$: 避難者グループの数。
    - $$ D $$: 避難可能な最大距離。
    - $$ p_j $$: 避難者グループ $$j$$ の人口。
    - $$ c_i $$: 避難所 $$i$$ の収容人数上限。
    - $$ d_{ij} $$: 避難者グループ $$j$$ から避難所 $$i$$ までの距離。

    #### 変数:
    - $$ x_i $$: 避難所 $$i$$ が設置されるかどうかを示すバイナリ変数 (1 なら設置、0 なら未設置)。
    - $$ y_{ij} $$: 避難者グループ $$j$$ が避難所 $$i$$ に割り当てられる割合を示す連続変数 (0 から 1 の範囲)。
    - $$ z_{ij} $$: 避難者グループ $$j$$ が避難所 $$i$$ に割り当てられるかどうかを示すバイナリ変数 (1 なら割り当て、0 なら割り当てない)。

    #### 目的関数:
    避難所の設置数を最小化する。
    $$
    \\text{Minimize} \\quad \\sum_{i=1}^{n} x_i
    $$

    #### 制約条件:
    1. **避難者グループの割り当て制約**:
    各避難者グループはちょうど一つの避難所に割り当てられる。
    $$ \\sum_{i=1}^{n} y_{ij} = 1 \\quad \\forall j \\in \\{1, 2, \\ldots, m\\} $$

    2. **避難所設置制約**:
    避難者グループが避難所に割り当てられる場合、その避難所が設置されている必要がある。
    $$
    y_{ij} \\leq x_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$

    3. **避難所の収容人数制約**:
    避難所の収容人数が収容人数上限を超えないようにする。
    $$
    \\sum_{j=1}^{m} y_{ij} \\cdot p_j \\leq c_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}
    $$

    4. **距離制約**:
    避難者グループ $$j$$ が避難所 $$i$$ に割り当てられる場合、その距離が最大距離 $$D$$ を超えないようにする。
    $$
    y_{ij} \\leq z_{ij} \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    $$
    z_{ij} \\leq x_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    $$
    d_{ij} \\cdot z_{ij} \\leq D \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    """


def get_shelter_installation_parameters() -> dict[str, Any]:
    D = st.number_input("避難可能な最大距離 (D)", min_value=1000, max_value=10000, value=3000)

    return {"D": D}


def optimize_shelter_installation(
    n: int,
    m: int,
    group_populations: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    D: int,
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]:
    """
    避難所の設置数を最小化するための最適化を行う関数。

    Args:
    n (int): 避難所の候補地の数。
    m (int): 避難者グループの数。
    group_populations (np.ndarray): 避難者グループの人口。
    c (np.ndarray): 各避難所の収容人数上限。
    d (np.ndarray): 避難者グループから避難所までの距離行列。
    D (int): 避難可能な最大距離。

    Returns:
    tuple[dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]: 避難所の設置決定変数、避難者グループの割り当て変数。
    """
    model = pulp.LpProblem("Minimize_Shelter_Installation", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)

    model += pulp.lpSum(x[i] for i in range(n))

    # 避難者グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        for j in range(m):
            # 避難所設置制約
            model += y[i][j] <= x[i]
            # 距離制約
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            model += d[i][j] * z[i][j] <= D

    # 避難所の収容人数制約
    for i in range(n):
        model += pulp.lpSum(y[i][j] * group_populations[j] for j in range(m)) <= c[i]

    model.solve()

    # to dict from LpVariable
    x = {i: x[i] for i in range(n)}
    y = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    status = pulp.LpStatus[model.status]
    return status, x, y


DESC_OPTIMIZE_EVACUATION_TIME = """
    ### 避難距離の最大値最小化モデルの定式化

    #### 概要
    避難所配置問題を解く際に、全ての避難者が避難できるという条件のもとで、避難距離の最大値を最小化することを目的とする。

    #### 定数
    - $$ n $$: 避難所の候補地の数。
    - $$ m $$: 避難者グループの数。
    - $$ D $$: 避難可能な最大距離。
    - $$ p_j $$: 避難者グループ $$ j $$ の人口。
    - $$ c_i $$: 避難所 $$ i $$ の収容人数上限。
    - $$ d_{ij} $$: 避難者グループ $$ j $$ から避難所 $$ i $$ までの距離。

    #### 変数
    - $$ x_i $$: 避難所 $$ i $$ が設置されるかどうかを示すバイナリ変数 (1 なら設置、0 なら未設置)。
    - $$ y_{ij} $$: 避難者グループ $$ j $$ が避難所 $$ i $$ に割り当てられる割合を示す連続変数 (0 から 1 の範囲)。
    - $$ z_{ij} $$: 避難者グループ $$ j $$ が避難所 $$ i $$ に割り当てられるかどうかを示すバイナリ変数 (1 なら割り当て、0 なら割り当てない)。
    - $$ T $$: 最大避難距離を示す連続変数。

    #### 目的関数
    避難距離の最大値を最小化する。
    $$
    \\text{Minimize} \\quad T
    $$

    #### 制約条件
    1. **避難者グループの割り当て制約**:
        各避難者グループはちょうど一つの避難所に割り当てられる。
        $$
        \\sum_{i=1}^{n} y_{ij} = 1 \\quad \\forall j \\in \\{1, 2, \\ldots, m\\}
        $$

    2. **避難所設置制約**:
        避難者グループが避難所に割り当てられる場合、その避難所が設置されている必要がある。
        $$
        y_{ij} \\leq x_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
        $$

    3. **距離制約**:
        避難者グループ $$ j $$ が避難所 $$ i $$ に割り当てられる場合、その距離が最大距離 $$ D $$ を超えないようにする。
        $$
        y_{ij} \\leq z_{ij} \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
        $$
        $$
        z_{ij} \\leq x_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
        $$
        $$
        d_{ij} \\cdot z_{ij} \\leq D \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
        $$

    4. **避難所の収容人数制約**:
        避難所の収容人数が収容人数上限を超えないようにする。
        $$
        \\sum_{j=1}^{m} y_{ij} \\cdot p_j \\leq c_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}
        $$

    5. **最大避難距離制約**:
        各避難者グループの避難距離が最大避難距離 $$ T $$ を超えないようにする。
        $$
        z_{ij} \\cdot d_{ij} \\leq T \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
        $$
    """


def get_evacuation_time_parameters() -> dict[str, Any]:
    D = st.number_input("避難可能な最大距離 (D)", min_value=1000, max_value=10000, value=3000)

    return {"D": D}


def optimize_evacuation_time(
    n: int,
    m: int,
    group_populations: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    D: int,
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable], int]:
    model = pulp.LpProblem("Minimize_Evacuation_Time", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)
    T = pulp.LpVariable("T", lowBound=0, cat=pulp.LpContinuous)  # 最大避難距離

    # 目的関数
    model += T

    # 避難者グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        for j in range(m):
            # 避難所設置制約
            model += y[i][j] <= x[i]
            # 距離制約
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            model += z[i][j] * d[i][j] <= D

    # 避難所の収容人数制約
    for i in range(n):
        model += pulp.lpSum(y[i][j] * group_populations[j] for j in range(m)) <= c[i]

    # 最大避難距離制約
    for i in range(n):
        for j in range(m):
            model += z[i][j] * d[i][j] <= T

    model.solve()

    # to dict from LpVariable
    x = {i: x[i] for i in range(n)}
    y = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    T = int(T.value())
    status = pulp.LpStatus[model.status]
    return status, x, y, T


DESC_MAXIMIZE_SATISFACTION = """
    ### 利益の最大化モデルの定式化

    #### 概要
    このモデルでは、避難所の配置および避難者の割り当てを考慮し、全体の利益を最大化することを目的とします。

    #### 定数
    - $$ I $$: 避難所の候補地集合。
    - $$ J $$: 避難者グループ集合。
    - $$ D $$: 避難可能な最大距離。
    - $$ B $$: 予算。
    - $$ p_j $$: グループ $$ j $$ の人数。
    - $$ c_i $$: 避難所 $$ i $$ の収容上限。
    - $$ d_{ij} $$: 避難所 $$ i $$ とグループ $$ j $$ の間の距離。
    - $$ f_j $$: グループ $$ j $$ の設置コスト。

    #### 変数
    - $$ x_i $$: 避難所 $$ i $$ が設置されるかどうかを示すバイナリ変数 (1 なら設置、0 なら未設置)。
    - $$ y_{ij} $$: グループ $$ j $$ が避難所 $$ i $$ に割り当てられる割合を示す連続変数 (0 から 1 の範囲)。
    - $$ z_{ij} $$: グループ $$ j $$ が避難所 $$ i $$ に割り当てられるかどうかを示すバイナリ変数 (1 なら割り当て、0 なら割り当てない)。

    #### 目的関数
    全体の利益を最大化する。
    $$
    \\text{Maximize} \\quad \\sum_{j \\in J} \\sum_{i \\in I} \\frac{p_j y_{ij} x_i}{d_{ij}}
    $$

    #### 制約条件
    1. **避難者グループの割り当て制約**:
        各避難者グループは1つ以下の避難所に割り当てられる。
        $$
        \\sum_{i \\in I} y_{ij} \\leq 1 \\quad \\forall j \\in J
        $$

    2. **避難所設置制約**:
        避難者グループが避難所に割り当てられる場合、その避難所が設置されている必要がある。
        $$
        y_{ij} \\leq x_i \\quad \\forall i \\in I, \\forall j \\in J
        $$

    3. **避難所の収容人数制約**:
        避難所の収容人数が収容人数上限を超えないようにする。
        $$
        \\sum_{j \\in J} y_{ij} p_j \\leq c_i \\quad \\forall i \\in I
        $$

    4. **割り当て可能性制約**:
        避難者グループ $$ j $$ が避難所 $$ i $$ に割り当てられる場合、その避難所が設置されている必要がある。
        $$
        y_{ij} \\leq z_{ij} \\quad \\forall i \\in I, \\forall j \\in J
        $$
        $$
        z_{ij} \\leq x_i \\quad \\forall i \\in I, \\forall j \\in J
        $$
        $$
        d_{ij} z_{ij} \\leq D \\quad \\forall i \\in I, \\forall j \\in J
        $$

    5. **予算制約**:
        全体の設置コストが予算を超えないようにする。
        $$
        \\sum_{j \\in J} f_j x_j \\leq B
        $$

    6. **変数の範囲**:
        変数の範囲を設定する。
        $$
        x_i \\in \\{0, 1\\} \\quad \\forall i \\in I
        $$
        $$
        y_{ij} \\in [0, 1] \\quad \\forall i \\in I, \\forall j \\in J
        $$
        $$
        z_{ij} \\in \\{0, 1\\} \\quad \\forall i \\in I, \\forall j \\in J
        $$
    """


def get_satisfaction_parameters() -> dict[str, Any]:
    D = st.number_input("避難可能な最大距離 (D)", min_value=1000, max_value=10000, value=3000)
    B = st.number_input("予算 (B)", min_value=1, max_value=1000, value=100)

    return {"D": D, "B": B}


def optimize_satisfaction(
    n: int, m: int, group_populations: np.ndarray, c: np.ndarray, d: np.ndarray, D: int, B: int
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]:
    f = np.ones(m)  # 設置コスト，とりあえず全て1

    model = pulp.LpProblem("Maximize_satisfaction", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)
    l = pulp.LpVariable.dicts("l", (range(n), range(m)), cat=pulp.LpContinuous)  # noqa

    # 目的関数
    model += pulp.lpSum(
        (group_populations[i] * l[i][j]) / d[i][j] for i in range(n) for j in range(m)
    )

    # 避難者グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) <= 1

    for i in range(n):
        for j in range(m):
            # 避難所設置制約
            model += y[i][j] <= x[i]
            # 距離制約
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            model += z[i][j] * d[i][j] <= D

            # l の制約
            model += l[i][j] <= y[i][j]
            model += l[i][j] <= x[i]
            model += l[i][j] >= y[i][j] - (1 - x[i])

    # 避難所の収容人数制約
    for i in range(n):
        model += pulp.lpSum(y[i][j] * group_populations[j] for j in range(m)) <= c[i]

    # 予算制約
    model += pulp.lpSum(f[j] * x[j] for j in range(n)) <= B

    model.solve()

    # to dict from LpVariable
    x = {i: x[i] for i in range(n)}
    y = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    status = pulp.LpStatus[model.status]
    return status, x, y


DESC_OPTIMIZE_SAKAUE_MODEL = """
    ### 坂上モデルの定式化

    #### 概要:
    全員が避難できるという条件のもとで，避難所の設置数を最小化する
    ただし，避難所と避難者グループの間の距離が避難可能な最大距離を超えないようにする

    #### 定数:
    - $$ n $$: 避難所の候補地の数。
    - $$ m $$: 避難者グループの数。
    - $$ D $$: 避難可能な最大距離。
    - $$ l_{ij} $$: 避難所 $$i$$ と避難者グループ $$j$$ の間の避難可能距離。
    - $$ p_j $$: 避難者グループ $$j$$ の人口。
    - $$ c_i $$: 避難所 $$i$$ の収容人数上限。
    - $$ d_{ij} $$: 避難者グループ $$j$$ から避難所 $$i$$ までの距離。

    #### 変数:
    - $$ x_i $$: 避難所 $$i$$ が設置されるかどうかを示すバイナリ変数 (1 なら設置、0 なら未設置)。
    - $$ y_{ij} $$: 避難者グループ $$j$$ が避難所 $$i$$ に割り当てられる割合を示す連続変数 (0 から 1 の範囲)。
    - $$ z_{ij} $$: 避難者グループ $$j$$ が避難所 $$i$$ に割り当てられるかどうかを示すバイナリ変数 (1 なら割り当て、0 なら割り当てない)。

    #### 目的関数:
    避難所の設置数を最小化する。
    $$
    \\text{Minimize} \\quad \\sum_{i=1}^{n} x_i
    $$

    #### 制約条件:
    1. **避難者グループの割り当て制約**:
    各避難者グループはちょうど一つの避難所に割り当てられる。
    $$ \\sum_{i=1}^{n} y_{ij} = 1 \\quad \\forall j \\in \\{1, 2, \\ldots, m\\} $$

    2. **避難所設置制約**:
    避難者グループが避難所に割り当てられる場合、その避難所が設置されている必要がある。
    $$
    y_{ij} \\leq x_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$

    3. **避難所の収容人数制約**:
    避難所の収容人数が収容人数上限を超えないようにする。
    $$
    \\sum_{j=1}^{m} y_{ij} \\cdot p_j \\leq c_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}
    $$

    4. **距離制約**:
    避難者グループ $$j$$ が避難所 $$i$$ に割り当てられる場合、その距離が最大距離 $$D$$ を超えないようにする。
    $$
    y_{ij} \\leq z_{ij} \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    $$
    z_{ij} \\leq x_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    $$
    d_{ij} \\cdot z_{ij} \\leq D \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    $$
    d_{ij} \\cdot z_{ij} \\leq l_{ij} \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    """


def get_sakaue_parameters() -> dict[str, Any]:
    D = st.number_input("避難可能な最大距離 (D)", min_value=1000, max_value=10000, value=3000)

    return {"D": D}


def optimize_sakaue(
    n: int,
    m: int,
    group_populations: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    l: np.ndarray,  # noqa
    D: int,
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]:
    """
    避難所の設置数を最小化するための最適化を行う関数。

    Args:
    n (int): 避難所の候補地の数。
    m (int): 避難者グループの数。
    group_populations (np.ndarray): 避難者グループの人口。
    c (np.ndarray): 各避難所の収容人数上限。
    d (np.ndarray): 避難者グループから避難所までの距離行列。
    l (np.ndarray): 避難所グループから避難所までの避難可能距離。
    D (int): 避難可能な最大距離。

    Returns:
    tuple[dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]: 避難所の設置決定変数、避難者グループの割り当て変数。
    """
    model = pulp.LpProblem("Minimize_Shelter_Installation", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)

    model += pulp.lpSum(x[i] for i in range(n))

    # 避難者グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        for j in range(m):
            # 避難所設置制約
            model += y[i][j] <= x[i]
            # 距離制約
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            model += d[i][j] * z[i][j] <= D
            model += d[i][j] * z[i][j] <= l[i][j]

    # 避難所の収容人数制約
    for i in range(n):
        model += pulp.lpSum(y[i][j] * group_populations[j] for j in range(m)) <= c[i]

    model.solve()

    # to dict from LpVariable
    x = {i: x[i] for i in range(n)}
    y = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    status = pulp.LpStatus[model.status]
    return status, x, y


DESC_OPTIMIZE_SAKAUE_MODEL2 = """
    ### 坂上モデル2の定式化

    #### 概要:
    全員が避難できるという条件のもとで，避難所の設置数を最小化する.
    加えて，避難距離が小さくなるようにペナルティを与える.
    ただし，避難所と避難者グループの間の距離が避難可能な最大距離を超えないようにする

    #### 定数:
    - $$ n $$: 避難所の候補地の数。
    - $$ m $$: 避難者グループの数。
    - $$ D $$: 避難可能な最大距離。
    - $$ l_{ij} $$: 避難所 $$i$$ と避難者グループ $$j$$ の間の避難可能距離。
    - $$ p_j $$: 避難者グループ $$j$$ の人口。
    - $$ c_i $$: 避難所 $$i$$ の収容人数上限。
    - $$ d_{ij} $$: 避難者グループ $$j$$ から避難所 $$i$$ までの距離。
    - $$ \\alpha $$: ペナルティ係数.

    #### 変数:
    - $$ x_i $$: 避難所 $$i$$ が設置されるかどうかを示すバイナリ変数 (1 なら設置、0 なら未設置)。
    - $$ y_{ij} $$: 避難者グループ $$j$$ が避難所 $$i$$ に割り当てられる割合を示す連続変数 (0 から 1 の範囲)。
    - $$ z_{ij} $$: 避難者グループ $$j$$ が避難所 $$i$$ に割り当てられるかどうかを示すバイナリ変数 (1 なら割り当て、0 なら割り当てない)。

    #### 目的関数:
    避難所の設置数を最小化する。
    $$
    \\text{Minimize} \\quad \\sum_{i=1}^{n} x_i + \\alpha \\sum_{i=1}^{n} \\sum_{j=1}^{m} d_{ij} \\cdot p_{j} \\cdot y_{ij}
    $$

    #### 制約条件:
    1. **避難者グループの割り当て制約**:
    各避難者グループはちょうど一つの避難所に割り当てられる。
    $$ \\sum_{i=1}^{n} y_{ij} = 1 \\quad \\forall j \\in \\{1, 2, \\ldots, m\\} $$

    2. **避難所設置制約**:
    避難者グループが避難所に割り当てられる場合、その避難所が設置されている必要がある。
    $$
    y_{ij} \\leq x_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$

    3. **避難所の収容人数制約**:
    避難所の収容人数が収容人数上限を超えないようにする。
    $$
    \\sum_{j=1}^{m} y_{ij} \\cdot p_j \\leq c_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}
    $$

    4. **距離制約**:
    避難者グループ $$j$$ が避難所 $$i$$ に割り当てられる場合、その距離が最大距離 $$D$$ を超えないようにする。
    $$
    y_{ij} \\leq z_{ij} \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    $$
    z_{ij} \\leq x_i \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    $$
    d_{ij} \\cdot z_{ij} \\leq D \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    $$
    $$
    d_{ij} \\cdot z_{ij} \\leq l_{ij} \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}, \\forall j \\in \\{1, 2, \\ldots, m\\}
    """


def get_sakaue2_parameters() -> dict[str, Any]:
    D = st.number_input("避難可能な最大距離 (D)", min_value=1000, max_value=10000, value=3000)
    alpha = st.slider("ペナルティ係数", 0.0, 1.0, 0.0)

    return {"D": D, "alpha": alpha}


def optimize_sakaue2(
    n: int,
    m: int,
    group_populations: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    l: np.ndarray,  # noqa
    D: int,
    alpha: float,
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]:
    """
    避難所の設置数を最小化するための最適化を行う関数。

    Args:
    n (int): 避難所の候補地の数。
    m (int): 避難者グループの数。
    group_populations (np.ndarray): 避難者グループの人口。
    c (np.ndarray): 各避難所の収容人数上限。
    d (np.ndarray): 避難者グループから避難所までの距離行列。
    l (np.ndarray): 避難所グループから避難所までの避難可能距離。
    D (int): 避難可能な最大距離。
    alpha (float): 避難所の設置数と避難輸送距離の重み。

    Returns:
    tuple[dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]: 避難所の設置決定変数、避難者グループの割り当て変数。
    """
    model = pulp.LpProblem("Minimize_Shelter_Installation", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)

    # 目的関数
    # 避難所の設置数 + 避難輸送距離の最小化
    model += (1 - alpha) * pulp.lpSum(x[i] for i in range(n)) + alpha * pulp.lpSum(
        d[i][j] * y[i][j] * group_populations[j] for i in range(n) for j in range(m)
    )

    # 避難者グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        for j in range(m):
            # 避難所設置制約
            model += y[i][j] <= x[i]
            # 距離制約
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            model += d[i][j] * z[i][j] <= D
            model += d[i][j] * z[i][j] <= l[i][j]

    # 避難所の収容人数制約
    for i in range(n):
        model += pulp.lpSum(y[i][j] * group_populations[j] for j in range(m)) <= c[i]

    model.solve()

    # to dict from LpVariable
    x = {i: x[i] for i in range(n)}
    y = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    status = pulp.LpStatus[model.status]
    return status, x, y


REGISTRY: dict[str, dict[str, Callable[..., dict[str, Any]] | str]] = {
    "避難所の設置数最小化": {
        "description": DESC_OPTIMIZE_SHELTER_INSTALLATION,
        "param_fn": get_shelter_installation_parameters,
    },
    "避難距離最小化": {
        "description": DESC_OPTIMIZE_EVACUATION_TIME,
        "param_fn": get_evacuation_time_parameters,
    },
    "満足度最大化": {
        "description": DESC_MAXIMIZE_SATISFACTION,
        "param_fn": get_satisfaction_parameters,
    },
    "坂上モデル": {"description": DESC_OPTIMIZE_SAKAUE_MODEL, "param_fn": get_sakaue_parameters},
    "坂上モデル2": {
        "description": DESC_OPTIMIZE_SAKAUE_MODEL2,
        "param_fn": get_sakaue2_parameters,
    },
}


def visualize_population_data(map_info: MapInfo) -> plt.Figure:
    fig, ax = map_info.plot_map()
    map_info.evacuation_sites.plot(ax=ax, color="green", markersize=40, label="避難所候補地")
    map_info.evacuee_gdf.plot(ax=ax, color="red", markersize=30, label="避難者グループ")

    # 避難所候補地の数を表示
    fontsize = 5
    plusy = 0.0002
    for i, (idx, row) in enumerate(map_info.evacuation_sites.iterrows()):
        ax.text(
            row.geometry.x,
            row.geometry.y + plusy,
            f"避難所：{i} 収容可能人数:{int(row['capacity'])}",
            fontsize=fontsize,
            color="blue",
            ha="center",
        )

    # 避難者グループの人数を表示
    for j, (idx, row) in enumerate(map_info.evacuee_gdf.iterrows()):
        ax.text(
            row.geometry.x,
            row.geometry.y + plusy,
            f"避難所グループ:{j} 避難人数:{int(row['people'])}",
            fontsize=fontsize,
            color="red",
            ha="center",
        )

    ax.legend()

    return fig


def visualize_evacuation_plan(
    map_info: MapInfo,
    distance_matrix: np.ndarray,
    shelter_coords: np.ndarray,
    group_coords: np.ndarray,
    group_populations: np.ndarray,
    x: dict[int, pulp.LpVariable],
    y: dict[tuple[int, int], pulp.LpVariable],
    c: np.ndarray,
    title: str = "避難所配置問題の最適化結果",
) -> plt.Figure:
    """
    最適化の結果を可視化する関数。

    Args:
    map_info
    shelter_coords (np.ndarray): 避難所の座標。
    group_coords (np.ndarray): 避難者グループの座標。
    group_populations (np.ndarray): 避難者グループの人口。
    x (dict[int, pulp.LpVariable]): 避難所の設置決定変数。
    y (dict[tuple[int, int], pulp.LpVariable]): 避難者グループの割り当て変数。
    c (np.ndarray): 各避難所の収容人数上限。
    title (str): 図のタイトル。

    Returns:
    plt.Figure: 可視化された図。
    """
    fig, ax = map_info.plot_map()
    map_info.evacuee_gdf.plot(ax=ax, color="red", markersize=30, label="避難者グループ")

    fontsize = 5
    plusy = 0.0002
    for i in range(len(shelter_coords)):
        if pulp.value(x[i]) > 0:
            ax.scatter(
                shelter_coords[i, 0],
                shelter_coords[i, 1],
                c="green",
                s=30,
                label="設置した避難所" if i == 0 else "",
                alpha=0.6,
            )
            # 避難所名を表示 (中央に表示)
            ax.text(
                shelter_coords[i, 0],
                shelter_coords[i, 1] + plusy,
                f"避難所: {i} 収容人数: {c[i]}",
                fontsize=fontsize,
                horizontalalignment="center",
            )
        else:
            ax.scatter(
                shelter_coords[i, 0],
                shelter_coords[i, 1],
                c="black",
                s=30,
                alpha=0.6,
            )
            ax.text(
                shelter_coords[i, 0],
                shelter_coords[i, 1],
                f"避難所: {i} (未選択), 収容人数: {c[i]}",
                fontsize=fontsize,
                horizontalalignment="center",
            )

    for j in range(len(group_coords)):
        for i in range(len(shelter_coords)):
            if pulp.value(y[(i, j)]) > 0:
                shelter_node = ox.distance.nearest_nodes(
                    map_info.graph, shelter_coords[i, 0], shelter_coords[i, 1]
                )
                group_node = ox.distance.nearest_nodes(
                    map_info.graph, group_coords[j, 0], group_coords[j, 1]
                )
                route = nx.shortest_path(map_info.graph, shelter_node, group_node)
                ox.plot_graph_route(
                    map_info.graph,
                    route,
                    ax=ax,
                    route_linewidth=2,
                    node_size=0,
                    route_color="y",
                    label="避難ルート",
                )

    # 避難の総距離をタイトルに追加
    total_distance = 0
    for j in range(len(group_coords)):
        for i in range(len(shelter_coords)):
            total_distance += pulp.value(y[(i, j)]) * group_populations[j] * distance_matrix[i, j]
    title += f" (避難総距離: {total_distance:.2f})"

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True)
    return fig


def set_page_config() -> None:
    """
    ページ全体の設定を行う関数。
    """
    st.set_page_config(
        page_title="避難所配置問題の最適化",
        page_icon=":hospital:",
    )


def get_parameters() -> tuple[int, int, int, int, int, int]:
    """
    パラメータを取得する関数。

    Returns:
    tuple[int, int, int, int]: seed、避難所の候補地の数、避難者グループの数、各避難所の収容人数上限、各避難者グループから避難所までの距離の最大値。
    """
    seed = st.number_input("乱数シード (seed)", min_value=0, value=42)
    n = st.number_input("避難所の候補地の数 (n)", min_value=1, max_value=10000, value=5)

    
    capacity_df = pd.DataFrame(
        {f"避難所{i}": 100 for i in range(n)},
        index=["収容人数上限"],
    ).T
    capacities = st.data_editor(capacity_df, num_rows="fixed", key="capacity_df")[
        "収容人数上限"
    ].to_list()

    m = st.number_input("避難者グループの数 (m)", min_value=1, max_value=100000, value=5)
    total_population = st.number_input(
        "全避難グループの合計人数", min_value=1, max_value=1000000000, value=100
    )
    max_distance = st.number_input(
        "各避難者グループから避難所までの距離の最大値 (max_distance)",
        min_value=1000,
        max_value=10000000,
        value=3000,
    )

    # to int from Number
    seed = int(seed)
    n = int(n)
    m = int(m)
    max_distance = int(max_distance)

    return seed, n, m, capacities, total_population, max_distance


def main() -> None:
    set_page_config()
    st.title("避難所配置の最適化")

    # パラメータ入力
    with st.sidebar:
        st.write("パラメータ設定")
        seed, n, m, capacities, total_population, max_distance = get_parameters()

    if st.button("データ生成"):
        map_info, shelter_coords, group_coords, group_populations, c, d = generate_data(
            n, m, capacities, total_population, max_distance, seed
        )
        fig1 = visualize_population_data(map_info)
        st.session_state["data_fig"] = fig1
        st.session_state["data"] = (
            map_info,
            shelter_coords,
            group_coords,
            group_populations,
            c,
            d,
        )

    if "data_fig" in st.session_state:
        st.pyplot(st.session_state["data_fig"])

    with st.sidebar:
        st.write("最適化モデルの選択")
        model_option = st.selectbox("最適化モデルを選択してください", REGISTRY.keys())
        if model_option is not None:
            kwargs: dict[str, Any] = REGISTRY[model_option]["param_fn"]()  # type: ignore
            desc: str = REGISTRY[model_option]["description"]  # type: ignore

    if "data" in st.session_state:
        map_info, shelter_coords, group_coords, group_populations, c, d_ = st.session_state["data"]

        st.write("避難グループから避難所までの避難可能距離")
        columns = [f"グループ{j}" for j in range(m)]
        index = [f"避難所{i}" for i in range(n)]
        l_df = pd.DataFrame(d_, columns=columns, index=index)
        l_df = st.data_editor(l_df, num_rows="fixed")
        d = l_df.to_numpy()

        if st.button("最適化実行"):
            with st.expander("最適化問題の詳細"):
                st.markdown(desc)
            with st.spinner("最適化中..."):
                if model_option == "避難所の設置数最小化":
                    status, x, y = optimize_shelter_installation(
                        n, m, group_populations, c, d, **kwargs
                    )
                    n_shelters = int(sum([pulp.value(x[i]) for i in range(len(shelter_coords))]))
                    title = f"避難所の設置数最小化の結果 (避難所数: {n_shelters})"
                elif model_option == "避難距離最小化":
                    status, x, y, T = optimize_evacuation_time(
                        n, m, group_populations, c, d, **kwargs
                    )
                    title = f"避難距離最小化の結果 (最大避難距離: {T})"
                elif model_option == "満足度最大化":
                    status, x, y = optimize_satisfaction(n, m, group_populations, c, d, **kwargs)
                    title = "満足度最大化の結果"
                elif model_option == "坂上モデル":
                    status, x, y = optimize_sakaue(
                        n, m, group_populations, c=c, d=d, l=l_df.to_numpy(), **kwargs
                    )
                    title = "坂上モデルの結果"
                elif model_option == "坂上モデル2":
                    status, x, y = optimize_sakaue2(
                        n, m, group_populations, c=c, d=d, l=l_df.to_numpy(), **kwargs
                    )
                    title = "坂上モデル2の結果"

            fig2 = visualize_evacuation_plan(
                map_info, d, shelter_coords, group_coords, group_populations, x, y, c, title=title
            )
            st.session_state["status"] = status
            st.session_state["opt_fig"] = fig2

            index = [f"避難所: {i}" for i in range(n)]
            columns = [f"グループ{j}" for j in range(m)]
            x_df = pd.DataFrame(
                [pulp.value(x[i]) for i in range(len(x))],
                columns=["避難所が選択されたかどうか"],
                index=index,
            )
            y_df = pd.DataFrame(
                [[pulp.value(y[(i, j)]) for j in range(m)] for i in range(n)],
                index=index,
                columns=columns,
            )
            st.session_state["x_df"] = x_df
            st.session_state["y_df"] = y_df

    if "status" in st.session_state:
        st.write(f"最適化ステータス: {st.session_state['status']}")
    if "opt_fig" in st.session_state:
        st.pyplot(st.session_state["opt_fig"])
    if "x_df" in st.session_state:
        st.write("選択された避難所")
        st.dataframe(st.session_state["x_df"])
    if "y_df" in st.session_state:
        st.write("避難グループからの避難割合")
        st.dataframe(st.session_state["y_df"])


if __name__ == "__main__":
    main()

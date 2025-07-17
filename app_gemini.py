import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from geopandas import GeoDataFrame
from networkx import MultiDiGraph
from shapely.geometry import Point

# --- データ構造定義 ---
@dataclass
class MapInfo:
    """地図関連の地理空間データを保持するクラス。"""
    place_name: str
    gdf: GeoDataFrame
    graph: MultiDiGraph
    roads: GeoDataFrame
    rivers: GeoDataFrame

    def plot_base_map(self) -> tuple[plt.Figure, plt.Axes]:
        """ベースとなる地図をプロットする。"""
        fig, ax = plt.subplots(figsize=(12, 12))
        minx, miny, maxx, maxy = self.gdf.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        self.gdf.plot(ax=ax, facecolor="whitesmoke", edgecolor="black", linewidth=1.5, label="行政境界")
        self.rivers.plot(ax=ax, color="skyblue", linewidth=2, label="河川")
        self.roads.plot(ax=ax, color="gray", linewidth=0.5, alpha=0.6, label="道路")
        ax.set_title(f"{self.place_name} の地図")
        ax.set_xlabel("経度")
        ax.set_ylabel("緯度")
        return fig, ax

# ================================================================================
# 分析テーマ設定
# ================================================================================
THEME_REGISTRY = {
    "コンビニ配置問題(3班)": {
        "name": "コンビニ配置問題(3班)",
        "facility_tags": {"shop": "convenience"},
        "facility_label": "コンビニ",
        "demand_label": "住民グループ",
    },
    "公園配置問題(4班)": {
        "name": "公園配置問題(4班)",
        "facility_tags": {"leisure": "park"},
        "facility_label": "公園",
        "demand_label": "子供グループ",
    }
}

# ================================================================================
# データ生成
# ================================================================================
@st.cache_data(show_spinner="地図データを読み込んでいます...")
def load_map_data(place_name: str) -> MapInfo:
    """指定された地名の地理空間データを取得する。"""
    gdf = ox.geocode_to_gdf(place_name)
    graph = ox.graph_from_place(place_name, network_type="walk")
    roads = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    river_tags = {"waterway": ["river", "stream", "canal"]}
    rivers = ox.features_from_place(place_name, river_tags)
    return MapInfo(place_name, gdf, graph, roads, rivers)

@st.cache_data(show_spinner="施設と需要地のデータを生成しています...")
def generate_scenario_data(
    _map_info: MapInfo, theme_config: Dict[str, Any], n: int, m: int, total_population: int, seed: int
) -> tuple[GeoDataFrame, GeoDataFrame, np.ndarray]:
    """地図とテーマに基づいて、施設候補地と需要グループのデータを生成する。"""
    np.random.seed(seed)
    
    # テーマに応じた施設候補を取得
    sites = ox.features_from_place(_map_info.place_name, theme_config["facility_tags"])
    sites = sites[sites.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])].copy()
    sites["geometry"] = sites["geometry"].centroid
    
    n_sites = min(n, len(sites))
    if n_sites == 0:
        st.error(f"{_map_info.place_name}で{theme_config['facility_label']}候補地が見つかりませんでした。")
        return pd.DataFrame(), pd.DataFrame(), np.array([])
        
    facilities_gdf = sites.sample(n=n_sites, random_state=seed).copy()

    # 需要グループの位置をランダムに生成
    demand_points = []
    minx, miny, maxx, maxy = _map_info.gdf.total_bounds
    while len(demand_points) < m:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if _map_info.gdf.contains(p).any():
            demand_points.append(p)

    pop_ratios = np.random.rand(m)
    pop_ratios /= pop_ratios.sum()
    demand_populations = (pop_ratios * total_population).astype(int)
    
    demand_gdf = gpd.GeoDataFrame(
        {"people": demand_populations},
        geometry=demand_points,
        crs=_map_info.gdf.crs,
    )

    # 距離行列の計算
    distance_matrix = np.full((n_sites, m), 1e7)
    for i, facility in enumerate(facilities_gdf.geometry):
        facility_node = ox.distance.nearest_nodes(_map_info.graph, facility.x, facility.y)
        for j, demand in enumerate(demand_gdf.geometry):
            demand_node = ox.distance.nearest_nodes(_map_info.graph, demand.x, demand.y)
            try:
                dist = nx.shortest_path_length(_map_info.graph, facility_node, demand_node, weight="length")
                distance_matrix[i, j] = dist
            except nx.NetworkXNoPath:
                continue

    return facilities_gdf, demand_gdf, distance_matrix

# ================================================================================
# 最適化問題の定義
# ================================================================================
def optimize_convenience_store_mindistance(
    n: int, m: int, group_populations: np.ndarray, d: np.ndarray, D: int, p: int
) -> tuple[str, dict, dict, float]:
    
    model = pulp.LpProblem("Minimize_Max_Convenience_Store_Distance", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", (range(n), range(m)), 0, 1)
    T = pulp.LpVariable("T", 0)

    model += T

    model += pulp.lpSum(x[i] for i in range(n)) == p
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1
    for i in range(n):
        for j in range(m):
            model += y[i][j] <= x[i]
            model += d[i, j] * y[i][j] <= T
            if d[i, j] > D:
                model += y[i][j] == 0
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    T_value = pulp.value(T) if status == 'Optimal' else -1
    x_val = {i: pulp.value(x[i]) for i in range(n)}
    y_val = {(i, j): pulp.value(y[i][j]) for i in range(n) for j in range(m)}
    return status, x_val, y_val, T_value

def optimize_convenience_store_total_distance(
    n: int, m: int, group_populations: np.ndarray, d: np.ndarray, D: int, p: int
) -> tuple[str, dict, dict]:
    model = pulp.LpProblem("Minimize_Total_Convenience_Store_Distance", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", (range(n), range(m)), 0, 1)
    model += pulp.lpSum(d[i, j] * y[i][j] * group_populations[j] for i in range(n) for j in range(m))
    model += pulp.lpSum(x[i] for i in range(n)) == p
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1
    for i in range(n):
        for j in range(m):
            model += y[i][j] <= x[i]
            if d[i, j] > D:
                model += y[i][j] == 0
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_val = {i: pulp.value(x[i]) for i in range(n)}
    y_val = {(i, j): pulp.value(y[i][j]) for i in range(n) for j in range(m)}
    return status, x_val, y_val

def optimize_park_mindistance(
    n: int, m: int, group_populations: np.ndarray, lower_capacities: list[int], upper_capacities: list[int], d: np.ndarray, D: int, p: int
) -> tuple[str, dict, dict, float]:
    model = pulp.LpProblem("Minimize_Max_Park_Distance", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", (range(n), range(m)), 0, 1)
    T = pulp.LpVariable("T", 0)
    model += T
    model += pulp.lpSum(x[i] for i in range(n)) == p
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1
    for i in range(n):
        total_assigned = pulp.lpSum(y[i][j] * group_populations[j] for j in range(m))
        model += total_assigned >= lower_capacities[i] * x[i]
        model += total_assigned <= upper_capacities[i] * x[i]
        for j in range(m):
            model += y[i][j] <= x[i]
            model += d[i, j] * y[i][j] <= T
            if d[i, j] > D:
                model += y[i][j] == 0
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    T_value = pulp.value(T) if status == 'Optimal' else -1
    x_val = {i: pulp.value(x[i]) for i in range(n)}
    y_val = {(i, j): pulp.value(y[i][j]) for i in range(n) for j in range(m)}
    return status, x_val, y_val, T_value

def optimize_park_total_distance(
    n: int, m: int, group_populations: np.ndarray, lower_capacities: list[int], upper_capacities: list[int], d: np.ndarray, D: int, p: int
) -> tuple[str, dict, dict]:

    model = pulp.LpProblem("Minimize_Total_Park_Distance", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", (range(n), range(m)), 0, 1)
    model += pulp.lpSum(d[i, j] * y[i][j] * group_populations[j] for i in range(n) for j in range(m))
    model += pulp.lpSum(x[i] for i in range(n)) == p
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1
    for i in range(n):
        total_assigned = pulp.lpSum(y[i][j] * group_populations[j] for j in range(m))
        model += total_assigned >= lower_capacities[i] * x[i]
        model += total_assigned <= upper_capacities[i] * x[i]
        for j in range(m):
            model += y[i][j] <= x[i]
            if d[i, j] > D:
                model += y[i][j] == 0
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_val = {i: pulp.value(x[i]) for i in range(n)}
    y_val = {(i, j): pulp.value(y[i][j]) for i in range(n) for j in range(m)}
    return status, x_val, y_val

MODEL_REGISTRY_CONVENIENCE_STORE = {
    "最大移動距離の最小化": optimize_convenience_store_mindistance,
    "総移動距離の最小化": optimize_convenience_store_total_distance,
}
MODEL_REGISTRY_PARK = {
    "最大アクセス距離の最小化": optimize_park_mindistance,
    "総移動距離の最小化": optimize_park_total_distance,
}

# ================================================================================
# UI関連関数
# ================================================================================
def get_convenience_store_parameters() -> dict:
    params = {}
    params["seed"] = st.number_input("乱数シード", 0, 100, 42, key="conv_seed")
    params["n"] = st.slider("コンビニの候補地の数", 1, 100, 20, key="conv_n")
    params["m"] = st.slider("住民グループの数", 1, 200, 50, key="conv_m")
    params["total_population"] = st.number_input("全住民グループの合計人数", 100, 50000, 5000, key="conv_pop")
    params["max_distance"] = st.number_input("移動可能な最大距離 (m)", 100, 5000, 800, key="conv_dist")
    params["model_key"] = st.selectbox("最適化モデルを選択", list(MODEL_REGISTRY_CONVENIENCE_STORE.keys()), key="store_model")
    params["p"] = st.number_input("設置するコンビニの数", 1, params["n"], 10, key="conv_p")
    return params

def get_park_parameters() -> dict:
    params = {}
    params["seed"] = st.number_input("乱数シード", 0, 100, 42, key="park_seed")
    params["n"] = st.slider("公園の候補地の数", 1, 50, 10, key="park_n")
    st.write("各公園の収容人数（下限と上限）")
    initial_caps = {"収容人数下限": [50] * params["n"], "収容人数上限": [200] * params["n"]}
    capacity_df = pd.DataFrame(initial_caps, index=[f"公園{i}" for i in range(params["n"])])
    edited_df = st.data_editor(capacity_df, key="park_capacity_df")
    params["lower_capacities"] = edited_df["収容人数下限"].to_list()
    params["upper_capacities"] = edited_df["収容人数上限"].to_list()
    params["m"] = st.slider("子供グループの数", 1, 100, 20, key="park_m")
    params["total_population"] = st.number_input("全子供グループの合計人数", 100, 10000, 1000, key="park_pop")
    params["max_distance"] = st.number_input("移動可能な最大距離 (m)", 100, 5000, 1000, key="park_dist")
    params["model_key"] = st.selectbox("最適化モデルを選択", list(MODEL_REGISTRY_PARK.keys()), key="park_model")
    params["p"] = st.number_input("設置する公園の数", 1, params["n"], 5, key="park_p")
    return params

def visualize_initial_data(map_info: MapInfo, facilities_gdf: GeoDataFrame, demand_gdf: GeoDataFrame, theme_config: Dict[str, Any]):
    fig, ax = map_info.plot_base_map()
    facilities_gdf.plot(ax=ax, color="blue", marker="s", markersize=80, label=f"{theme_config['facility_label']}候補地")
    demand_gdf.plot(ax=ax, color="red", marker="o", markersize=40, label=theme_config['demand_label'])
    for idx, row in facilities_gdf.iterrows():
        ax.text(row.geometry.x, row.geometry.y, f"  {idx}", fontsize=9, color="darkblue", weight='bold')
    for idx, row in demand_gdf.iterrows():
        ax.text(row.geometry.x, row.geometry.y, f"  {idx}", fontsize=9, color="darkred")
    ax.legend()
    return fig

def visualize_optimized_result(map_info: MapInfo, facilities_gdf: GeoDataFrame, demand_gdf: GeoDataFrame, x: dict, y: dict, title: str, theme_config: Dict[str, Any]):
    fig, ax = map_info.plot_base_map()
    demand_gdf.plot(ax=ax, color="red", marker="o", markersize=40, label=theme_config['demand_label'])
    
    # 凡例用のダミープロット
    ax.scatter([], [], c="green", marker="s", s=100, label=f"設置された{theme_config['facility_label']}")
    ax.scatter([], [], c="gray", marker="x", s=100, label=f"未設置の候補地")

    for i, facility in facilities_gdf.iterrows():
        is_opened = x.get(i, 0) > 0.5
        color = "green" if is_opened else "gray"
        marker = "s" if is_opened else "x"
        ax.scatter(facility.geometry.x, facility.geometry.y, c=color, marker=marker, s=120, zorder=5, alpha=0.9)
        ax.text(facility.geometry.x, facility.geometry.y, f"  {i}", fontsize=9, weight='bold')

    for (i, j), assigned in y.items():
        if assigned > 0.5:
            facility_geom = facilities_gdf.geometry.iloc[i]
            demand_geom = demand_gdf.geometry.iloc[j]
            ax.plot([facility_geom.x, demand_geom.x], [demand_geom.y, facility_geom.y], 'k--', linewidth=0.6, alpha=0.7)
    
    ax.legend()
    ax.set_title(title)
    return fig

# ================================================================================
# main(アプリケーション設定)
# ================================================================================
def main() -> None:
    st.set_page_config(layout="wide", page_title="施設配置最適化")
    st.title("施設配置の最適化シミュレーション")

    with st.sidebar:
        st.header("1. 分析テーマ選択")
        theme_key = st.selectbox("分析したいテーマを選択", list(THEME_REGISTRY.keys()))
        selected_theme = THEME_REGISTRY[theme_key]

        st.header("2. パラメータ設定")
        if theme_key == "コンビニ配置問題(3班)":
            params = get_convenience_store_parameters()
        elif theme_key == "公園配置問題(4班)":
            params = get_park_parameters()
        else:
            params = {}

        if st.button("分析を実行"):
            st.session_state["execute"] = True
            st.session_state["params"] = params
            st.session_state["theme_key"] = theme_key
            st.session_state["selected_theme"] = selected_theme
    
    if st.session_state.get("execute"):
        params = st.session_state["params"]
        selected_theme = st.session_state["selected_theme"]
        theme_key = st.session_state["theme_key"]
        
        # データ生成
        map_info = load_map_data("国立市, 東京都") # 場所は固定
        facilities_gdf, demand_gdf, distances = generate_scenario_data(
            map_info, selected_theme, params["n"], params["m"], params["total_population"], params["seed"]
        )
        
        # 最適化実行
        model_key = params["model_key"]
        if theme_key == "コンビニ配置問題(3班)":
            opt_func = MODEL_REGISTRY_CONVENIENCE_STORE[model_key]
            opt_args = {
                "n": params["n"], "m": params["m"], "group_populations": demand_gdf["people"].values,
                "d": distances, "D": params["max_distance"], "p": params.get("p", params["n"])
            }
            if "mindistance" in opt_func.__name__:
                status, x, y, T = opt_func(**opt_args)
                title = f"{model_key}の結果 (最大距離: {T:.0f}m)"
            else:
                status, x, y = opt_func(**opt_args)
                title = f"{model_key}の結果"

        elif theme_key == "公園配置問題(4班)":
            opt_func = MODEL_REGISTRY_PARK[model_key]
            opt_args = {
                "n": params["n"], "m": params["m"], "group_populations": demand_gdf["people"].values,
                "lower_capacities": params["lower_capacities"], "upper_capacities": params["upper_capacities"],
                "d": distances, "D": params["max_distance"], "p": params.get("p", params["n"])
            }
            if "mindistance" in opt_func.__name__:
                status, x, y, T = opt_func(**opt_args)
                title = f"{model_key}の結果 (最大距離: {T:.0f}m)"
            else:
                status, x, y = opt_func(**opt_args)
                title = f"{model_key}の結果"

        # 結果表示
        st.header(f"分析結果: {theme_key}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("初期配置")
            fig_initial = visualize_initial_data(map_info, facilities_gdf, demand_gdf, selected_theme)
            st.pyplot(fig_initial)
        
        with col2:
            st.subheader("最適化結果")
            if status == "Optimal":
                st.success(f"最適解が見つかりました！")
                fig_opt = visualize_optimized_result(map_info, facilities_gdf, demand_gdf, x, y, title, selected_theme)
                st.pyplot(fig_opt)
            else:
                st.error(f"最適解が見つかりませんでした (ステータス: {status})。")

    else:
        st.info("サイドバーでテーマを選択し、パラメータを設定後、「分析を実行」ボタンを押してください。")

if __name__ == "__main__":
    main()

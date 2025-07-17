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

# --- データ構造定義 ---
@dataclass
class MapInfo:
    """地図関連の地理空間データを保持するクラス。"""
    place_name: str
    gdf: GeoDataFrame
    graph: MultiDiGraph
    roads: GeoDataFrame
    rivers: GeoDataFrame


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
# ================================================================================
# 分析テーマ定義
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

# --- データ生成関連 ---
@st.cache_data(show_spinner="地図データを読み込んでいます...")
def load_map_data(place_name: str) -> MapInfo:


@st.cache_data(show_spinner="施設と需要地のデータを生成しています...")
def generate_scenario_data

# ================================================================================
# 最適化問題定義関数
# ================================================================================
# コンビニへの最大距離を最小化する最適化関数
def optimize_convenience_store_mindistance(
    n: int,
    m: int,
    group_populations: np.ndarray,
    d: np.ndarray,
    D: int,
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable], float]:
    """
    住民がコンビニまで移動する距離の最大値を最小化するための最適化を行う関数。

    Args:
    n (int): コンビニの候補地の数。
    m (int): 住民グループの数。
    group_populations (np.ndarray): 住民グループの人口。
    d (np.ndarray): 住民グループからコンビニまでの距離行列。
    D (int): 移動可能な絶対的な最大距離。

    Returns:
    tuple[str, dict, dict, float]:
        最適化ステータス、コンビニの設置決定変数(x)、住民グループの割り当て変数(y)、最小化された最大移動距離(T)。
    """
    # 1. モデルの定義
    model = pulp.LpProblem("Minimize_Max_Convenience_Store_Distance", pulp.LpMinimize)

    # 2. 変数の定義
    # x_i: 候補地iにコンビニを設置するか (1/0)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    
    # y_ij: 住民グループjがコンビニiに割り当てられる割合 (0-1)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    
    # z_ij: 住民グループjがコンビニiに割り当てられるか (1/0)
    # y_ij > 0 のときに z_ij = 1 となるように制約で関連付ける
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)

    # T: 全ての住民の移動距離の最大値
    T = pulp.LpVariable("T", lowBound=0, cat=pulp.LpContinuous)

    # 3. 目的関数の設定
    # 住民が移動する距離の最大値を最小化する
    model += T

    # 4. 制約条件の追加
    # 4.1. 住民グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1, f"Assign_Group_{j}"

    for i in range(n):
        for j in range(m):
            # 4.2. 出店制約 (y_ij > 0 なら x_i = 1)
            model += y[i][j] <= x[i], f"Link_y_x_{i}_{j}"

            # 4.3. z変数との連携 (y_ij > 0 なら z_ij = 1)
            model += y[i][j] <= z[i][j], f"Link_y_z_{i}_{j}"
            model += z[i][j] <= x[i], f"Link_z_x_{i}_{j}" # z_ij=1ならx_i=1 (冗長だが明確化のため)

            # 4.4. 絶対距離制約 (D)
            # z_ij=1 の場合のみ、距離がD以下でなければならない
            # d[i,j]は定数なので、z_ij=1のときd[i,j]<=Dとなる。
            # もしd[i,j] > D ならば、z_ijは0でなければならない。
            if d[i, j] > D:
                model += z[i][j] == 0, f"Absolute_Distance_{i}_{j}"

            # 4.5. 最大移動距離の定義 (T)
            # z_ij=1 の場合、その移動距離 d[i,j] は T 以下でなければならない
            model += d[i, j] * z[i][j] <= T, f"Max_Distance_Def_{i}_{j}"

    # 5. 最適化の実行
    model.solve()

    # 6. 結果の返却
    status = pulp.LpStatus[model.status]
    T_value = pulp.value(T)
    
    # pulp変数を辞書に変換
    x_val = {i: x[i] for i in range(n)}
    y_val = {(i, j): y[i][j] for i in range(n) for j in range(m)}

    return status, x_val, y_val, T_value

# コンビニへの総移動距離を最小化する最適化関数
def optimize_convenience_store_total_distance(
    n: int,
    m: int,
    group_populations: np.ndarray,
    d: np.ndarray,
    D: int,
    p: int, # 設置するコンビニの数
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]:
    """
    住民がコンビニまで移動する距離の合計値を最小化するための最適化を行う関数。

    Args:
    n (int): コンビニの候補地の数。
    m (int): 住民グループの数。
    group_populations (np.ndarray): 住民グループの人口。
    d (np.ndarray): 住民グループからコンビニまでの距離行列。
    D (int): 移動可能な絶対的な最大距離。
    p (int): 設置するコンビニの数。

    Returns:
    tuple[str, dict, dict]:
        最適化ステータス、コンビニの設置決定変数(x)、住民グループの割り当て変数(y)。
    """
    # 1. モデルの定義
    model = pulp.LpProblem("Minimize_Total_Convenience_Store_Distance", pulp.LpMinimize)

    # 2. 変数の定義
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )

    # 3. 目的関数の設定
    # (距離 * 人口) の合計を最小化する
    model += pulp.lpSum(
        d[i, j] * y[i][j] * group_populations[j] for i in range(n) for j in range(m)
    )

    # 4. 制約条件の追加
    # 4.1. 設置するコンビニの数を指定
    model += pulp.lpSum(x[i] for i in range(n)) == p, "Num_Stores_Constraint"
    
    # 4.2. 住民グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1, f"Assign_Group_{j}"

    for i in range(n):
        for j in range(m):
            # 4.3. 出店制約 (y_ij > 0 なら x_i = 1)
            model += y[i][j] <= x[i], f"Link_y_x_{i}_{j}"

            # 4.4. 絶対距離制約 (D)
            # もしd[i,j] > D ならば、y_ijは0でなければならない。
            if d[i, j] > D:
                model += y[i][j] == 0, f"Absolute_Distance_{i}_{j}"

    # 5. 最適化の実行
    model.solve()

    # 6. 結果の返却
    status = pulp.LpStatus[model.status]
    
    # pulp変数を辞書に変換
    x_val = {i: x[i] for i in range(n)}
    y_val = {(i, j): y[i][j] for i in range(n) for j in range(m)}

    return status, x_val, y_val

# 公園への最大距離を最小化する最適化関数
def optimize_park_mindistance(
    n: int,
    m: int,
    group_populations: np.ndarray,
    lower_capacities: list[int],
    upper_capacities: list[int],
    d: np.ndarray,
    D: int,
    p: int, # 設置する公園の数
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable], float]:
    """
    子供が公園まで移動する距離の最大値を最小化するための最適化を行う関数。

    Args:
    n (int): 公園の候補地の数。
    m (int): 子供グループの数。
    group_populations (np.ndarray): 子供グループの人口。
    lower_capacities (list[int]): 各公園の収容人数の下限。
    upper_capacities (list[int]): 各公園の収容人数の上限。
    d (np.ndarray): 子供グループから公園までの距離行列。
    D (int): 移動可能な絶対的な最大距離。
    p (int): 設置する公園の数。

    Returns:
    tuple[str, dict, dict, float]:
        最適化ステータス、公園の設置決定変数(x)、子供グループの割り当て変数(y)、最小化された最大移動距離(T)。
    """
    # 1. モデルの定義
    model = pulp.LpProblem("Minimize_Max_Park_Distance", pulp.LpMinimize)

    # 2. 変数の定義
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)
    T = pulp.LpVariable("T", lowBound=0, cat=pulp.LpContinuous)

    # 3. 目的関数の設定
    model += T

    # 4. 制約条件の追加
    # 4.1. 設置する公園の数を指定
    model += pulp.lpSum(x[i] for i in range(n)) == p, "Num_Parks_Constraint"

    # 4.2. 子供グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1, f"Assign_Child_Group_{j}"

    for i in range(n):
        # 4.3. 公園の収容人数制約 (下限と上限)
        total_assigned_population = pulp.lpSum(y[i][j] * group_populations[j] for j in range(m))
        # 設置される場合(x[i]=1)のみ、収容人数制約が有効になる
        model += total_assigned_population >= lower_capacities[i] * x[i], f"Lower_Capacity_{i}"
        model += total_assigned_population <= upper_capacities[i] * x[i], f"Upper_Capacity_{i}"

        for j in range(m):
            # 4.4. 設置制約 (y_ij > 0 なら x_i = 1)
            model += y[i][j] <= x[i], f"Link_y_x_{i}_{j}"

            # 4.5. z変数との連携
            model += y[i][j] <= z[i][j], f"Link_y_z_{i}_{j}"
            model += z[i][j] <= x[i], f"Link_z_x_{i}_{j}"

            # 4.6. 絶対距離制約 (D)
            if d[i, j] > D:
                model += z[i][j] == 0, f"Absolute_Distance_{i}_{j}"

            # 4.7. 最大移動距離の定義 (T)
            model += d[i, j] * z[i][j] <= T, f"Max_Distance_Def_{i}_{j}"

    # 5. 最適化の実行
    model.solve()

    # 6. 結果の返却
    status = pulp.LpStatus[model.status]
    T_value = pulp.value(T)
    
    x_val = {i: x[i] for i in range(n)}
    y_val = {(i, j): y[i][j] for i in range(n) for j in range(m)}

    return status, x_val, y_val, T_value

# 公園への総移動距離を最小化する最適化関数
def optimize_park_total_distance(
    n: int,
    m: int,
    group_populations: np.ndarray,
    lower_capacities: list[int],
    upper_capacities: list[int],
    d: np.ndarray,
    D: int,
    p: int, # 設置する公園の数
) -> tuple[str, dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]:
    """
    子供が公園まで移動する距離の合計値を最小化するための最適化を行う関数。

    Args:
    n (int): 公園の候補地の数。
    m (int): 子供グループの数。
    group_populations (np.ndarray): 子供グループの人口。
    lower_capacities (list[int]): 各公園の収容人数の下限。
    upper_capacities (list[int]): 各公園の収容人数の上限。
    d (np.ndarray): 子供グループから公園までの距離行列。
    D (int): 移動可能な絶対的な最大距離。
    p (int): 設置する公園の数。

    Returns:
    tuple[str, dict, dict]:
        最適化ステータス、公園の設置決定変数(x)、子供グループの割り当て変数(y)。
    """
    # 1. モデルの定義
    model = pulp.LpProblem("Minimize_Total_Park_Distance", pulp.LpMinimize)

    # 2. 変数の定義
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )

    # 3. 目的関数の設定
    # (距離 * 人口) の合計を最小化する
    model += pulp.lpSum(
        d[i, j] * y[i][j] * group_populations[j] for i in range(n) for j in range(m)
    )

    # 4. 制約条件の追加
    # 4.1. 設置する公園の数を指定
    model += pulp.lpSum(x[i] for i in range(n)) == p, "Num_Parks_Constraint"

    # 4.2. 子供グループの割り当て制約
    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1, f"Assign_Child_Group_{j}"

    for i in range(n):
        # 4.3. 公園の収容人数制約 (下限と上限)
        total_assigned_population = pulp.lpSum(y[i][j] * group_populations[j] for j in range(m))
        model += total_assigned_population >= lower_capacities[i] * x[i], f"Lower_Capacity_{i}"
        model += total_assigned_population <= upper_capacities[i] * x[i], f"Upper_Capacity_{i}"

        for j in range(m):
            # 4.4. 設置制約 (y_ij > 0 なら x_i = 1)
            model += y[i][j] <= x[i], f"Link_y_x_{i}_{j}"

            # 4.5. 絶対距離制約 (D)
            if d[i, j] > D:
                model += y[i][j] == 0, f"Absolute_Distance_{i}_{j}"

    # 5. 最適化の実行
    model.solve()

    # 6. 結果の返却
    status = pulp.LpStatus[model.status]
    
    x_val = {i: x[i] for i in range(n)}
    y_val = {(i, j): y[i][j] for i in range(n) for j in range(m)}

    return status, x_val, y_val

MODEL_REGISTRY_convenience_store = {
    "コンビニへの最大距離を最小化": optimize_convenience_store_mindistance,
    "コンビニへの総移動距離を最小化": optimize_convenience_store_total_distance,
}
MODEL_REGISTRY_park = {
    "公園への最大距離を最小化": optimize_park_mindistance,
    "公園への総移動距離を最小化": optimize_park_total_distance,
}


# ================================================================================
# パラメータ取得関数
# ================================================================================
def get_convenience_store_parameters() -> dict:
    """
    コンビニ配置問題のパラメータを取得する関数

    Returns:
        dict: seed, n, m, total_population, max_distance
    """
    seed = st.number_input("乱数シード (seed)", min_value=0, value=42)
    n = st.number_input("コンビニの候補地の数 (n)", min_value=1, max_value=1000, value=10)
    m = st.number_input("住民グループの数 (m)", min_value=1, max_value=100000, value=5)
    total_population = st.number_input(
        "全住民グループの合計人数", min_value=1, max_value=1000000000, value=100
    )
    max_distance = st.number_input(
        "各住民グループからコンビニまでの距離の最大値 (max_distance)",
        min_value=1000,
        max_value=10000000,
        value=3000,
    )

    seed = int(seed)
    n = int(n)
    m = int(m)
    total_population = int(total_population)
    max_distance = int(max_distance)

    return {
        "seed": seed,
        "n": n,
        "m": m,
        "total_population": total_population,
        "max_distance": max_distance
    }

def get_park_parameters() -> dict:
    """
    公園配置問題のパラメータを取得する関数

    Returns:
        dict: seed, n, m, lower_capacities, upper_capacities, total_population, max_distance
    """
    seed = st.number_input("乱数シード (seed)", min_value=0, value=42)
    n = st.number_input("公園の候補地の数 (n)", min_value=1, max_value=1000, value=10)
    initial_capacities = {
        "収容人数下限": [50] * n,
        "収容人数上限": [200] * n,
    }
    capacity_df = pd.DataFrame(
        initial_capacities,
        index=[f"公園{i}" for i in range(n)]
    )
    edited_df = st.data_editor(capacity_df, key="park_capacity_df")
    lower_capacities = edited_df["収容人数下限"].to_list()
    upper_capacities = edited_df["収容人数上限"].to_list()
    m = st.number_input("子供グループの数 (m)", min_value=1, max_value=100000, value=5)
    total_population = st.number_input(
        "全子供グループの合計人数", min_value=1, max_value=1000000000, value=100
    )
    max_distance = st.number_input(
        "各子供グループから公園までの距離の最大値 (max_distance)",
        min_value=1000,
        max_value=10000000,
        value=3000,
    )

    seed = int(seed)
    n = int(n)
    m = int(m)
    lower_capacities = [int(cp) for cp in lower_capacities]
    upper_capacities = [int(cp) for cp in upper_capacities]
    total_population = int(total_population)
    max_distance = int(max_distance)

    return{
        "seed": seed,
        "n": n,
        "m": m,
        "lower_capacities": lower_capacities,
        "upper_capacities": upper_capacities,
        "total_population": total_population,
        "max_distance": max_distance
    }


# ================================================================================
# main(アプリケーション設定)
# ================================================================================
def main() -> None:
    st.set_page_config()
    st.title("施設配置の最適化シミュレーション")
    # テーマ設定
    with st.sidebar:
        st.header("1. 分析テーマ（班）選択")
        theme_key = st.selectbox("分析したいテーマ（班番号）を選択", list(THEME_REGISTRY.keys()))
        selected_theme = THEME_REGISTRY[theme_key]

        st.header("2. パラメータ設定")
        if theme_key == "コンビニ配置問題(3班)":
            params = get_convenience_store_parameters()
        elif theme_key == "公園配置問題(4班)":
            params = get_park_parameters()

        st.write("3. 最適化モデルの選択")
        if theme_key == "コンビニ配置問題(3班)":
            model_options = st.selectbox("最適化モデルを選択", MODEL_REGISTRY_convenience_store.keys())
        elif theme_key == "公園配置問題(4班)":
            model_options = st.selectbox("最適化モデルを選択", MODEL_REGISTRY_park.keys())
        
        if st.button("データ生成"):
            st.session_state["params"] = params
            st.session_state["selected_theme"] = selected_theme
            st.session_state["model_options"] = model_options

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("初期データ")
        if st.button("シナリオデータを生成・表示"):
            try:
                map_info = load_map_data(place_name)
                facilities_gdf, demand_gdf, distances = generate_scenario_data(
                    map_info, selected_theme, n, m, capacities, total_population, seed
                )
                if not facilities_gdf.empty:
                    st.session_state["map_info"] = map_info
                    st.session_state["facilities_gdf"] = facilities_gdf
                    st.session_state["demand_gdf"] = demand_gdf
                    st.session_state["distances"] = distances
                    st.session_state["initial_fig"] = visualize_initial_data(map_info, facilities_gdf, demand_gdf, selected_theme)
            except Exception as e:
                st.error(f"データの生成中にエラーが発生しました: {e}")
        if "initial_fig" in st.session_state:
            st.pyplot(st.session_state["initial_fig"])

    with col2:
        st.subheader("最適化結果")
        if submit_button and "map_info" in st.session_state:
            with st.spinner("最適化計算を実行中..."):
                map_info, facilities_gdf, demand_gdf, distances = (
                    st.session_state["map_info"], st.session_state["facilities_gdf"],
                    st.session_state["demand_gdf"], st.session_state["distances"]
                )
                status, x, y, other_results = selected_model.solve(
                    n=len(facilities_gdf), m=len(demand_gdf),
                    populations=demand_gdf["people"].values,
                    capacities=facilities_gdf["capacity"].values,
                    distances=distances, **model_params
                )
                if pulp.LpStatus[status] == "Optimal":
                    st.session_state["optimized_fig"] = visualize_optimized_result(
                        map_info, facilities_gdf, demand_gdf, distances, x, y, selected_model.name, selected_theme
                    )
                    st.session_state["other_results"] = other_results
                    st.session_state["status"] = "Optimal"
                else:
                    st.session_state["status"] = pulp.LpStatus[status]

        if "status" in st.session_state:
            if st.session_state["status"] == "Optimal":
                st.success(f"最適化完了！ (ステータス: {st.session_state['status']})")
                if st.session_state.get("other_results"):
                    for key, value in st.session_state["other_results"].items():
                        st.metric(label=key, value=value)
                st.pyplot(st.session_state["optimized_fig"])
            else:
                st.error(f"最適解が見つかりませんでした (ステータス: {st.session_state['status']})。パラメータを調整してください。")

if __name__ == "__main__":
    main()

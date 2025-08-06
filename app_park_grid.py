import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit as st
from typing import Dict, Tuple

# ランドマークの定義
LANDMARK_DEFINITIONS = {
    "stations": {"label": "駅", "color": "purple", "marker": "P", "influence": 2.0},
    "schools": {"label": "学校", "color": "orange", "marker": "s", "influence": 1.5},
    "hospitals": {"label": "病院", "color": "green", "marker": "H", "influence": 1.0},
    "supermarkets": {"label": "スーパー", "color": "cyan", "marker": "D", "influence": -2.0},
}


@st.cache_data
def generate_park_data(
    area_size: int,
    grid_resolution: int,
    landmark_counts: Dict[str, int],
    m: int,  # 需要地の数
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    公園配置のための人工データを生成する。
    需要は「住民」とし、「需要地」の座標と「需要量（人口など）」を生成する。
    候補地の収容人数はここでは生成しない。
    """
    np.random.seed(seed)

    # 1. ランドマークの配置
    landmarks = {
        key: np.random.rand(count, 2) * area_size for key, count in landmark_counts.items()
    }

    # 2. 候補地のグリッド配置とランドマークに基づく間引き
    grid_points = np.linspace(0, area_size, grid_resolution)
    initial_candidates = np.array(np.meshgrid(grid_points, grid_points)).T.reshape(-1, 2)

    final_candidates_list = []
    for cand_coord in initial_candidates:
        prob = 0.2
        for key, definition in LANDMARK_DEFINITIONS.items():
            if landmark_counts.get(key, 0) > 0:
                dist = np.min(np.linalg.norm(cand_coord - landmarks[key], axis=1))
                influence = definition["influence"]
                prob += influence * np.exp(-(dist**2) / (2 * (area_size * 0.15) ** 2))

        if np.random.rand() < prob:
            final_candidates_list.append(cand_coord)

    candidate_coords = (
        np.array(final_candidates_list) if final_candidates_list else np.empty((0, 2))
    )
    n = len(candidate_coords)
    if n == 0:
        return (np.empty((0, 2)), np.empty((0, 2)), np.empty(0), np.empty((0, 0)), landmarks)

    # 3. 需要地（住民が集中するエリア）と需要量（人口など）の生成
    demand_coords = np.random.rand(m, 2) * area_size
    mean_pop = [50, 100]
    std_pop = [10, 20]
    weights_pop = [0.5, 0.5]  # パラメータを住民向けに調整
    demand_populations = np.array(
        [
            max(10, int(np.random.normal(mean_pop[c], std_pop[c])))
            for c in np.random.choice(len(weights_pop), m, p=weights_pop)
        ]
    )

    # 4. 距離行列の計算
    d = np.linalg.norm(
        candidate_coords[:, np.newaxis, :] - demand_coords[np.newaxis, :, :], axis=2
    )

    return candidate_coords, demand_coords, demand_populations, d, landmarks


# --- 数理モデルの説明 ---
DESC_PARK_MAX_DIST = """
### 最大利用距離最小化問題 (p-センター問題)

#### 概要
全ての住民がどこかの公園を利用できる条件下で、移動距離の最大値を最小化します。

#### 定数
- $I$: 公園の候補地集合
- $J$: 需要地の集合（住民が集中するエリア）
- $p_j$: 需要地$j$の住民の需要量（人口など）
- $c_i^u$: 公園$i$の最大受け入れ人数
- $c_i^l$: 公園$i$の最低必要利用人数
- $d_{ij}$: 公園$i$と需要地$j$の距離

#### 変数
- $x_i$: 候補地$i$に公園を設置する場合 1, しない場合 0
- $y_{ij}$: 需要地$j$の住民が公園$i$を利用する割合 (0~1)
- $z_{ij}$: 需要地$j$が公園$i$を利用可能な場合 1, そうでない場合 0
- $H$: 最大利用距離

#### 定式化
**目的関数: 最大移動距離の最小化**
$$\\text{Minimize} \\quad H$$

**制約条件:**
1. **住民の割り当て**: 全ての需要地の住民がいずれかの公園を利用する
   $$\\sum_{i \\in I} y_{ij} = 1 \\quad (\\forall j \\in J)$$
2. **収容人数（上限）**: 公園の利用人数は、最大受け入れ人数を超えてはならない
   $$\\sum_{j \\in J} p_j y_{ij} \\leq c_i^u \\quad (\\forall i \\in I)$$
3. **最低利用人数**: 公園を設置する場合、最低利用人数を満たさなければならない
   $$\\sum_{j \\in J} p_j y_{ij} \\geq c_i^l x_i \\quad (\\forall i \\in I)$$
4. **設置と利用の関係**: 公園が設置されている場合のみ、その公園は利用可能
   $$y_{ij} \\leq z_{ij}, \\quad z_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
5. **最大移動距離**: 全ての住民の移動距離は$H$以下でなければならない
   $$d_{ij} z_{ij} \\leq H \\quad (\\forall i \\in I, \\forall j \\in J)$$
"""


def optimize_max_dist(
    n: int, m: int, p: np.ndarray, c_phy: np.ndarray, c_low: np.ndarray, d: np.ndarray
) -> Tuple[str, Dict, Dict, float]:
    """
    p-センター問題（最大利用距離の最小化）を解く。
    """
    model = pulp.LpProblem("Minimize_Max_Park_Distance", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)
    # 修正点1: 変数名をTからHに変更
    H = pulp.LpVariable("H", lowBound=0, cat=pulp.LpContinuous)

    model += H

    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        total_users = pulp.lpSum(p[j] * y[i][j] for j in range(m))
        model += total_users <= c_phy[i]
        model += total_users >= c_low[i] * x[i]

        for j in range(m):
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            # 修正点1: 制約式内の変数名をTからHに変更
            model += d[i][j] * z[i][j] <= H

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_sol = {i: x[i] for i in range(n)}
    y_sol = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    # 修正点1: 戻り値の変数名をTからHに変更
    return status, x_sol, y_sol, H.value() if H.value() is not None else -1.0


def visualize_initial_data(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_populations: np.ndarray,
    landmarks: Dict[str, np.ndarray],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        demand_coords[:, 0],
        demand_coords[:, 1],
        c="blue",
        s=demand_populations * 0.5,
        label="需要地（住民）",
        alpha=0.6,
    )
    ax.scatter(
        candidate_coords[:, 0],
        candidate_coords[:, 1],
        c="red",
        s=30,
        label="公園候補地",
        alpha=0.7,
        marker="s",
    )
    for key, coords in landmarks.items():
        if coords.size > 0:
            definition = LANDMARK_DEFINITIONS[key]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=definition["color"],
                s=200,
                label=definition["label"],
                marker=definition["marker"],
                edgecolors="black",
            )
    ax.legend()
    ax.set_xlabel("X座標")
    ax.set_ylabel("Y座標")
    ax.set_title("初期データ：候補地・需要地・ランドマークの配置")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


def visualize_optimization_result(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_populations: np.ndarray,
    x: Dict,
    y: Dict,
    landmarks: Dict,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter([], [], c="red", s=150, label="設置された公園", marker="s", edgecolors="black")
    ax.scatter([], [], c="gray", s=100, label="設置されなかった候補地", alpha=0.4, marker="s")
    ax.scatter([], [], c="blue", s=100, label="需要地（住民）", alpha=0.7)
    for key, definition in LANDMARK_DEFINITIONS.items():
        if landmarks.get(key, np.array([])).size > 0:
            ax.scatter(
                [],
                [],
                c=definition["color"],
                s=200,
                label=definition["label"],
                marker=definition["marker"],
                edgecolors="black",
            )

    for key, coords in landmarks.items():
        if coords.size > 0:
            definition = LANDMARK_DEFINITIONS[key]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=definition["color"],
                s=200,
                marker=definition["marker"],
                edgecolors="black",
                alpha=0.7,
            )

    for i, coord in enumerate(candidate_coords):
        if pulp.value(x.get(i)) > 0.5:
            total_users = sum(
                pulp.value(y.get((i, j), 0)) * demand_populations[j]
                for j in range(len(demand_coords))
            )
            ax.scatter(
                coord[0], coord[1], c="red", s=150, alpha=0.8, marker="s", edgecolors="black"
            )  # サイズを固定値に変更
            ax.text(
                coord[0],
                coord[1] - 3,
                f"利用者: {total_users:.0f}人",
                fontsize=9,
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )  # 面積表示を削除
        else:
            ax.scatter(coord[0], coord[1], c="gray", s=20, alpha=0.4, marker="s")
    ax.scatter(
        demand_coords[:, 0], demand_coords[:, 1], c="blue", s=demand_populations * 0.5, alpha=0.6
    )
    for i, j in y.keys():
        if pulp.value(y.get((i, j), 0)) > 1e-3:
            ax.plot(
                [demand_coords[j, 0], candidate_coords[i, 0]],
                [demand_coords[j, 1], candidate_coords[i, 1]],
                color="green",
                alpha=0.6,
                linestyle="--",
            )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("X座標")
    ax.set_ylabel("Y座標")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


# UI設定関数
def set_page_config():
    st.set_page_config(page_title="公園配置 最適化シミュレーター", page_icon="🏞️", layout="wide")


def get_data_generation_params():
    st.header("データ生成パラメータ")
    seed = st.number_input("乱数シード", 0, value=42)
    area_size = st.slider("シミュレーションエリアの広さ", 50, 300, 100)
    m = st.slider("需要地（住民エリア）の数 (m)", 1, 50, 15, key="m_slider")

    st.subheader("ランドマークの数")
    cols = st.columns(len(LANDMARK_DEFINITIONS))
    landmark_counts = {
        key: col.number_input(definition["label"], 0, 10, 2, key=f"landmark_{key}")
        for col, (key, definition) in zip(cols, LANDMARK_DEFINITIONS.items())
    }

    st.subheader("公園候補地の生成")
    grid_resolution = st.slider(
        "候補地生成グリッドの解像度", 5, 30, 10, help="高いほど候補地が多くなります。"
    )

    return area_size, grid_resolution, landmark_counts, m, seed


def get_model_parameters(n_parks):
    st.header("最適化モデルのパラメータ")

    with st.expander(f"公園ごとの収容人数を設定 ({n_parks}件)", expanded=True):
        c_low_list = []
        c_phy_list = []
        for i in range(n_parks):
            cols = st.columns(2)
            with cols[0]:
                # 修正点2: 最小収容人数のデフォルト値を10に変更
                c_low = st.number_input(f"候補地 {i} の最小収容人数", 0, value=10, key=f"clow_{i}")
                c_low_list.append(c_low)
            with cols[1]:
                # 修正点2: 最大収容人数のデフォルト値を100に変更
                c_phy = st.number_input(
                    f"候補地 {i} の最大収容人数", 0, value=100, key=f"cphy_{i}"
                )
                c_phy_list.append(c_phy)

    return {"c_low_list": c_low_list, "c_phy_list": c_phy_list}


def main():
    set_page_config()
    st.title("🏞️ 公園配置 最適化シミュレーター")
    st.markdown("### 最大利用距離最小化問題 (p-センター問題)")

    with st.sidebar:
        st.title("⚙️ 設定")
        area_size, grid_res, landmark_counts, m, seed = get_data_generation_params()

    col1, col2 = st.columns(2)
    with col1:
        st.header("📍 初期データ")
        if st.button("データを生成・表示"):
            data = generate_park_data(area_size, grid_res, landmark_counts, m, seed)
            if data[0].shape[0] > 0:
                st.session_state["data_park"] = data
                candidate_coords, demand_coords, p, d, landmarks = data
                fig = visualize_initial_data(candidate_coords, demand_coords, p, landmarks)
                st.session_state["initial_fig_park"] = fig
                st.success(f"データ生成完了！ (公園候補地: {len(data[0])}件)")
            else:
                st.error(
                    "候補地が0件になりました。グリッド解像度を上げるか、ランドマークの影響を調整してください。"
                )
                if "initial_fig_park" in st.session_state:
                    del st.session_state["initial_fig_park"]

        if "initial_fig_park" in st.session_state:
            st.pyplot(st.session_state["initial_fig_park"])
        else:
            st.info("「データを生成・表示」ボタンを押してシミュレーションを開始してください。")

    with col2:
        st.header("📈 最適化結果")
        if "data_park" in st.session_state:
            n_actual = len(st.session_state["data_park"][0])
            with st.sidebar:
                model_params = get_model_parameters(n_actual)

            if st.button("最適化を実行"):
                with st.expander("最適化問題の詳細", expanded=False):
                    st.markdown(DESC_PARK_MAX_DIST, unsafe_allow_html=True)
                with st.spinner("最適化計算を実行中..."):
                    candidate_coords, demand_coords, p, d, landmarks = st.session_state[
                        "data_park"
                    ]
                    n, m = len(candidate_coords), len(demand_coords)

                    c_low = np.array(model_params["c_low_list"])
                    c_phy = np.array(model_params["c_phy_list"])

                    status, x, y, result_value = optimize_max_dist(n, m, p, c_phy, c_low, d)

                    if "Optimal" in status or "Feasible" in status:
                        title = f"最適化結果: 最大利用距離の最小化\n(最大距離: {result_value:.2f})"

                        fig = visualize_optimization_result(
                            candidate_coords, demand_coords, p, x, y, landmarks, title
                        )
                        st.session_state["status_park"] = status
                        st.session_state["result_fig_park"] = fig
                    else:
                        st.error(
                            f"最適化に失敗 (Status: {status})。制約が厳しすぎる可能性があります。"
                        )
                        if "result_fig_park" in st.session_state:
                            del st.session_state["result_fig_park"]

            if "result_fig_park" in st.session_state:
                st.success(f"最適化ステータス: **{st.session_state['status_park']}**")
                st.pyplot(st.session_state["result_fig_park"])
            else:
                st.info("「最適化を実行」ボタンを押して計算を開始してください。")
        else:
            st.warning("先に「データを生成・表示」ボタンでデータを生成してください。")


if __name__ == "__main__":
    main()

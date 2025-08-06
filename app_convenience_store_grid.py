import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit as st
from typing import Any, Dict, Tuple

# --- 定数 ---
LANDMARK_DEFINITIONS = {
    "stations": {"label": "駅", "color": "purple", "marker": "P", "influence": 2.0},
    "schools": {"label": "学校", "color": "orange", "marker": "s", "influence": 1.5},
    "hospitals": {"label": "病院", "color": "green", "marker": "H", "influence": 1.0},
    "supermarkets": {"label": "スーパー", "color": "cyan", "marker": "D", "influence": -2.0},
}


@st.cache_data
def generate_conbini_data(
    area_size: int, grid_resolution: int, landmark_counts: Dict[str, int], m: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    ランドマークを考慮したコンビニの人工データを生成する。
    """
    np.random.seed(seed)
    landmarks = {
        key: np.random.rand(count, 2) * area_size for key, count in landmark_counts.items()
    }

    grid_points = np.linspace(0, area_size, grid_resolution)
    initial_candidates = np.array(np.meshgrid(grid_points, grid_points)).T.reshape(-1, 2)

    final_candidates_list = []
    for cand_coord in initial_candidates:
        prob = 0.15
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

    demand_coords = np.random.rand(m, 2) * area_size
    demand_populations = np.random.randint(50, 200, size=m)

    d = np.linalg.norm(
        candidate_coords[:, np.newaxis, :] - demand_coords[np.newaxis, :, :], axis=2
    )
    return candidate_coords, demand_coords, demand_populations, d, landmarks


# --- 数理モデルの説明 ---
DESC_P_CENTER_STRICT = """
### 最小の最大移動距離問題（p-センター問題）

#### 前提条件
- **ビジネス的な採算性**: 出店する候補地は、その**商圏内に、指定された「最低商圏内人口」以上**の住民がいる必要があります。この条件を満たさない候補地は、最適化計算からあらかじめ除外されます。

#### 定数
- $I$: コンビニの候補地集合（採算ラインをクリアした場所のみ）
- $J$: 利用者グループの集合
- $D$: 徒歩で到達可能な最大距離
- $d_{ij}$: 候補地$i$と利用者グループ$j$の距離
- $N$: 設置するコンビニの上限店舗数

#### 変数
- $x_i$: 候補地$i$にコンビニを設置するか {0,1}
- $y_{ij}$: 利用者グループ$j$の住民がコンビニ$i$を利用する割合 [0,1]
- $z_{ij}$: 利用者グループ$j$がコンビニ$i$を利用可能か {0,1}
- $T$: 最大利用距離

#### 定式化
**目的関数: 最大利用距離の最小化**
$$\\text{min} \\quad T$$

**制約条件:**
1.  $$\\sum_{i \\in I} x_i \\leq N$$
2.  $$\\sum_{i \\in I} y_{ij} = 1 \\quad (\\forall j \\in J)$$
3.  $$y_{ij} \\leq z_{ij} \\quad (\\forall i \\in I, \\forall j \\in J)$$
4.  $$z_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
5.  $$d_{ij} z_{ij} \\leq D \\quad (\\forall i \\in I, \\forall j \\in J)$$
6.  $$d_{ij} z_{ij} \\leq T \\quad (\\forall i \\in I, \\forall j \\in J)$$
"""

DESC_P_MEDIAN_STRICT = """
### 合計移動距離を最小にする問題（p-メディアン問題）

#### 前提条件
- **ビジネス的な採算性**: 出店する候補地は、その**商圏内に、指定された「最低商圏内人口」以上**の住民がいる必要があります。この条件を満たさない候補地は、最適化計算からあらかじめ除外されます。

#### 定数
- $I$: コンビニの候補地集合（採算ラインをクリアした場所のみ）
- $J$: 利用者グループの集合
- $D$: 徒歩で到達可能な最大距離
- $d_{ij}$: 候補地$i$と利用者グループ$j$の距離
- $N$: 設置するコンビニの上限店舗数

#### 変数
- $x_i$: 候補地$i$にコンビニを設置するか {0,1}
- $y_{ij}$: 利用者グループ$j$の住民がコンビニ$i$を利用する割合 [0,1]
- $z_{ij}$: 利用者グループ$j$がコンビニ$i$を利用可能か {0,1}

#### 定式化
**目的関数: 合計移動距離の最小化**
$$\\text{min} \\quad \\sum_{i \\in I} \\sum_{j \\in J} d_{ij} z_{ij}$$

**制約条件:**
1.  $$\\sum_{i \\in I} x_i \\leq N$$
2.  $$\\sum_{i \\in I} y_{ij} = 1 \\quad (\\forall j \\in J)$$
3.  $$y_{ij} \\leq z_{ij} \\quad (\\forall i \\in I, \\forall j \\in J)$$
4.  $$z_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
5.  $$d_{ij} z_{ij} \\leq D \\quad (\\forall i \\in I, \\forall j \\in J)$$
"""


# --- 最適化関数 ---
def optimize_p_center_strict(
    n: int, m: int, dists: np.ndarray, N_stores: int, D_max: float
) -> Tuple[str, Dict, Dict, float]:
    model = pulp.LpProblem("p-center-strict", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", (range(n), range(m)), lowBound=0, upBound=1)
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)
    T = pulp.LpVariable("T", lowBound=0)

    model += T

    model += pulp.lpSum(x[i] for i in range(n)) <= N_stores

    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        for j in range(m):
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            if dists[i][j] > D_max:
                model += z[i][j] == 0
            model += dists[i][j] * z[i][j] <= T

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_sol = {i: x[i].value() for i in range(n)}
    y_sol = {(i, j): y[i][j].value() for i in range(n) for j in range(m)}
    obj_val = T.value() if T.value() is not None else -1.0
    return status, x_sol, y_sol, obj_val


def optimize_p_median_strict(
    n: int, m: int, dists: np.ndarray, N_stores: int, D_max: float
) -> Tuple[str, Dict, Dict, float]:
    model = pulp.LpProblem("p-median-strict", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", (range(n), range(m)), lowBound=0, upBound=1)
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)

    model += pulp.lpSum(dists[i][j] * z[i][j] for i in range(n) for j in range(m))

    model += pulp.lpSum(x[i] for i in range(n)) <= N_stores

    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        for j in range(m):
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            if dists[i][j] > D_max:
                model += z[i][j] == 0

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_sol = {i: x[i].value() for i in range(n)}
    y_sol = {(i, j): y[i][j].value() for i in range(n) for j in range(m)}
    obj_val = model.objective.value() if model.objective.value() is not None else -1.0
    return status, x_sol, y_sol, obj_val


# --- UI設定関数 ---
def get_data_generation_params():
    st.header("データ生成パラメータ")
    seed = st.number_input("乱数シード", 0, value=42)
    m = st.slider("住民グループの数 (m)", 1, 50, 30)
    area_size = st.slider("シミュレーションエリアの広さ", 50, 200, 100)
    st.subheader("ランドマークの数")
    cols = st.columns(len(LANDMARK_DEFINITIONS))
    landmark_counts = {
        key: col.number_input(definition["label"], 0, 10, 2, key=f"landmark_{key}")
        for col, (key, definition) in zip(cols, LANDMARK_DEFINITIONS.items())
    }
    st.subheader("コンビニ候補地の生成")
    grid_resolution = st.slider(
        "候補地生成グリッドの解像度", 5, 40, 20, help="高いほど候補地が多くなります。"
    )
    return area_size, grid_resolution, landmark_counts, m, seed


def get_model_parameters(n_candidates: int):
    st.header("最適化モデルのパラメータ")
    params = {}
    params["N"] = st.slider(
        "設置するコンビニの上限店舗数 (N)",
        1,
        n_candidates,
        min(5, n_candidates),
        help="設置するコンビニの最大数。",
    )
    params["D"] = st.number_input("住民が到達可能な最大距離 (D)", 1.0, value=250.0, format="%.1f")

    st.subheader("ビジネス上の採算性評価")
    params["R"] = st.slider(
        "商圏半径 (R)", 10, 100, 50, help="各候補地の集客範囲（この半径内の住民を対象とする）。"
    )
    params["M_min"] = st.slider(
        "最低商圏内人口", 0, 2000, 300, help="出店に必要となる商圏内の最低人口。"
    )
    return params


# --- 可視化関数 ---
def visualize_initial_data(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_populations: np.ndarray,
    landmarks: Dict,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        demand_coords[:, 0],
        demand_coords[:, 1],
        c="blue",
        s=demand_populations,
        label="住民グループ (人口)",
        alpha=0.6,
    )
    ax.scatter(
        candidate_coords[:, 0],
        candidate_coords[:, 1],
        c="red",
        s=50,
        label="コンビニ候補地",
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
    ax.set_title("初期データ：候補地・住民グループ・ランドマークの配置")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


def visualize_optimization_result(
    all_candidate_coords: np.ndarray,
    eligible_indices: np.ndarray,
    demand_coords: np.ndarray,
    demand_populations: np.ndarray,
    x: Dict,
    y: Dict,
    landmarks: Dict,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter([], [], c="red", s=100, label="設置されたコンビニ", marker="s", edgecolors="black")
    ax.scatter([], [], c="lightgray", s=50, label="採算ライン未満の候補地", alpha=0.8, marker="x")
    ax.scatter([], [], c="gray", s=50, label="設置されなかった候補地", alpha=0.5, marker="s")
    ax.scatter([], [], c="blue", s=100, label="住民グループ (人口)", alpha=0.6)

    for i in range(len(all_candidate_coords)):
        coord = all_candidate_coords[i]
        if i in eligible_indices:
            local_idx = list(eligible_indices).index(i)
            if x.get(local_idx, 0) > 0.5:
                ax.scatter(
                    coord[0], coord[1], c="red", s=100, marker="s", edgecolors="black", zorder=5
                )
                ax.text(
                    coord[0],
                    coord[1],
                    f"店{i}",
                    fontsize=10,
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                )
            else:
                ax.scatter(coord[0], coord[1], c="gray", s=50, alpha=0.5, marker="s")
        else:
            ax.scatter(coord[0], coord[1], c="lightgray", s=50, alpha=0.8, marker="x")

    ax.scatter(demand_coords[:, 0], demand_coords[:, 1], c="blue", s=demand_populations, alpha=0.6)
    for (i_local, j), assignment_ratio in y.items():
        if assignment_ratio > 1e-6:
            i_global = eligible_indices[i_local]
            ax.plot(
                [demand_coords[j, 0], all_candidate_coords[i_global, 0]],
                [demand_coords[j, 1], all_candidate_coords[i_global, 1]],
                color="green",
                alpha=0.3 + 0.7 * assignment_ratio,
                linestyle="--",
                linewidth=0.5 + 2 * assignment_ratio,
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
                alpha=0.5,
            )

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("X座標")
    ax.set_ylabel("Y座標")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


# --- Main App ---
REGISTRY: Dict[str, Dict[str, Any]] = {
    "合計移動距離を最小化": {
        "description": DESC_P_MEDIAN_STRICT,
        "func": optimize_p_median_strict,
    },
    "最大移動距離を最小化": {
        "description": DESC_P_CENTER_STRICT,
        "func": optimize_p_center_strict,
    },
}


def main() -> None:
    st.set_page_config(
        page_title="コンビニ配置 最適化シミュレーター", page_icon="🏪", layout="wide"
    )
    st.title("🏪 コンビニ配置 最適化シミュレーター")

    with st.sidebar:
        st.title("⚙️ 設定")
        model_option = st.selectbox("最適化モデルを選択してください", list(REGISTRY.keys()))
        area_size, grid_res, landmark_counts, m, seed = get_data_generation_params()

        if "data_conbini" in st.session_state:
            st.markdown("---")
            all_coords, _, _, _, _ = st.session_state["data_conbini"]
            n_total = len(all_coords) if len(all_coords) > 0 else 1
            model_params = get_model_parameters(n_total)
        else:
            model_params = None

    col1, col2 = st.columns(2)
    with col1:
        st.header("📍 初期データ")
        if st.button("データを生成・表示"):
            coords, demands, pops, dists, landmarks = generate_conbini_data(
                area_size, grid_res, landmark_counts, m, seed
            )
            if coords.shape[0] > 0:
                st.session_state["data_conbini"] = (coords, demands, pops, dists, landmarks)
                fig = visualize_initial_data(coords, demands, pops, landmarks)
                st.session_state["initial_fig_conbini"] = fig
                st.success(f"データ生成完了！ (全候補地: {len(coords)}件)")
                st.rerun()
            else:
                st.error(
                    "候補地が0件になりました。グリッド解像度を上げるか、ランドマークの影響を調整してください。"
                )
                if "initial_fig_conbini" in st.session_state:
                    del st.session_state["initial_fig_conbini"]

        if "initial_fig_conbini" in st.session_state:
            st.pyplot(st.session_state["initial_fig_conbini"])
        else:
            st.info("「データを生成・表示」ボタンを押してシミュレーションを開始してください。")

    with col2:
        st.header("📈 最適化結果")
        if "data_conbini" in st.session_state and model_params is not None:
            all_coords, all_demands, all_pops, all_dists, all_landmarks = st.session_state[
                "data_conbini"
            ]

            if st.button("最適化を実行"):
                R = model_params["R"]
                M_min = model_params["M_min"]
                eligible_indices = [
                    i
                    for i, dists_from_i in enumerate(all_dists)
                    if np.sum(all_pops[np.where(dists_from_i <= R)[0]]) >= M_min
                ]

                if not eligible_indices:
                    st.error(
                        "採算ラインをクリアした候補地が0件です。商圏半径を広げるか、最低商圏内人口を減らしてください。"
                    )
                else:
                    eligible_dists = all_dists[np.ix_(eligible_indices, range(len(all_demands)))]
                    n_eligible = len(eligible_indices)
                    m_demands = len(all_demands)

                    st.info(
                        f"全 {len(all_coords)}件の候補地のうち、採算ラインをクリアしたのは {n_eligible}件です。この中から最適配置を計算します。"
                    )

                    with st.expander("最適化問題の詳細", expanded=False):
                        st.markdown(REGISTRY[model_option]["description"], unsafe_allow_html=True)

                    with st.spinner("最適化計算を実行中..."):
                        opt_func = REGISTRY[model_option]["func"]
                        args = (
                            n_eligible,
                            m_demands,
                            eligible_dists,
                            model_params["N"],
                            model_params["D"],
                        )
                        status, x, y, obj_val = opt_func(*args)

                        if "Optimal" in status or "Feasible" in status:
                            if "最大移動距離" in model_option:
                                title = f"最適化結果: {model_option}\n(最小化された最大距離 T: {obj_val:.2f})"
                            else:
                                title = (
                                    f"最適化結果: {model_option}\n(目的関数値 Σdz: {obj_val:,.2f})"
                                )

                            fig = visualize_optimization_result(
                                all_coords,
                                np.array(eligible_indices),
                                all_demands,
                                all_pops,
                                x,
                                y,
                                all_landmarks,
                                title,
                            )
                            st.session_state["status_conbini"] = status
                            st.session_state["result_fig_conbini"] = fig
                        else:
                            st.error(
                                f"最適化に失敗 (Status: {status})。制約が厳しすぎる可能性があります。"
                            )
                            if "result_fig_conbini" in st.session_state:
                                del st.session_state["result_fig_conbini"]

            if "result_fig_conbini" in st.session_state:
                st.success(f"最適化ステータス: **{st.session_state['status_conbini']}**")
                st.pyplot(st.session_state["result_fig_conbini"])
            else:
                st.info("「最適化を実行」ボタンを押して計算を開始してください。")
        elif "data_conbini" not in st.session_state:
            st.warning("先に左側の「データを生成・表示」ボタンでデータを生成してください。")


if __name__ == "__main__":
    main()

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit as st
from typing import Any, Dict, Tuple


@st.cache_data
def generate_data(
    n: int, m: int, area_size: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    コンビニ候補地と住民グループの人工データを生成する。

    Args:
        n (int): コンビニ候補地の数。
        m (int): 住民グループの数。
        area_size (int): シミュレーションエリアの広さ（一辺の長さ）。
        seed (int): 乱数のシード。

    Returns:
        tuple[np.ndarray, ...]: 候補地の座標、住民グループの座標、住民数、距離行列。
    """
    np.random.seed(seed)
    candidate_coords = np.random.rand(n, 2) * area_size
    demand_coords = np.random.rand(m, 2) * area_size

    # 住民グループの人口をランダムに生成
    demand_populations = np.random.randint(50, 200, size=m)

    # 候補地と住民グループ間の距離行列を計算
    d = np.linalg.norm(
        candidate_coords[:, np.newaxis, :] - demand_coords[np.newaxis, :, :], axis=2
    )

    return candidate_coords, demand_coords, demand_populations, d


DESC_CONBINI_P_CENTER = """
### [cite_start]最大徒歩距離の最小化 (p-センター問題) [cite: 200]

#### 概要
[cite_start]設置する店舗数を指定し、全ての住民がいずれかの店舗を利用する上で、住民が歩く最大距離が最も小さくなるような店舗の配置を求めます [cite: 201, 203]。

#### 定数
- [cite_start]$I$: コンビニ候補地の集合 [cite: 216]
- [cite_start]$J$: 住民グループの集合 [cite: 217]
- [cite_start]$p$: 設置するコンビニの店舗数 [cite: 204]
- [cite_start]$D$: 住民が到達可能な最大距離 [cite: 218]
- [cite_start]$p_j$: 住民グループ$j$の人口 [cite: 219]
- [cite_start]$d_{ij}$: 候補地$i$と住民グループ$j$の距離 [cite: 220]

#### 変数
- [cite_start]$x_i$: 候補地$i$にコンビニを設置する場合 1, しない場合 0 [cite: 231]
- $y_{ij}$: 住民グループ$j$がコンビニ$i$を利用する場合 1, しない場合 0
- [cite_start]$T$: 全ての住民の最大利用距離 [cite: 238]

#### 定式化
[cite_start]**目的関数: 最大利用距離の最小化** [cite: 214]
$$\\text{Minimize} \\quad T$$

**制約条件:**
1. **店舗設置数**: 指定された数だけ店舗を設置する
   $$\\sum_{i \\in I} x_i = p$$
2. [cite_start]**住民の割り当て**: 全ての住民グループが、いずれか１つの設置された店舗を利用する [cite: 222, 225]
   $$\\sum_{i \\in I} y_{ij} = 1 \\quad (\\forall j \\in J)$$
   $$y_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
3. [cite_start]**最大距離制約**: 全ての住民の移動距離は$T$以下でなければならない [cite: 237]
   $$\\sum_{i \\in I} d_{ij} y_{ij} \\leq T \\quad (\\forall j \\in J)$$
4. [cite_start]**到達可能性**: 住民は、到達可能な最大距離$D$以内の店舗しか利用できない [cite: 234]
   $$d_{ij} y_{ij} \\leq D \\quad (\\forall i \\in I, \\forall j \\in J)$$
"""

DESC_CONBINI_P_MEDIAN = """
### [cite_start]総徒歩距離の最小化 (p-メディアン問題) [cite: 242]

#### 概要
[cite_start]設置する店舗数を指定し、全ての住民の移動距離の合計（総徒歩距離）が最も小さくなるような店舗の配置を求めます [cite: 243, 245]。

#### 定数
- [cite_start]$I$: コンビニ候補地の集合 [cite: 254]
- [cite_start]$J$: 住民グループの集合 [cite: 255]
- [cite_start]$p$: 設置するコンビニの店舗数 [cite: 246]
- [cite_start]$D$: 住民が到達可能な最大距離 [cite: 257]
- [cite_start]$p_j$: 住民グループ$j$の人口 [cite: 259]
- [cite_start]$d_{ij}$: 候補地$i$と住民グループ$j$の距離 [cite: 260]

#### 変数
- [cite_start]$x_i$: 候補地$i$にコンビニを設置する場合 1, しない場合 0 [cite: 262]
- $y_{ij}$: 住民グループ$j$がコンビニ$i$を利用する場合 1, しない場合 0

#### 定式化
[cite_start]**目的関数: 総徒歩距離の最小化** [cite: 268]
$$\\text{Minimize} \\quad \\sum_{i \\in I} \\sum_{j \\in J} p_j d_{ij} y_{ij}$$

**制約条件:**
1. **店舗設置数**: 指定された数だけ店舗を設置する
   $$\\sum_{i \\in I} x_i = p$$
2. [cite_start]**住民の割り当て**: 全ての住民グループが、いずれか１つの設置された店舗を利用する [cite: 269]
   $$\\sum_{i \\in I} y_{ij} = 1 \\quad (\\forall j \\in J)$$
   $$y_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
3. [cite_start]**到達可能性**: 住民は、到達可能な最大距離$D$以内の店舗しか利用できない [cite: 280]
   $$d_{ij} y_{ij} \\leq D \\quad (\\forall i \\in I, \\forall j \\in J)$$
"""


def get_model_parameters(n_candidates: int) -> Dict[str, Any]:
    st.header("最適化モデルのパラメータ")
    params: Dict[str, Any] = {}

    # スライダーの上限値を候補地の数（n）に動的に設定
    params["p"] = st.slider(
        "設置するコンビニの店舗数 (p)",
        min_value=1,
        max_value=n_candidates,
        value=min(3, n_candidates),  # デフォルト値を3とnの小さい方にする
        help="候補地の中から、実際に設置するコンビニの数を選択します。",
    )

    params["D"] = st.number_input(
        "住民が到達可能な最大距離 (D)",
        min_value=1,
        value=100,
        help="住民がコンビニまで歩ける最大の距離。これを超えると利用できません。",
    )
    return params


def optimize_p_center(
    n: int, m: int, populations: np.ndarray, distances: np.ndarray, p_stores: int, D_max: int
) -> Tuple[str, Dict, Dict, float]:
    model = pulp.LpProblem("p-center", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", (range(n), range(m)), cat=pulp.LpBinary)
    T = pulp.LpVariable("T", lowBound=0)

    # Objective Function
    model += T

    # Constraints
    model += pulp.lpSum(x[i] for i in range(n)) == p_stores

    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1
        model += pulp.lpSum(distances[i][j] * y[i][j] for i in range(n)) <= T

    for i in range(n):
        for j in range(m):
            model += y[i][j] <= x[i]
            model += distances[i][j] * y[i][j] <= D_max

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_sol = {i: x[i].value() for i in range(n)}
    y_sol = {(i, j): y[i][j].value() for i in range(n) for j in range(m)}
    obj_val = T.value() if T.value() is not None else -1.0

    return status, x_sol, y_sol, obj_val


def optimize_p_median(
    n: int, m: int, populations: np.ndarray, distances: np.ndarray, p_stores: int, D_max: int
) -> Tuple[str, Dict, Dict, float]:
    model = pulp.LpProblem("p-median", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", (range(n), range(m)), cat=pulp.LpBinary)

    # Objective Function
    model += pulp.lpSum(
        populations[j] * distances[i][j] * y[i][j] for i in range(n) for j in range(m)
    )

    # Constraints
    model += pulp.lpSum(x[i] for i in range(n)) == p_stores

    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        for j in range(m):
            model += y[i][j] <= x[i]
            model += distances[i][j] * y[i][j] <= D_max

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_sol = {i: x[i].value() for i in range(n)}
    y_sol = {(i, j): y[i][j].value() for i in range(n) for j in range(m)}
    obj_val = model.objective.value() if model.objective.value() is not None else -1.0

    return status, x_sol, y_sol, obj_val


def visualize_initial_data(
    candidate_coords: np.ndarray, demand_coords: np.ndarray, demand_populations: np.ndarray
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        demand_coords[:, 0],
        demand_coords[:, 1],
        c="blue",
        s=demand_populations,
        label="住民グループ (円の大きさは人口)",
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

    for i, (x, y) in enumerate(demand_coords):
        ax.text(x, y, f"{i}", fontsize=9, ha="center", va="center", color="white")

    ax.legend()
    ax.set_xlabel("X座標")
    ax.set_ylabel("Y座標")
    ax.set_title("初期データ：コンビニ候補地と住民グループの配置")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


def visualize_optimization_result(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_populations: np.ndarray,
    x: Dict,
    y: Dict,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10))

    # 凡例用のダミープロット
    ax.scatter([], [], c="red", s=100, label="設置されたコンビニ", marker="s", edgecolors="black")
    ax.scatter([], [], c="gray", s=50, label="設置されなかった候補地", alpha=0.5, marker="s")
    ax.scatter([], [], c="blue", s=100, label="住民グループ", alpha=0.6)

    for i, coord in enumerate(candidate_coords):
        if x.get(i, 0) > 0.5:
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

    ax.scatter(demand_coords[:, 0], demand_coords[:, 1], c="blue", s=demand_populations, alpha=0.6)

    for (i, j), is_assigned in y.items():
        if is_assigned > 0.5:
            ax.plot(
                [demand_coords[j, 0], candidate_coords[i, 0]],
                [demand_coords[j, 1], candidate_coords[i, 1]],
                color="green",
                alpha=0.5,
                linestyle="--",
            )

    ax.legend()
    ax.set_xlabel("X座標")
    ax.set_ylabel("Y座標")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


def set_page_config() -> None:
    st.set_page_config(
        page_title="コンビニ配置 最適化シミュレーター",
        page_icon="🏪",
        layout="wide",
    )


def get_common_parameters() -> Tuple[int, int, int, int]:
    st.header("データ生成パラメータ")
    seed = st.number_input("乱数シード", 0, value=42)
    n = st.slider("コンビニ候補地の数 (n)", 1, 50, 15)
    m = st.slider("住民グループの数 (m)", 1, 50, 30)
    area_size = st.slider("シミュレーションエリアの広さ", 50, 200, 100)
    return int(seed), int(n), int(m), int(area_size)


REGISTRY: Dict[str, Dict[str, Any]] = {
    "最大徒歩距離の最小化": {
        "description": DESC_CONBINI_P_CENTER,
        "func": optimize_p_center,
    },
    "総徒歩距離の最小化": {
        "description": DESC_CONBINI_P_MEDIAN,
        "func": optimize_p_median,
    },
}


def main() -> None:
    set_page_config()
    st.title("🏪 コンビニ配置 最適化シミュレーター")

    with st.sidebar:
        st.title("⚙️ 設定")
        seed, n, m, area_size = get_common_parameters()
        model_option = st.selectbox("最適化モデルを選択してください", list(REGISTRY.keys()))
        if not model_option:
            return

        model_params = get_model_parameters(n)

        desc = REGISTRY[model_option]["description"]
        opt_func = REGISTRY[model_option]["func"]

    col1, col2 = st.columns(2)

    with col1:
        st.header("📍 初期データ")
        if st.button("データを生成・表示"):
            coords, demands, pops, dists = generate_data(n, m, area_size, seed)
            st.session_state["data_conbini"] = (coords, demands, pops, dists)
            fig = visualize_initial_data(coords, demands, pops)
            st.session_state["initial_fig_conbini"] = fig

        if "initial_fig_conbini" in st.session_state:
            st.pyplot(st.session_state["initial_fig_conbini"])
        else:
            st.info("「データを生成・表示」ボタンを押してシミュレーションを開始してください。")

    with col2:
        st.header("📈 最適化結果")
        if "data_conbini" in st.session_state:
            if st.button("最適化を実行"):
                with st.expander("最適化問題の詳細", expanded=False):
                    st.markdown(desc, unsafe_allow_html=True)
                with st.spinner("最適化計算を実行中..."):
                    coords, demands, pops, dists = st.session_state["data_conbini"]
                    p_val = model_params["p"]
                    D_val = model_params["D"]

                    status, x, y, obj_val = opt_func(n, m, pops, dists, p_val, D_val)

                    if "Optimal" in status or "Feasible" in status:
                        if "p-センター" in model_option:
                            title = f"最適化結果: {model_option}\n(最大距離: {obj_val:.2f})"
                        else:
                            title = f"最適化結果: {model_option}\n(総距離: {obj_val:,.0f})"

                        fig = visualize_optimization_result(coords, demands, pops, x, y, title)
                        st.session_state["status_conbini"] = status
                        st.session_state["result_fig_conbini"] = fig
                    else:
                        st.error(f"最適化に失敗しました。ステータス: {status}")
                        if "result_fig_conbini" in st.session_state:
                            del st.session_state["result_fig_conbini"]

            if "result_fig_conbini" in st.session_state:
                st.success(f"最適化ステータス: **{st.session_state['status_conbini']}**")
                st.pyplot(st.session_state["result_fig_conbini"])
            else:
                st.info("「最適化を実行」ボタンを押して計算を開始してください。")
        else:
            st.warning("先に左側の「データを生成・表示」ボタンでデータを生成してください。")


if __name__ == "__main__":
    main()

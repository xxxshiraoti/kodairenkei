import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit as st
from typing import Any, Callable, Dict, Tuple


@st.cache_data
def generate_data(
    n: int, m: int, max_distance: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    人工データを生成する関数（収容人数を除く）。

    Args:
        n (int): 公園の候補地の数。
        m (int): 小学校の数。
        max_distance (int): 座標の最大値。
        seed (int): 乱数のシード。

    Returns:
        tuple[np.ndarray, ...]: 候補地の座標、小学校の座標、児童数、距離行列。
    """
    np.random.seed(seed)
    candidate_coords = np.random.rand(n, 2) * max_distance
    demand_coords = np.random.rand(m, 2) * max_distance

    mean_population = [20, 40]
    std_dev_population = [5, 10]
    weights_population = [0.5, 0.5]
    demand_populations_list = []
    for _ in range(m):
        component = np.random.choice(len(weights_population), p=weights_population)
        population = np.random.normal(mean_population[component], std_dev_population[component])
        demand_populations_list.append(max(5, int(population)))
    demand_populations = np.array(demand_populations_list)

    d = np.linalg.norm(
        candidate_coords[:, np.newaxis, :] - demand_coords[np.newaxis, :, :], axis=2
    )

    return candidate_coords, demand_coords, demand_populations, d


DESC_PARK_MAX_DIST = """
### 最大利用距離最小化問題 (p-センター問題)

#### 概要
全ての児童がどこかの公園を利用できる条件下で、移動距離の最大値を最小化します。公園には定員があり、設置するからには最低利用人数を満たす必要があります。

#### 定数
- $I$: 公園の候補地集合
- $J$: 小学校の集合
- $D$: 児童が到達可能な最大距離
- $p_j$: 小学校$j$の児童数
- $c_i^{\\text{up}}$: 公園$i$の最大受け入れ人数
- $c_i^{\\text{low}}$: 公園$i$の最小必要利用人数
- $d_{ij}$: 公園$i$と小学校$j$の距離

#### 変数
- $x_i$: 候補地$i$に公園を設置する場合 1, しない場合 0
- $y_{ij}$: 小学校$j$の児童が公園$i$を利用する割合 (0~1)
- $z_{ij}$: 小学校$j$が公園$i$を利用可能な場合 1, そうでない場合 0
- $T$: 最大利用距離

#### 定式化
**目的関数: 最大移動距離の最小化**
$$\\text{Minimize} \\quad T$$

**制約条件:**
1. **児童の割り当て**: 全ての児童がいずれか１つの公園を利用する
   $$\\sum_{i \\in I} y_{ij} = 1 \\quad (\\forall j \\in J)$$
2. **設置と利用**: 公園が設置されている場合のみ、その公園は利用可能
   $$y_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
3. **収容人数**: 公園の利用人数は、上限と下限の範囲内でなければならない
   $$\\sum_{j \\in J} p_j y_{ij} \\leq c_i^{\\text{up}} x_i \\quad (\\forall i \\in I)$$
   $$\\sum_{j \\in J} p_j y_{ij} \\geq c_i^{\\text{low}} x_i \\quad (\\forall i \\in I)$$
4. **利用可能性**: 児童が公園を利用する場合、その公園は利用可能でなければならない
   $$y_{ij} \\leq z_{ij}, \\quad z_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
5. **距離制約**: 利用可能な公園は、最大到達可能距離$D$以内でなければならない
   $$d_{ij} z_{ij} \\leq D \\quad (\\forall i \\in I, \\forall j \\in J)$$
6. **最大移動距離**: 全ての児童の移動距離は$T$以下でなければならない
   $$d_{ij} z_{ij} \\leq T \\quad (\\forall i \\in I, \\forall j \\in J)$$
"""

DESC_PARK_MAX_USERS = """
### 合計利用人数最大化問題 (最大被覆問題)

#### 概要
各公園の定員や最低利用人数を守りつつ、公園を利用する児童の総数を最大化します。全ての児童が公園を利用できるとは限りません。

#### 定数
- $I$: 公園の候補地集合
- $J$: 小学校の集合
- $D$: 児童が到達可能な最大距離
- $p_j$: 小学校$j$の児童数
- $c_i^{\\text{up}}$: 公園$i$の最大受け入れ人数
- $c_i^{\\text{low}}$: 公園$i$の最小必要利用人数
- $d_{ij}$: 公園$i$と小学校$j$の距離

#### 変数
- $x_i$: 候補地$i$に公園を設置する場合 1, しない場合 0
- $y_{ij}$: 小学校$j$の児童が公園$i$を利用する割合 (0~1)
- $z_{ij}$: 小学校$j$が公園$i$を利用可能な場合 1, そうでない場合 0

#### 定式化
**目的関数: 合計利用人数の最大化**
$$\\text{Maximize} \\quad \\sum_{i \\in I} \\sum_{j \\in J} p_j y_{ij}$$

**制約条件:**
1. **児童の割り当て**: 各小学校の児童は、最大でも1つの公園しか利用できない
   $$\\sum_{i \\in I} y_{ij} \\leq 1 \\quad (\\forall j \\in J)$$
2. **設置と利用**: 公園が設置されている場合のみ、その公園は利用可能
   $$y_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
3. **収容人数**: 公園の利用人数は、上限と下限の範囲内でなければならない
   $$\\sum_{j \\in J} p_j y_{ij} \\leq c_i^{\\text{up}} x_i \\quad (\\forall i \\in I)$$
   $$\\sum_{j \\in J} p_j y_{ij} \\geq c_i^{\\text{low}} x_i \\quad (\\forall i \\in I)$$
4. **利用可能性**: 児童が公園を利用する場合、その公園は利用可能でなければならない
   $$y_{ij} \\leq z_{ij}, \\quad z_{ij} \\leq x_i \\quad (\\forall i \\in I, \\forall j \\in J)$$
5. **距離制約**: 利用可能な公園は、最大到達可能距離$D$以内でなければならない
   $$d_{ij} z_{ij} \\leq D \\quad (\\forall i \\in I, \\forall j \\in J)$$
"""


def get_model_parameters() -> Dict[str, Any]:
    st.header("最適化モデルのパラメータ")
    params: Dict[str, Any] = {}
    params["D"] = st.number_input(
        "児童が到達可能な最大距離 (D)",
        min_value=1,
        max_value=1000,
        value=70,
        help="児童が公園まで歩いて行ける最大の距離。この距離を超える公園は利用できません。",
    )

    # st.session_stateから公園の数を取得して、動的に入力欄を生成
    n_parks = st.session_state.get("n_slider", 10)  # デフォルト値を設定

    c_up_list = []
    c_low_list = []

    with st.expander("公園ごとの収容人数を設定", expanded=True):
        for i in range(n_parks):
            col1, col2 = st.columns(2)
            with col1:
                c_up_val = st.number_input(
                    f"候補地 {i} の最大収容人数", min_value=1, value=150, key=f"cup_{i}"
                )
                c_up_list.append(c_up_val)
            with col2:
                c_low_val = st.number_input(
                    f"候補地 {i} の最低利用人数", min_value=0, value=10, key=f"clow_{i}"
                )
                c_low_list.append(c_low_val)

    params["c_up_list"] = c_up_list
    params["c_low_list"] = c_low_list

    return params


def optimize_max_dist(
    n: int, m: int, p: np.ndarray, c_up: np.ndarray, c_low: np.ndarray, d: np.ndarray, D: int
) -> Tuple[str, Dict[int, pulp.LpVariable], Dict[Tuple[int, int], pulp.LpVariable], float]:
    model = pulp.LpProblem("Minimize_Max_Park_Distance", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)
    T = pulp.LpVariable("T", lowBound=0, cat=pulp.LpContinuous)

    model += T

    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1
    for i in range(n):
        model += pulp.lpSum(p[j] * y[i][j] for j in range(m)) <= c_up[i]
        model += pulp.lpSum(p[j] * y[i][j] for j in range(m)) >= c_low[i] * x[i]
        for j in range(m):
            model += y[i][j] <= x[i]
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            model += d[i][j] * z[i][j] <= D
            model += d[i][j] * z[i][j] <= T

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_sol = {i: x[i] for i in range(n)}
    y_sol = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    if T.value() is not None:
        return status, x_sol, y_sol, T.value()
    return status, x_sol, y_sol, -1.0


def optimize_max_users(
    n: int, m: int, p: np.ndarray, c_up: np.ndarray, c_low: np.ndarray, d: np.ndarray, D: int
) -> Tuple[str, Dict[int, pulp.LpVariable], Dict[Tuple[int, int], pulp.LpVariable], float]:
    model = pulp.LpProblem("Maximize_Total_Park_Users", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts(
        "y", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous
    )
    z = pulp.LpVariable.dicts("z", (range(n), range(m)), cat=pulp.LpBinary)

    model += pulp.lpSum(p[j] * y[i][j] for i in range(n) for j in range(m))

    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) <= 1
    for i in range(n):
        model += pulp.lpSum(p[j] * y[i][j] for j in range(m)) <= c_up[i]
        model += pulp.lpSum(p[j] * y[i][j] for j in range(m)) >= c_low[i] * x[i]
        for j in range(m):
            model += y[i][j] <= x[i]
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            model += d[i][j] * z[i][j] <= D

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    x_sol = {i: x[i] for i in range(n)}
    y_sol = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    if model.objective is not None and model.objective.value() is not None:
        return status, x_sol, y_sol, model.objective.value()
    return status, x_sol, y_sol, -1.0


def visualize_initial_data(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_populations: np.ndarray,
    c_up: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        demand_coords[:, 0], demand_coords[:, 1], c="blue", s=300, label="小学校", alpha=0.7
    )
    ax.scatter(
        candidate_coords[:, 0],
        candidate_coords[:, 1],
        c="red",
        s=300,
        label="公園候補地",
        alpha=0.7,
        marker="s",
    )

    for i, (x, y) in enumerate(demand_coords):
        ax.text(x, y + 2, f"小学校{i}", fontsize=10, ha="center", va="bottom")
        ax.text(x, y - 2, f"児童数: {demand_populations[i]}", fontsize=10, ha="center", va="top")

    for i, (x, y) in enumerate(candidate_coords):
        ax.text(x, y + 2, f"候補地{i}", fontsize=10, ha="center", va="bottom")
        ax.text(x, y - 2, f"定員: {c_up[i]}", fontsize=10, ha="center", va="top")

    ax.legend(markerscale=0.7)
    ax.set_xlabel("X座標")
    ax.set_ylabel("Y座標")
    ax.set_title("初期データ：公園候補地と小学校の配置")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


def visualize_optimization_result(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_populations: np.ndarray,
    x: Dict[int, pulp.LpVariable],
    y: Dict[Tuple[int, int], pulp.LpVariable],
    c_up: np.ndarray,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10))

    # 凡例用のダミープロット
    ax.scatter([], [], c="red", s=400, label="設置された公園", marker="s", edgecolors="black")
    ax.scatter([], [], c="gray", s=300, label="設置されなかった候補地", alpha=0.4, marker="s")
    ax.scatter([], [], c="blue", s=300, label="小学校", alpha=0.7)

    for i, coord in enumerate(candidate_coords):
        if pulp.value(x[i]) > 0.5:
            ax.scatter(
                coord[0], coord[1], c="red", s=400, alpha=0.8, marker="s", edgecolors="black"
            )
            ax.text(
                coord[0],
                coord[1] + 2.5,
                f"候補地{i}",
                fontsize=10,
                ha="center",
                va="bottom",
                weight="bold",
            )
            ax.text(
                coord[0], coord[1] - 2.5, f"定員: {c_up[i]}", fontsize=10, ha="center", va="top"
            )
        else:
            ax.scatter(coord[0], coord[1], c="gray", s=300, alpha=0.4, marker="s")

    ax.scatter(demand_coords[:, 0], demand_coords[:, 1], c="blue", s=300, alpha=0.7)
    for j, coord in enumerate(demand_coords):
        ax.text(coord[0], coord[1] + 2, f"小学校{j}", fontsize=10, ha="center", va="bottom")
        ax.text(
            coord[0],
            coord[1] - 2,
            f"児童数: {demand_populations[j]}",
            fontsize=10,
            ha="center",
            va="top",
        )

    for i in range(len(candidate_coords)):
        for j in range(len(demand_coords)):
            if (
                y.get((i, j)) is not None
                and pulp.value(y[(i, j)]) is not None
                and pulp.value(y[(i, j)]) > 1e-3
            ):
                ax.plot(
                    [demand_coords[j, 0], candidate_coords[i, 0]],
                    [demand_coords[j, 1], candidate_coords[i, 1]],
                    color="green",
                    alpha=0.6,
                    linestyle="--",
                )
                assigned_pop = pulp.value(y[(i, j)]) * demand_populations[j]
                ax.text(
                    (demand_coords[j, 0] + candidate_coords[i, 0]) / 2,
                    (demand_coords[j, 1] + candidate_coords[i, 1]) / 2,
                    f"{assigned_pop:.1f}人",
                    fontsize=9,
                    color="darkgreen",
                    bbox=dict(
                        facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2"
                    ),
                )

    ax.legend()
    ax.set_xlabel("X座標")
    ax.set_ylabel("Y座標")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


def set_page_config() -> None:
    st.set_page_config(
        page_title="公園配置 最適化シミュレーター",
        page_icon="🏞️",
        layout="wide",
    )


def get_common_parameters() -> Tuple[int, int, int, int]:
    st.header("データ生成パラメータ")
    seed = st.number_input(
        "乱数シード", min_value=0, value=42, help="データ生成に使用する乱数のシード値です。"
    )
    # nのスライダーにキーを設定し、他の場所から値を取得できるようにする
    n = st.slider(
        "公園候補地の数 (n)", 1, 50, 10, key="n_slider", help="地図上に配置する公園候補地の数。"
    )
    m = st.slider("小学校の数 (m)", 1, 50, 15, help="地図上に配置する小学校の数。")
    # 名称を分かりやすく変更
    area_size = st.slider(
        "シミュレーションエリアの広さ",
        50,
        200,
        100,
        help="公園や小学校が配置される仮想的な正方形エリアの一辺の長さを設定します。",
    )
    return int(seed), int(n), int(m), int(area_size)


REGISTRY: Dict[str, Dict[str, Any]] = {
    "最大利用距離の最小化": {
        "description": DESC_PARK_MAX_DIST,
        "param_fn": get_model_parameters,
    },
    "合計利用人数の最大化": {
        "description": DESC_PARK_MAX_USERS,
        "param_fn": get_model_parameters,
    },
}


def main() -> None:
    set_page_config()
    st.title("🏞️ 公園配置 最適化シミュレーター")

    with st.sidebar:
        st.title("⚙️ 設定")
        model_option = st.selectbox("最適化モデルを選択してください", list(REGISTRY.keys()))
        if model_option is None:
            return

        desc: str = REGISTRY[model_option]["description"]
        param_fn: Callable[..., Dict[str, Any]] = REGISTRY[model_option]["param_fn"]

        # Get parameters for data generation
        seed, n, m, area_size = get_common_parameters()

        # Get parameters for the optimization model (this will now include individual capacities)
        model_params = param_fn()

    col1, col2 = st.columns(2)

    with col1:
        st.header("📍 初期データ")
        if st.button("データを生成・表示"):
            # Generate base data (coordinates, populations, distances)
            candidate_coords, demand_coords, p, d = generate_data(n, m, area_size, seed)

            # Get the user-defined capacity arrays from the sidebar
            c_up = np.array(model_params["c_up_list"])
            c_low = np.array(model_params["c_low_list"])

            # Store all data, including user-defined capacities, in the session state
            st.session_state["data_park"] = (candidate_coords, demand_coords, p, c_up, c_low, d)

            # Visualize the initial data with the specified capacities
            fig = visualize_initial_data(candidate_coords, demand_coords, p, c_up)
            st.session_state["initial_fig_park"] = fig

        if "initial_fig_park" in st.session_state:
            st.pyplot(st.session_state["initial_fig_park"])
        else:
            st.info("「データを生成・表示」ボタンを押して、シミュレーションを開始してください。")

    with col2:
        st.header("📈 最適化結果")
        if "data_park" in st.session_state:
            if st.button("最適化を実行"):
                with st.expander("最適化問題の詳細", expanded=False):
                    st.markdown(desc, unsafe_allow_html=True)
                with st.spinner("最適化計算を実行中..."):
                    # Retrieve all data from session state. c_up and c_low are now user-defined arrays.
                    candidate_coords, demand_coords, p, c_up_model, c_low_model, d = (
                        st.session_state["data_park"]
                    )

                    d_param = model_params["D"]

                    title = ""
                    status = "Not Solved"
                    if model_option == "最大利用距離の最小化":
                        status, x, y, max_dist = optimize_max_dist(
                            n, m, p, c_up_model, c_low_model, d, D=d_param
                        )
                        if "Optimal" in status or "Feasible" in status:
                            title = f"最適化結果: {model_option}\n(最大距離: {max_dist:.2f})"

                    elif model_option == "合計利用人数の最大化":
                        status, x, y, total_users = optimize_max_users(
                            n, m, p, c_up_model, c_low_model, d, D=d_param
                        )
                        if "Optimal" in status or "Feasible" in status:
                            total_pop = sum(p)
                            title = f"最適化結果: {model_option}\n(合計利用者数: {total_users:.0f}人 / 全体児童数: {total_pop}人)"

                    if "Optimal" in status or "Feasible" in status:
                        fig = visualize_optimization_result(
                            candidate_coords, demand_coords, p, x, y, c_up_model, title
                        )
                        st.session_state["status_park"] = status
                        st.session_state["result_fig_park"] = fig
                    else:
                        st.error(f"最適化に失敗しました。ステータス: {status}")
                        if "result_fig_park" in st.session_state:
                            del st.session_state["result_fig_park"]

            if "result_fig_park" in st.session_state:
                st.success(f"最適化ステータス: **{st.session_state['status_park']}**")
                st.pyplot(st.session_state["result_fig_park"])
            else:
                st.info("「最適化を実行」ボタンを押して、計算を開始してください。")
        else:
            st.warning("先に左側の「データを生成・表示」ボタンでデータを生成してください。")


if __name__ == "__main__":
    main()

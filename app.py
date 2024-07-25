import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit as st


@st.cache_data
def generate_data(
    n: int, m: int, max_capacity: int, max_distance: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    人工データを生成する関数。

    Args:
    n (int): 避難所の候補地の数。
    m (int): 避難者グループの数。
    max_capacity (int): 各避難所の収容人数上限。
    max_distance (int): 各避難者グループから避難所までの距離の最大値。
    D (int): 避難可能な最大距離。

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 避難所の座標、避難者グループの座標、避難者グループの人口、各避難所の収容人数上限、避難者グループから避難所までの距離行列。
    """
    np.random.seed(42)
    shelter_coords = np.random.rand(n, 2) * max_distance

    group_coords = np.random.rand(m, 2) * max_distance
    mean_population = [10, 20]
    std_dev_population = [5, 10]
    weights_population = [0.5, 0.5]

    group_populations = []
    for _ in range(m):
        component = np.random.choice(len(weights_population), p=weights_population)
        population = np.random.normal(mean_population[component], std_dev_population[component])
        group_populations.append(max(1, int(population)))

    group_populations = np.array(group_populations)
    c = np.random.randint(30, max_capacity + 1, size=n)
    d = np.linalg.norm(shelter_coords[:, np.newaxis, :] - group_coords[np.newaxis, :, :], axis=2)

    return shelter_coords, group_coords, group_populations, c, d


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


def optimize_shelter_installation(
    n: int, m: int, D: int, group_populations: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[dict[int, pulp.LpVariable], dict[tuple[int, int], pulp.LpVariable]]:
    """
    避難所の設置数を最小化するための最適化を行う関数。

    Args:
    n (int): 避難所の候補地の数。
    m (int): 避難者グループの数。
    D (int): 避難可能な最大距離。
    group_populations (np.ndarray): 避難者グループの人口。
    c (np.ndarray): 各避難所の収容人数上限。
    d (np.ndarray): 避難者グループから避難所までの距離行列。

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

    for j in range(m):
        model += pulp.lpSum(y[i][j] for i in range(n)) == 1

    for i in range(n):
        for j in range(m):
            model += y[i][j] <= x[i]
            model += y[i][j] <= z[i][j]
            model += z[i][j] <= x[i]
            if d[i][j] > D:
                model += z[i][j] == 0

    for i in range(n):
        model += pulp.lpSum(y[i][j] * group_populations[j] for j in range(m)) <= c[i]

    model.solve()

    # to dict from LpVariable
    x = {i: x[i] for i in range(n)}
    y = {(i, j): y[i][j] for i in range(n) for j in range(m)}
    return x, y


def visualize_population_data(
    shelter_coords: np.ndarray,
    group_coords: np.ndarray,
    group_populations: np.ndarray,
    c: np.ndarray,
) -> plt.Figure:
    """
    人口データを可視化する関数。

    Args:
    shelter_coords (np.ndarray): 避難所の座標。
    group_coords (np.ndarray): 避難者グループの座標。
    group_populations (np.ndarray): 避難者グループの人口。
    c (np.ndarray): 各避難所の収容人数上限。

    Returns:
    plt.Figure: 可視化された図。
    """
    ax: plt.Axes  # type: ignore
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        group_coords[:, 0],
        group_coords[:, 1],
        c="blue",
        label="避難者グループ",
        s=group_populations * 10,
        alpha=0.6,
    )
    ax.scatter(
        shelter_coords[:, 0],
        shelter_coords[:, 1],
        c="red",
        label="避難所の候補地",
        s=c * 10,
        alpha=0.6,
    )
    for i in range(len(group_coords)):
        ax.text(
            group_coords[i, 0],
            group_coords[i, 1],
            f"{group_populations[i]}",
            fontsize=9,
            ha="right",
        )
    for i in range(len(shelter_coords)):
        ax.text(shelter_coords[i, 0], shelter_coords[i, 1], f"{c[i]}", fontsize=9, ha="right")
    ax.legend(markerscale=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    return fig


def visualize_evacuation_plan(
    shelter_coords: np.ndarray,
    group_coords: np.ndarray,
    group_populations: np.ndarray,
    x: dict[int, pulp.LpVariable],
    y: dict[tuple[int, int], pulp.LpVariable],
    c: np.ndarray,
) -> plt.Figure:
    """
    最適化の結果を可視化する関数。

    Args:
    shelter_coords (np.ndarray): 避難所の座標。
    group_coords (np.ndarray): 避難者グループの座標。
    group_populations (np.ndarray): 避難者グループの人口。
    x (dict[int, pulp.LpVariable]): 避難所の設置決定変数。
    y (dict[tuple[int, int], pulp.LpVariable]): 避難者グループの割り当て変数。
    c (np.ndarray): 各避難所の収容人数上限。

    Returns:
    plt.Figure: 可視化された図。
    """
    ax: plt.Axes  # type: ignore
    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(len(shelter_coords)):
        if pulp.value(x[i]) > 0:
            ax.scatter(
                shelter_coords[i, 0],
                shelter_coords[i, 1],
                c="blue",
                s=200,
                label=f"設置した避難所 {i}" if i == 0 else "",
            )
            ax.text(shelter_coords[i, 0], shelter_coords[i, 1], f"{c[i]}", fontsize=9, ha="right")
        else:
            ax.scatter(
                shelter_coords[i, 0],
                shelter_coords[i, 1],
                c="black",
                s=200,
                label="未選択の避難所" if i == 0 else "",
            )

    for j in range(len(group_coords)):
        ax.scatter(
            group_coords[j, 0],
            group_coords[j, 1],
            c="red",
            s=group_populations[j] * 10,
            label="避難者グループ" if j == 0 else "",
        )
        ax.text(
            group_coords[j, 0],
            group_coords[j, 1],
            f"{group_populations[j]}",
            fontsize=9,
            ha="right",
        )
        for i in range(len(shelter_coords)):
            if pulp.value(y[(i, j)]) > 0:
                ax.arrow(
                    group_coords[j, 0],
                    group_coords[j, 1],
                    shelter_coords[i, 0] - group_coords[j, 0],
                    shelter_coords[i, 1] - group_coords[j, 1],
                    head_width=0,
                    head_length=0,
                    fc="green",
                    ec="green",
                    alpha=0.4,
                )
                ax.text(
                    (group_coords[j, 0] + shelter_coords[i, 0]) / 2,
                    (group_coords[j, 1] + shelter_coords[i, 1]) / 2,
                    f"{pulp.value(y[(i, j)]) * group_populations[j]:.2f}",
                    fontsize=9,
                    color="green",
                )

    n_shelters = int(sum([pulp.value(x[i]) for i in range(len(shelter_coords))]))
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"避難所配置問題の最適化結果 (避難所の設置数: {n_shelters})")
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


def get_parameters() -> tuple[int, int, int, int, int]:
    """
    パラメータを取得する関数。

    Returns:
    tuple[int, int, int, int, int]: 避難所の候補地の数、避難者グループの数、各避難所の収容人数上限、各避難者グループから避難所までの距離の最大値、避難可能な最大距離。
    """
    n = st.number_input("避難所の候補地の数 (n)", min_value=1, max_value=100, value=5)
    m = st.number_input("避難者グループの数 (m)", min_value=1, max_value=100, value=10)
    max_capacity = st.number_input(
        "各避難所の収容人数上限 (max_capacity)", min_value=1, max_value=1000, value=50
    )
    max_distance = st.number_input(
        "各避難者グループから避難所までの距離の最大値 (max_distance)",
        min_value=1,
        max_value=1000,
        value=100,
    )
    D = st.number_input("避難可能な最大距離 (D)", min_value=1, max_value=1000, value=50)

    # to int from Number
    n = int(n)
    m = int(m)
    max_capacity = int(max_capacity)
    max_distance = int(max_distance)
    D = int(D)

    return n, m, max_capacity, max_distance, D


def main() -> None:
    set_page_config()
    st.title("避難所配置問題の最適化")

    # パラメータ入力
    n, m, max_capacity, max_distance, D = get_parameters()

    if st.button("データ生成"):
        shelter_coords, group_coords, group_populations, c, d = generate_data(
            n, m, max_capacity, max_distance
        )
        fig1 = visualize_population_data(shelter_coords, group_coords, group_populations, c)
        st.session_state["data_fig"] = fig1
        st.session_state["data"] = (shelter_coords, group_coords, group_populations, c, d)

    if "data_fig" in st.session_state:
        st.pyplot(st.session_state["data_fig"])

    if "data" in st.session_state:
        shelter_coords, group_coords, group_populations, c, d = st.session_state["data"]

        st.write("最適化モデルの選択")
        model_option = st.selectbox(
            "最適化モデルを選択してください", ["避難所の設置数最小化", "避難時間最小化"]
        )
        with st.expander("最適化問題の詳細"):
            if model_option == "避難所の設置数最小化":
                st.markdown(DESC_OPTIMIZE_SHELTER_INSTALLATION)

        if st.button("最適化実行"):
            if model_option == "避難所の設置数最小化":
                x, y = optimize_shelter_installation(n, m, D, group_populations, c, d)
                st.write("避難所の設置数最小化の結果")
                fig2 = visualize_evacuation_plan(
                    shelter_coords, group_coords, group_populations, x, y, c
                )
                st.session_state["opt_fig"] = fig2

    if "opt_fig" in st.session_state:
        st.pyplot(st.session_state["opt_fig"])


if __name__ == "__main__":
    main()

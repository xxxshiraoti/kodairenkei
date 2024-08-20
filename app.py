import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit as st
from typing import Any
from typing import Callable
import pandas as pd


@st.cache_data
def generate_data(
    n: int, m: int, max_capacity: int, max_distance: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    人工データを生成する関数。

    Args:
    n (int): 避難所の候補地の数。
    m (int): 避難者グループの数。
    max_capacity (int): 各避難所の収容人数上限。
    max_distance (int): 各避難者グループから避難所までの距離の最大値。
    D (int): 避難可能な最大距離。
    seed (int): 乱数のシード。

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 避難所の座標、避難者グループの座標、避難者グループの人口、各避難所の収容人数上限、避難者グループから避難所までの距離行列。
    """
    np.random.seed(seed)
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


def get_shelter_installation_parameters() -> dict[str, Any]:
    D = st.number_input("避難可能な最大距離 (D)", min_value=1, max_value=1000, value=50)

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
    D = st.number_input("避難可能な最大距離 (D)", min_value=1, max_value=1000, value=50)

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
    D = st.number_input("避難可能な最大距離 (D)", min_value=1, max_value=1000, value=50)
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
    D = st.number_input("避難可能な最大距離 (D)", min_value=1, max_value=1000, value=50)

    return {"D": D}


def optimize_sakaue(
    n: int,
    m: int,
    group_populations: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    l: np.ndarray,
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
}


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
    l (pd.DataFrame): 避難所グループから避難所までの避難可能距離。

    Returns:
    plt.Figure: 可視化された図。
    """
    ax: plt.Axes  # type: ignore
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        group_coords[:, 0],
        group_coords[:, 1],
        c="blue",
        s=300,
        label="避難者グループ",
        alpha=0.6,
    )
    ax.scatter(
        shelter_coords[:, 0],
        shelter_coords[:, 1],
        c="red",
        s=300,
        label="避難所の候補地",
        alpha=0.6,
    )

    for i in range(len(group_coords)):
        # 避難者グループ名を表示 (中央に表示)
        ax.text(
            group_coords[i, 0],
            group_coords[i, 1] + 3,
            f"避難者グループ: {i}",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        # 避難者数を表示
        ax.text(
            group_coords[i, 0],
            group_coords[i, 1] + 3,
            f"人数: {group_populations[i]}",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="top",
        )

    for i in range(len(shelter_coords)):
        # 避難所名を表示 (中央に表示)
        ax.text(
            shelter_coords[i, 0],
            shelter_coords[i, 1] - 3,
            f"避難所: {i}",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        # 避難所の収容人数を表示
        ax.text(
            shelter_coords[i, 0],
            shelter_coords[i, 1] - 3,
            f"収容人数: {c[i]}",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="top",
        )

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
    title: str = "避難所配置問題の最適化結果",
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
    title (str): 図のタイトル。

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
                c="red",
                s=300,
                label="設置した避難所" if i == 0 else "",
                alpha=0.6,
            )
            # 避難所名を表示 (中央に表示)
            ax.text(
                shelter_coords[i, 0],
                shelter_coords[i, 1] - 3,
                f"避難所: {i}",
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            # 避難所の収容人数を表示
            ax.text(
                shelter_coords[i, 0],
                shelter_coords[i, 1] - 3,
                f"収容人数: {c[i]}",
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="top",
            )
        else:
            ax.scatter(
                shelter_coords[i, 0],
                shelter_coords[i, 1],
                c="black",
                s=300,
                alpha=0.6,
            )
            ax.text(
                shelter_coords[i, 0],
                shelter_coords[i, 1],
                f"避難所: {i} (未選択)",
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
            )

    for j in range(len(group_coords)):
        ax.scatter(
            group_coords[j, 0],
            group_coords[j, 1],
            c="blue",
            s=300,
            label="避難者グループ" if j == 0 else "",
            alpha=0.6,
        )
        # 避難者グループ名を表示 (中央に表示)
        ax.text(
            group_coords[j, 0],
            group_coords[j, 1] - 3,
            f"避難者グループ: {j}",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        # 避難者グループの人数を表示
        ax.text(
            group_coords[j, 0],
            group_coords[j, 1] - 3,
            f"人数: {group_populations[j]}",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="top",
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


def get_parameters() -> tuple[int, int, int, int, int]:
    """
    パラメータを取得する関数。

    Returns:
    tuple[int, int, int, int]: seed、避難所の候補地の数、避難者グループの数、各避難所の収容人数上限、各避難者グループから避難所までの距離の最大値。
    """
    seed = st.number_input("乱数シード (seed)", min_value=0, value=42)
    n = st.number_input("避難所の候補地の数 (n)", min_value=1, max_value=100, value=5)
    m = st.number_input("避難者グループの数 (m)", min_value=1, max_value=100, value=5)
    max_capacity = st.number_input(
        "各避難所の収容人数上限 (max_capacity)", min_value=1, max_value=1000, value=50
    )
    max_distance = st.number_input(
        "各避難者グループから避難所までの距離の最大値 (max_distance)",
        min_value=1,
        max_value=1000,
        value=100,
    )

    # to int from Number
    seed = int(seed)
    n = int(n)
    m = int(m)
    max_capacity = int(max_capacity)
    max_distance = int(max_distance)

    return seed, n, m, max_capacity, max_distance


def main() -> None:
    set_page_config()
    st.title("避難所配置の最適化")

    # パラメータ入力
    with st.sidebar:
        st.write("パラメータ設定")
        seed, n, m, max_capacity, max_distance = get_parameters()

        #  l_ijの設定
        st.write("避難グループから避難所までの避難可能距離")
        columns = [f"グループ{j}" for j in range(m)]
        index = [f"避難所{i}" for i in range(n)]
        l_df = pd.DataFrame(np.full((n, m), max_distance), columns=columns, index=index)
        l_df = st.data_editor(l_df, num_rows="fixed")

    if st.button("データ生成"):
        shelter_coords, group_coords, group_populations, c, d = generate_data(
            n, m, max_capacity, max_distance, seed
        )
        fig1 = visualize_population_data(shelter_coords, group_coords, group_populations, c)
        st.session_state["data_fig"] = fig1
        st.session_state["data"] = (shelter_coords, group_coords, group_populations, c, d)

    if "data_fig" in st.session_state:
        st.pyplot(st.session_state["data_fig"])

    with st.sidebar:
        st.write("最適化モデルの選択")
        model_option = st.selectbox("最適化モデルを選択してください", REGISTRY.keys())
        if model_option is not None:
            kwargs: dict[str, Any] = REGISTRY[model_option]["param_fn"]()  # type: ignore
            desc: str = REGISTRY[model_option]["description"]  # type: ignore

    if "data" in st.session_state:
        shelter_coords, group_coords, group_populations, c, d = st.session_state["data"]

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

            fig2 = visualize_evacuation_plan(
                shelter_coords, group_coords, group_populations, x, y, c, title=title
            )
            st.session_state["status"] = status
            st.session_state["opt_fig"] = fig2

    if "status" in st.session_state:
        st.write(f"最適化ステータス: {st.session_state['status']}")
    if "opt_fig" in st.session_state:
        st.pyplot(st.session_state["opt_fig"])


if __name__ == "__main__":
    main()

import pytest
import pandas as pd
import numpy as np
import polars as pl
from degree_functions import (
    create_degree_list_python,
    create_degree_list_pandas,
    create_degree_list_polars,
    create_degree_list_numpy,
    create_degree_listnetx
)

def generate_edges(num_nodes, num_edges, seed=42):
    import random
    random.seed(seed)
    edges = set()
    while len(edges) < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            edge = tuple(sorted((u, v)))
            edges.add(edge)
    return list(edges)

def to_pandas_df(degree_list):
    df = pd.DataFrame(degree_list, columns=["node", "degree"])
    df = df.sort_values("node").reset_index(drop=True)
    return df

def to_pandas_df_from_polars(polars_df):
    df = polars_df.to_pandas()
    df = df.sort_values("nodes").reset_index(drop=True)
    df.columns = ["node", "degree"]
    return df

def to_pandas_df_from_numpy(np_arr):
    df = pd.DataFrame(np_arr, columns=["node", "degree"])
    df = df.sort_values("node").reset_index(drop=True)
    return df

@pytest.mark.parametrize("num_nodes,num_edges", [(10, 14), (100, 200)])
def test_degree_functions_equivalence(num_nodes, num_edges):
    edges = generate_edges(num_nodes, num_edges)
    # Python
    py_deg = to_pandas_df(create_degree_list_python(edges, num_nodes))
    # Pandas
    all_nodes = pd.DataFrame(edges, columns=["source", "target"]).melt(value_name="node")["node"]
    pd_deg = to_pandas_df(create_degree_list_pandas(all_nodes))
    # Polars
    df = pl.DataFrame({"source": [e[0] for e in edges], "target": [e[1] for e in edges]})
    all_nodes_df = pl.DataFrame({"nodes": pl.concat([df["source"], df["target"]])})
    pl_deg = to_pandas_df_from_polars(create_degree_list_polars(all_nodes_df))
    # NumPy
    edges_array = np.array(edges)
    np_deg = to_pandas_df_from_numpy(create_degree_list_numpy(edges_array))
    # NetX
    nx_deg = to_pandas_df(create_degree_listnetx(edges))
    # Compare all
    pd.testing.assert_frame_equal(py_deg, pd_deg)
    pd.testing.assert_frame_equal(py_deg, pl_deg)
    pd.testing.assert_frame_equal(py_deg, np_deg)
    pd.testing.assert_frame_equal(py_deg, nx_deg)

@pytest.mark.benchmark(group="degree_distribution")
@pytest.mark.parametrize("num_nodes,num_edges", [(100, 200), (1000, 2000)])
def test_benchmark_degree_functions(benchmark, num_nodes, num_edges):
    edges = generate_edges(num_nodes, num_edges)
    all_nodes = pd.DataFrame(edges, columns=["source", "target"]).melt(value_name="node")["node"]
    df = pl.DataFrame({"source": [e[0] for e in edges], "target": [e[1] for e in edges]})
    all_nodes_df = pl.DataFrame({"nodes": pl.concat([df["source"], df["target"]])})
    edges_array = np.array(edges)
    # Python
    benchmark(create_degree_list_python, edges, num_nodes)
    # Pandas
    benchmark(lambda: create_degree_list_pandas(all_nodes))
    # Polars
    benchmark(lambda: create_degree_list_polars(all_nodes_df))
    # NumPy
    benchmark(lambda: create_degree_list_numpy(edges_array))
    # NetX
    benchmark(lambda: create_degree_listnetx(edges))

@pytest.mark.parametrize("num_nodes,num_edges", [(10, 14)])
def test_compare_with_notebook_csv(num_nodes, num_edges):
    edges = generate_edges(num_nodes, num_edges)
    py_deg = to_pandas_df(create_degree_list_python(edges, num_nodes))
    # Load DataFrame generated from notebook
    notebook_df = pd.read_csv("notebook_degree_counts.csv").sort_values("node").reset_index(drop=True)
    pd.testing.assert_frame_equal(py_deg, notebook_df)

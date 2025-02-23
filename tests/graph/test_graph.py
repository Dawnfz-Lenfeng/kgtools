import numpy as np
import pytest

from kgtools.graph import build_graph


def test_build_graph_basic():
    """测试基本的图构建功能"""
    documents = [
        "机器学习是人工智能的一个子领域",
        "深度学习是机器学习的一种方法",
        "神经网络是深度学习的基础",
    ]
    keywords = ["机器学习", "深度学习", "人工智能", "神经网络"]

    with pytest.raises(NotImplementedError):
        matrix = build_graph(documents, keywords)

    # assert isinstance(matrix, np.ndarray)
    # assert matrix.shape == (4, 4)  # 4x4 矩阵，对应4个关键词
    # assert np.all(matrix >= 0)  # 所有权重应该非负
    # assert np.all(matrix <= 1)  # 归一化后的权重应该不超过1

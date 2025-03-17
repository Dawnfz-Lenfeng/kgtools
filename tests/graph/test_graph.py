from scipy import sparse

from kgtools.graph import build_graph


def test_build_graph():
    """测试基本的图构建功能"""
    docs = [
        "机器学习是人工智能的一个重要分支。\n深度学习是机器学习的一个子领域。\n神经网络是深度学习的基础。",
        "深度学习是机器学习的一个子领域。\n神经网络是深度学习的基础。\n自然语言处理是人工智能的一个重要分支。",
        "神经网络是深度学习的基础。\n自然语言处理是人工智能的一个重要分支。\n计算机视觉是人工智能的一个重要分支。",
    ]
    keywords = ["机器学习", "深度学习", "人工智能", "神经网络"]

    matrix = build_graph(docs, keywords, embedding_size=10, context_size=5)

    assert isinstance(matrix, sparse.csr_matrix)
    assert matrix.shape == (len(keywords), len(keywords))
    assert matrix.diagonal().sum() == 0  # 确保对角线为0
    assert 0 <= matrix.max() <= 1  # 确保归一化正确
    assert matrix.min() >= 0  # 确保没有负值

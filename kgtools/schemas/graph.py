from pydantic import BaseModel, Field


class GraphConfig(BaseModel):
    """图构建配置"""

    embedding_size: int = Field(
        default=200,
        ge=50,
        le=500,
        description="词向量维度",
    )
    context_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="共现窗口大小",
    )
    glove_epochs: int = Field(
        default=100,
        ge=1,
        description="GloVe 训练轮次",
    )
    glasso_epochs: int = Field(
        default=1000,
        ge=1,
        description="Graphical Lasso 最大迭代次数",
    )
    tolerance: float = Field(
        default=1e-3,
        gt=0,
        description="Graphical Lasso 收敛容差",
    )
    learning_rate: float = Field(
        default=0.05,
        gt=0,
        le=1.0,
        description="GloVe 学习率",
    )
    lambda_glasso: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Graphical Lasso 正则化参数",
    )
    min_weight: float = Field(
        default=0.1,
        ge=0,
        lt=1,
        description="最小关系权重阈值",
    )

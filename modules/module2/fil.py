import numpy as np
import dgl.function as fn
from dgl.utils import check_eq_shape, expand_as_pair
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FILConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm=None,
        activation=None,
    ):
        super(FILConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.norm = norm
        self.activation = activation
        #matrix P
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        #matrix Q
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)


        self.reset_parameters()
    def reset_parameters(self):
        """
        Initialize linear weights
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
       

    def forward(self, graph, feat, edge_weight=None):
        """
        graph : DGLGraph
        feat : Node characteristics
        edge_weight : Pearson coefficient between nodes
        ### 输入参数 
        1. graph (DGLGraph)
        - 类型 : DGL图对象
        - 功能 : 包含图的拓扑结构信息
        - 属性 :
        - graph.edges() : 边的连接关系
        - graph.num_nodes() : 节点数量
        - graph.num_edges() : 边数量
        - 示例 : 100个节点，500条边的脑网络图 

        2. feat (torch.Tensor)
        - 类型 : 节点特征张量
        - 形状 : [num_nodes, input_dim]
        - 功能 : 每个节点的初始特征向量
        - 示例值 : [100, 1] - 100个节点，每个节点1维特征
        - 具体数据 : 可能是脑区的激活强度、体积等 

        3. edge_weight (torch.Tensor, 可选)
        - 类型 : 边权重张量
        - 形状 : [num_edges]
        - 功能 : 边的权重信息
        - 示例值 : Pearson相关系数，范围[-1, 1]
        - 具体数据 : 脑区间的功能连接强度
        """
        with graph.local_scope():
            """
            - feat : 节点特征，可能的类型：
            - tuple : 包含两个元素的元组 (feat_src, feat_dst)
            - torch.Tensor : 单个特征张量
            - graph : DGL图对象，包含图结构信息
            
            但是实际代码使用的是第二种情况
            
            """
            if isinstance(feat, tuple):
                feat_src = feat[0]
                feat_dst = feat[1]
            else:
               # 当前代码的实际使用 ：使用的是同构图（homogeneous graph），传入单一张量
                feat_src = feat_dst = feat # 源节点和目标节点使用相同特征

                if graph.is_block: #实际上没用块图
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                # 功能 ：将边权重存储到图的边数据中
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")
            h_self = feat_dst
            # The case of boundless graphs
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)
            # (PX)W
            graph.srcdata["h"] = self.fc_neigh(feat_src)
            graph.update_all(msg_fn, fn.mean("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]
            FIL_out = self.fc_self(h_self) + h_neigh
            if self.activation is not None:
                FIL_out = self.activation(FIL_out)
            if self.norm is not None:
                FIL_out = self.norm(FIL_out)
            return FIL_out

class FILmodule(nn.Module):
    def __init__(self,
                 in_dim,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation):
        super(FILmodule, self).__init__()
        """
        参数1: in_dim=1
        - 含义 : 输入特征维度
        - 值 : 1
        - 形状 : 标量整数
        - 作用 : 定义每个节点的初始特征维度
        - 示例 : 如果图中每个节点只有1个特征值（如节点权重），则设为1
        - 数据流 : [num_nodes, 1] → FIL处理 → [num_nodes, n_hidden] 
        参数2: n_hidden=32
        - 含义 : FIL模块的隐藏层特征维度
        - 值 : 32
        - 形状 : 标量整数
        - 作用 : 控制FIL内部特征表示的维度大小
        - 示例 : 节点特征从1维扩展到32维，增强表示能力
        - 影响 : 更大的值提供更强的表示能力，但增加计算开销 
        参数3: out_dim=args.hidden_dim
        - 含义 : FIL模块的输出特征维度
        - 值 : 32 (根据Parser类定义， args.hidden_dim 默认为32)
        - 形状 : 标量整数
        - 作用 : 定义FIL处理后的最终特征维度
        - 示例 : 输出形状为 [num_nodes, 32]
        - 重要性 : 必须与后续GIN层的输入维度匹配 
        参数4: n_layers=2
        - 含义 : FIL模块内部的层数
        - 值 : 2
        - 形状 : 标量整数
        - 作用 : 控制FIL的深度和复杂度
        最终层数为n_layers+1

        ### 📊 数据流示例
        假设输入图有100个节点：

        ```
        # 输入数据
        input_features: [100, 1]     # 100个节点，每个1维特征
        edge_index: [2, E]           # E条边的连接信息

        # FIL处理流程
        Layer 0: [100, 1] → FILConv → [100, 32] → ReLU → [100, 32]
        Layer 1: [100, 32] → FILConv → [100, 32] → ReLU → [100, 32]  
        Layer 2: [100, 32] → FILConv → [100, 32] (无激活函数)

        # 最终输出
        output_features: [100, 32]   # 100个节点，每个32维特征
        ```
        """
        self.layers = nn.ModuleList()
        self.activation = activation
        self.layers.append(FILConv(in_dim, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(FILConv(n_hidden, n_hidden))
        self.layers.append(FILConv(n_hidden, out_dim))

    def forward(self, graph, inputs, edge_features):
        """
        1. graph (DGL图对象)

        - 类型 : dgl.DGLGraph
        - 功能 : 包含图的拓扑结构信息
        - 属性 :
        - graph.edges() : 边的连接关系
        - graph.num_nodes() : 节点数量
        - graph.num_edges() : 边数量
        - 示例 : 100个节点，500条边的脑网络图
        2. inputs (节点特征)

        - 类型 : torch.Tensor
        - 形状 : [num_nodes, input_dim]
        - 功能 : 每个节点的初始特征向量
        - 示例值 : [100, 1] - 100个节点，每个节点1维特征
        - 具体数据 : 可能是脑区的激活强度、体积等
        3. edge_features (边特征)

        - 类型 : torch.Tensor
        - 形状 : [num_edges]
        - 功能 : 边的权重信息
        - 示例值 : Pearson相关系数，范围[-1, 1]
        - 具体数据 : 脑区间的功能连接强度
        """
        h = inputs
        for l, layer in enumerate(self.layers):
            # - 形状 ： [num_nodes, output_dim]
            # - 功能 ：经过FIL处理后的增强节点特征
            h = layer(graph, h, edge_features)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h

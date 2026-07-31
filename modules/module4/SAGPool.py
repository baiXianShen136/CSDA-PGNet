from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch


class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=1.0,Conv=GCNConv,non_linearity=torch.tanh):

        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        x [N, in_channels] 节点特征矩阵，N是节点数
        edge_index [2, E] 边的索引，E是边数
        edge_attr [E, in_channels] 边的特征
        batch [N] 每个节点所属的图的索引
        
        """
        # 如果没有 batch 信息，假设所有节点属于同一个图
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
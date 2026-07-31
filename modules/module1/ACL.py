from modules.module1.functional import pr_drop_weights_new, normalize_edge_weights
from modules.module1.PAE import PAE
import torch.nn as nn
import torch

class AdaptiveConfidenceLearning(nn.Module):
    def __init__(self, args):
        super(AdaptiveConfidenceLearning, self).__init__()
        self.args = args
        self.device = args.device

        # Node Similarity Network (基于PAE的边网络)
         # 原来论文用表型的个数，我们也用 原来的dropout就是0.3

        # self.ph_edge_net = PAE(input_dim=len(args.scores), dropout=args.dropout)
        self.ph_edge_net = PAE(input_dim=3, dropout=args.dropout)

        
    

    def forward(self, ph_features, edge_index):
        # print('img_features:'+ str(img_features))
        # print('img_features.shape:'+ str(img_features.shape))
        # print('ph_features:'+ str(ph_features))
        # print('ph_features.shape:'+ str(ph_features.shape))
        # print('edge_index:'+ str(edge_index))
        # print('edge_index.shape:'+ str(edge_index.shape))

        # 1. 计算PageRank权重 (全局重要性)，这个就不用了，因为完全图他会出现buy，导致pv全是nan
        # pr_weights = pr_drop_weights_new(edge_index, aggr='sink', k=10).to(self.device)
        # print('pr_weights:'+ str(pr_weights))
        # print('pr_weights.shape:'+ str(pr_weights.shape))


        row, col = edge_index

         # 2. 构建边特征
    
        ph_edge_features = torch.cat([ph_features[row], ph_features[col]], dim=1)
        
        
        # 3. 分别计算相似度
        ph_similarity = self.ph_edge_net(ph_edge_features)

        
        # print('ph_similarity:'+ str(ph_similarity))
        # print('ph_similarity.shape:'+ str(ph_similarity.shape))

     

        # 5. 归一化以及最终融合 融合PageRank和相似度生成边重要性矩阵Q
        edge_similarity_norm = normalize_edge_weights(ph_similarity)
        # alpha = 0.5
        # Q_weights = alpha * edge_similarity_norm + (1 - alpha) * pr_weights
        Q_weights = edge_similarity_norm
        # print('pr_weights:'+str(pr_weights))
        # print('pr_weights.shape:'+ str(pr_weights.shape))
        # print('edge_similarity_norm:'+ str(edge_similarity_norm))
        # print('edge_similarity_norm.shape:'+ str(edge_similarity_norm.shape))
        # print('Q_weights:'+ str(Q_weights))
        # print('Q_weights.shape:'+ str(Q_weights.shape))

        
        return Q_weights
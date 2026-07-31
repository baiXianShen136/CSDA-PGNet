import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFineGrainedAttention(nn.Module):
    """
    跨模态细粒度注意力层 - 基于MMGL的MARL模块
    在 CSDA-PGNet 的 MRRL 模块中插入，实现模态间的细粒度依赖关系建模
    
    功能：
    1. 捕捉影像特征与非影像特征间的细粒度依赖关系
    2. 通过跨模态注意力矩阵建模"模态i对模态j的依赖度"
    3. 在VAE和RFE维度对齐后、Concat拼接前插入
    """
    
    def __init__(self, feature_dim, temperature=None):
        """
        初始化跨模态细粒度注意力层
        
        Args:
            feature_dim: 特征维度 (N×d中的d)
            temperature: 注意力温度参数，默认为sqrt(feature_dim)
        """
        super(CrossModalFineGrainedAttention, self).__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature or (feature_dim ** 0.5)
        
        # 步骤1：为每个模态创建Q、K、V的1×1卷积层（参数轻量）
        # 影像模态的Q、K、V变换
        self.img_query_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.img_key_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.img_value_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False)
        
        # 非影像模态的Q、K、V变换
        self.non_query_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.non_key_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.non_value_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for conv in [self.img_query_conv, self.img_key_conv, self.img_value_conv,
                     self.non_query_conv, self.non_key_conv, self.non_value_conv]:
            nn.init.xavier_uniform_(conv.weight)
    
    def forward(self, img_features, non_features):
        """
        前向传播
        
        Args:
            img_features: 影像特征 [N, d] - VAE升维后的非影像特征
            non_features: 非影像特征 [N, d] - RFE降维后的影像特征
            
        Returns:
            img_features_enhanced: 增强后的影像特征 [N, d]
            non_features_enhanced: 增强后的非影像特征 [N, d]
        """
        N, d = img_features.shape
        
        # 步骤1：生成Q、K、V
        # 将2D特征转换为3D以适配Conv1d: [N, d] -> [N, d, 1]
        img_features_3d = img_features.unsqueeze(-1)  # [N, d, 1]
        non_features_3d = non_features.unsqueeze(-1)  # [N, d, 1]
        
        # 计算Q、K、V
        # Q_img = W_q · X_img, K_non = W_k · X_non, V_non = W_v · X_non
        Q_img = self.img_query_conv(img_features_3d).squeeze(-1)  # [N, d]
        K_non = self.non_key_conv(non_features_3d).squeeze(-1)    # [N, d]
        V_non = self.non_value_conv(non_features_3d).squeeze(-1)  # [N, d]
        
        # Q_non = W_q · X_non, K_img = W_k · X_img, V_img = W_v · X_img
        Q_non = self.non_query_conv(non_features_3d).squeeze(-1)  # [N, d]
        K_img = self.img_key_conv(img_features_3d).squeeze(-1)     # [N, d]
        V_img = self.img_value_conv(img_features_3d).squeeze(-1)   # [N, d]
        
        # 步骤2：计算跨模态注意力权重
        # P_img→non = softmax(Q_img^T K_non / sqrt(d))
        # P_non→img = softmax(Q_non^T K_img / sqrt(d))
        
        # 计算注意力分数
        img_to_non_scores = torch.matmul(Q_img, K_non.T) / self.temperature  # [N, N]
        non_to_img_scores = torch.matmul(Q_non, K_img.T) / self.temperature  # [N, N]
        
        # 应用softmax得到注意力权重矩阵
        P_img_to_non = F.softmax(img_to_non_scores, dim=-1)  # [N, N]
        P_non_to_img = F.softmax(non_to_img_scores, dim=-1)  # [N, N]
        
        # 步骤3：用注意力权重对特征加权
        # X_img_new = X_img ⊙ P_img→non
        # X_non_new = X_non ⊙ P_non→img
        
        # 计算注意力加权的特征
        # 对于每个样本，使用其对应的注意力权重
        img_features_weighted = torch.zeros_like(img_features)
        non_features_weighted = torch.zeros_like(non_features)
        
        for i in range(N):
            # 使用第i个样本的注意力权重对所有样本的特征进行加权
            img_features_weighted[i] = torch.sum(
                img_features * P_img_to_non[i].unsqueeze(1), dim=0
            )
            non_features_weighted[i] = torch.sum(
                non_features * P_non_to_img[i].unsqueeze(1), dim=0
            )
        
        # 步骤4：保留原有特征分布，仅增强关联
        # 使用残差连接保持原有信息
        alpha = 0.15  # 注意力增强的权重系数 之前用的0.2 效果很好 0.15
        img_features_enhanced = img_features + alpha * img_features_weighted
        non_features_enhanced = non_features + alpha * non_features_weighted
        
        return img_features_enhanced, non_features_enhanced

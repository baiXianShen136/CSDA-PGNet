import numpy as np
import torch
from torch_geometric.utils import dropout_edge, dense_to_sparse,to_dense_adj
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from utils.tools import cal_feature_sim, EarlyStopping
from model.gtunet import GTUNet
import os
from modules.module1.ACL import AdaptiveConfidenceLearning
from modules.module2.fil import FILmodule
from modules.module3.CrossModalFineGrainedAttention import CrossModalFineGrainedAttention
from modules.module4.SAGPool import SAGPool
from torch_geometric.nn import GCNConv
import dgl

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.input_dim = len(args.scores)
        self.hidden_dim = 256
        self.latent_dim = args.node_dim # 默认500

        """
        ## 输入
        - 数据类型 ： torch.Tensor
        - 形状 ： [batch_size, input_dim]
        - 内容 ：表型特征数据（如年龄、性别、扫描站点等）
        - 数值范围 ：经过预处理的浮点数值
        - 具体示例 ：如果 input_dim = 3 （对应 ['SITE_ID', 'SEX', 'AGE_AT_SCAN'] ），batch_size = 32，
        则输入形状为 [32, 3]
        ## 输出
        - 数据类型 ： torch.Tensor
        - 形状 ： [batch_size, latent_dim * 2]
        - 内容 ：潜在分布的均值和对数方差的拼接
        - 结构组成 ：
        - 前半部分（ [:, :latent_dim] ）：潜在分布的均值 mu
        - 后半部分（ [:, latent_dim:] ）：潜在分布的对数方差 logvar
        - 具体示例 ：如果 latent_dim = 500 ，则输出形状为 [32, 1000] 
        ，其中前 500 维是均值，后 500 维是对数方差

        ### 各层功能：
        1. 第一层线性变换 ： nn.Linear(self.input_dim, self.hidden_dim)
        
        - 输入维度 ： input_dim （表型特征数量，通常为 3）
        - 输出维度 ： hidden_dim （隐藏层维度，固定为 256）
        - 功能 ：将低维表型特征映射到高维隐藏空间，增强表达能力
        2. 激活函数 ： nn.ReLU()
        
        - 功能 ：引入非线性，使网络能够学习复杂的非线性映射关系
        - 数学表达 ： f(x) = max(0, x)
        3. 第二层线性变换 ： nn.Linear(self.hidden_dim, self.latent_dim * 2)
        
        - 输入维度 ： hidden_dim （256）
        - 输出维度 ： latent_dim * 2 （通常为 500 × 2 = 1000）
        - 功能 ：将隐藏表示映射到潜在空间的统计参数

        ReLU（Rectified Linear Unit） 是一种非线性激活函数，其数学定义为：
        - f(x) = max(0, x)
        - 即：当输入 x > 0 时，输出 x ；当输入 x ≤ 0 时，输出 0
        # 第一个线性层输出（可能包含负值）
        hidden_output = [-0.5, 2.3, -1.2, 0.8, ...]  # shape: [batch_size, 256]

        # ReLU 激活后
        activated_output = [0.0, 2.3, 0.0, 0.8, ...]   # shape: [batch_size, 256]
        """
        self.encoder = nn.Sequential(
            # 第一层：输入层到隐藏层
            nn.Linear(self.input_dim, self.hidden_dim),
            # 激活函数
            nn.ReLU(),
            # 第二层：隐藏层到输出层e
            nn.Linear(self.hidden_dim, self.latent_dim * 2)
        )
        """
        - nn.Linear(self.latent_dim, self.hidden_dim) ：
            - 第一个全连接层，将潜在表示 z 从 latent_dim （500）维度线性变换到 hidden_dim （256）维度
            - 输入形状： [batch_size, 500]
            - 输出形状： [batch_size, 256]

        - nn.ReLU() ：
            - ReLU 激活函数，引入非线性，使网络能够学习复杂的映射关系
            - 数学表达： f(x) = max(0, x)
            - 形状保持不变： [batch_size, 256]

        - nn.Linear(self.hidden_dim, self.input_dim) ：
            - 第二个全连接层，将特征从 hidden_dim （256）维度线性变换到 input_dim （通常为3，对应表型特征数量）维度
            - 输入形状： [batch_size, 256]
            - 输出形状： [batch_size, 3]

        - nn.Sigmoid() ：
            - Sigmoid 激活函数，将输出值压缩到 [0, 1] 范围
            - 数学表达： f(x) = 1 / (1 + e^(-x))
            - 这适用于处理像表型特征这样可能归一化到此范围的数据
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            # used to process input data in the range [0, 1]
            nn.Sigmoid()
        )
        # the reconstruction loss function uses mean square error
        self.criterion = nn.MSELoss(reduction='sum')

    def reparameterize(self, mu, logvar):
        """
        ## 输入参数
        - mu ：潜在分布的均值向量
        
        - 数据类型： torch.Tensor
        - 形状： [batch_size, latent_dim]
        - 来源：编码器输出 mu_logvar 的前半部分
        - 含义：表示潜在空间中每个维度的均值
        - logvar ：潜在分布的对数方差向量
        
        - 数据类型： torch.Tensor
        - 形状： [batch_size, latent_dim]
        - 来源：编码器输出 mu_logvar 的后半部分
        - 含义：表示潜在空间中每个维度的对数方差（log variance）
        ## 输出
        - z ：从潜在分布中采样得到的潜在向量
        - 数据类型： torch.Tensor
        - 形状： [batch_size, latent_dim]
        - 含义：表示输入数据在潜在空间中的表示
        
        """
        std = torch.exp(0.5 * logvar)# 步骤1：计算标准差
        eps = torch.randn_like(std)# 步骤2：生成标准正态分布的随机噪声
        return mu + eps * std # 步骤3：重参数化采样

    def forward(self, x):
        """
        ## 输出
        - 返回值 ： mu_logvar - 编码器输出的组合向量
        - 数据类型 ： torch.Tensor
        - 形状 ： [batch_size, latent_dim * 2]
        - latent_dim ：潜在空间维度，等于 args.node_dim （默认500）
        - 输出维度是潜在维度的2倍，因为包含均值和对数方差
        - 数据内容 ：前半部分是均值向量，后半部分是对数方差向量
        
        """
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        """
        这行代码的作用是通过 VAE 模型的解码器（ self.decoder ）对潜在表示 z 进行解码，
        将其从低维潜在空间转换回原始数据空间，生成重构的表型特征 recon_x 。
        ## 输出
        - recon_x ：这是一个张量，包含了解码器重构出的表型特征
        - 形状 ： [batch_size, input_dim] ，其中 input_dim 是在 VAE 类的 __init__ 方法中通过 len(args.scores) 定义的，表示原始表型特征的数量
        - 数值范围 ： recon_x 的值在 [0, 1] 之间，因为解码器的最后一层使用了 nn.Sigmoid() 激活函数
        """
        recon_x = self.decoder(z)
        # calculate the reconstruction loss
        """
        重构损失计算 ：这行代码计算原始输入数据 x 与解码器重构输出 recon_x 之间的差异，衡量 VAE 模型重构原始数据的能力。
        重构损失是 VAE 总损失函数的重要组成部分，用于确保模型能够准确地从潜在表示中恢复原始数据。
        ## 输出
        - 返回值 ： reconstruction_loss
        - 数据类型 ： torch.Tensor （标量）
        - 数值含义 ：所有样本和所有特征维度上的重构误差总和
        - 计算公式 ：
        ```
        reconstruction_loss = Σ(recon_x - x)²
        ``` 其中求和遍历 batch 中的所有样本和所有特征维度
        由于 reduction='sum' ，所有误差平方被求和而不是平均
        """
        reconstruction_loss = self.criterion(recon_x, x)
        # calculating KL divergence
        """
        ## 输出结果
        - 返回值 : torch.Tensor (标量)
        - 数据类型 : float32
        - 含义 : 所有样本和所有潜在维度的KL散度总和
        - 用途 : 与重构损失相加构成VAE的总损失
        """
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = reconstruction_loss + kl_divergence
        return z, total_loss


class CustomConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.weight.data = torch.abs(self.weight.data)
        self.weight.data[:, 1, :, :] = - self.weight.data[:, 1, :, :]

    def forward(self, x):
        # Execute custom operation during forward pass
        self.weight.data[:, 1, :, :] = -torch.max(torch.abs(self.weight.data[:, 1, :, :]),
                                                  self.weight.data[:, 1, :, :] + self.weight.data[:, 2, :, :])
        self.weight.data[:, 0, :, :] = torch.max(self.weight.data[:, 0, :, :], self.weight.data[:, 2, :, :])
        return super(CustomConv2d, self).forward(x)


class RP_Attention(nn.Module):
    def __init__(self, args):
        super(RP_Attention, self).__init__()
        self.args = args
        self.num_score = len(args.scores)
        self.device = args.device

        # weight parameters for phenotypic data
        self.weights = nn.Parameter(torch.rand(self.num_score)).to(self.device)
        self.weights.data /= self.weights.data.sum()

        # reward, penalty and incentive coefficients
        self.conv = CustomConv2d(in_channels=3, out_channels=1)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_score):
            self.convs.append(self.conv)

    def forward(self, affinity_graphs):
        outputs = []
        value = []

        for i in range(self.num_score):
            outputs.append(F.sigmoid(self.convs[i](affinity_graphs[i]).squeeze()))
            reward = self.convs[i].weight.data[:, 0, :, :].squeeze()
            penalty = self.convs[i].weight.data[:, 1, :, :].squeeze()
            value.append(self.cal_value(affinity_graphs[i], reward, penalty))

        outputs = torch.stack(outputs)
        rp_graph = (self.weights.view(self.num_score, 1, 1) * outputs).sum(dim=0)

        value = torch.tensor(value).sum()

        return rp_graph, value

    def cal_value(self, graphs, reward, penalty):
        reward_graph = graphs[0, :, :]
        penalty_graph = graphs[1, :, :]

        value = torch.sum(F.relu(reward * reward_graph + penalty * penalty_graph)) / (self.args.num_subjects ** 2)
        return value


class Multimodal_Attention(nn.Module):
    def __init__(self, args):
        super(Multimodal_Attention, self).__init__()
        self.channel = args.out
        self.args = args
        self.shared_lin = nn.Linear(self.channel, self.channel)
        self.img_lin = nn.Linear(self.channel, self.channel)
        self.ph_lin = nn.Linear(self.channel, self.channel)

        # Importance Weight Vector logits (2-modalities). Softmax -> IWV.
        # End-to-end learnable; serves as dynamic modality confidence.
        # if args.module3 == 1:
        #     self.iwv_logits = nn.Parameter(torch.zeros(2))

    def cal_attention_score(self, attention, shared_attention):
        attention_score = torch.trace(torch.mm(attention, attention.t())) / torch.trace(
            torch.mm(shared_attention, shared_attention.t()))
        return attention_score

    def forward(self, img_embed, ph_embed):
        shared_embed = 0.5 * (img_embed + ph_embed)

        # base attentions
        img_attention = F.tanh(self.img_lin(img_embed))
        ph_attention = F.tanh(self.ph_lin(ph_embed))
        shared_attention = F.tanh(self.shared_lin(shared_embed))

  

        # modality weight calculation
        img_shared_score = self.cal_attention_score(img_attention, shared_attention)
        ph_shared_score = self.cal_attention_score(ph_attention, shared_attention)
        
        attention_scores = F.softmax(torch.tensor([img_shared_score, ph_shared_score]), dim=0)
        img_weight = attention_scores[0]
        ph_weight = attention_scores[1]
        # 消融实验：权重设置，完全依赖表型特征
        # img_weight = torch.tensor(0.0, device=ph_embed.device)  # 图片权重设为0
        # ph_weight = torch.tensor(1.0, device=ph_embed.device)   # 表型权重设为1

        # modality joint Representation
        joint_embed = shared_attention * shared_embed + img_attention * img_embed + ph_attention * ph_embed
        # 消融实验
        # joint_embed = ph_attention * ph_embed
        return joint_embed, img_weight, ph_weight

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

class CSDAPGNet(nn.Module):
    def __init__(self, args, fold):
        super(CSDAPGNet, self).__init__()
        self.args = args
        self.fold = fold
        self.device = args.device
        self.dropout = args.dropout
        self.edge_drop = args.edge_drop
        self.load_pretrain = True
        self.setup_vae_pretrain()
        self.setup_attenton()

        # # 添加ACL模块-module1
        if args.module1 == 1:
            self.acl_module = AdaptiveConfidenceLearning(args)
            # self.alpha_fusion = args.alpha_fusion if hasattr(args, 'alpha_fusion') else 0.7  # 融合权重
            if (args.dataset == 'TY'):
                self.alpha_fusion = 0.8
            else:
                self.alpha_fusion = 0.8

            # 添加fil模块-module2
        if args.module2 == 1 and args.module1 == 1:
             # FIL模块初始化
            self.FIL = FILmodule(16,32,args.out,2,F.relu)
        # 参数说明：
        # - in_dim=1: 输入特征维度
        # - n_hidden=32: 隐藏层维度
        # - out_dim=args.hidden_dim: 输出维度
        # - n_layers=2: FIL层数
        # - activation=F.relu: 激活函数
        
         # 新增：跨模态细粒度注意力层
        self.cross_modal_attention = CrossModalFineGrainedAttention(
            feature_dim=args.node_dim,
            temperature=(args.node_dim ** 0.5)
        )


        
        self.img_unet = GTUNet(in_channels=args.node_dim, hidden_channels=args.hidden, out_channels=args.out,
                            depth=args.img_depth, edge_dim=1, pool_ratios=args.pool_ratios, dropout=args.dropout)
        self.ph_unet = GTUNet(in_channels=args.node_dim, hidden_channels=args.hidden, out_channels=args.out,
                            depth=args.ph_depth, edge_dim=1, pool_ratios=args.pool_ratios, dropout=args.dropout)

        self.clf = nn.Sequential(
            torch.nn.Linear(args.out, 256),
            torch.nn.ReLU(),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, args.num_classes))

    def setup_vae_pretrain(self):
        self.vae = VAE(self.args)
        self.init_vae_optimizer()
        self.init_vae_save_path()
        self.init_vae_early_stop()

    def setup_attenton(self):
        self.rp_attention = RP_Attention(self.args)
        self.mm_attention = Multimodal_Attention(self.args)
        

    def init_vae_save_path(self):
        self.vae_save_path = self.args.ckpt_path + "/fold{}_pretrain.pth".format(self.fold)

    def init_vae_optimizer(self):
        """
        - 主要功能 ：清零（重置）VAE优化器中所有参数的梯度累积值
        - 作用机制 ：PyTorch 默认会累积梯度，如果不清零，新计算的梯度会与之前的梯度相加，导致梯度爆炸或训练不稳定
        - 必要性 ：在每次反向传播之前必须调用，确保当前批次的梯度计算是独立的
        """
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.args.vae_lr,
                                              weight_decay=5e-4)

    def init_vae_early_stop(self):
        self.early_stopping = EarlyStopping(patience=self.args.early_stop, verbose=True)

    def load_vae(self):
        """
        - 参数加载 ：
            - torch.load(self.vae_save_path, map_location=self.device) ：从指定路径加载预训练模型的状态字典
            - self.vae.load_state_dict() ：将加载的参数应用到当前的 VAE 模型中
            - map_location=self.device ：确保模型参数加载到正确的设备上

        - 参数冻结 ：
            - 遍历 VAE 模型的所有参数
            - 将每个参数的 requires_grad 属性设置为 False
            - 这样可以防止在主模型训练过程中更新 VAE 的参数
            self.vae_save_path ：VAE 模型保存路径，格式为 "{ckpt_path}/fold{fold}_pretrain.pth"
        """
        # loading pre-trained parameters
        self.vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))

        # freeze the parameters
        for param in self.vae.parameters():
            param.requires_grad = False

    def train_vae(self, ph_features):
        print("Start pretraining vae...")
        best_loss = 1e50
        best_epo = 0
        ph_features = ph_features.to(self.device)
        for epoch in range(3000):
            # - 主要作用 ：将 VAE（变分自编码器）模型设置为训练模式 - PyTorch 方法 ：这是 PyTorch 中 nn.Module 类的内置方法，用于控制模型的训练/评估状态
            self.vae.train()
            # 清零（重置）VAE优化器中所有参数的梯度累积值
            self.vae_optimizer.zero_grad()
            # 是 PyTorch 的一个上下文管理器 用于控制自动梯度计算的开启或关闭状态。
            with torch.set_grad_enabled(True):
                """
                - 特征编码与重构 ：将输入的表型特征编码到潜在空间，然后重构回原始空间
                - 损失计算 ：计算 VAE 的总损失（重构损失 + KL散度损失）
                ## 输出
                这行代码返回两个值，但使用了解构赋值：

                ### 第一个返回值（用 _ 忽略）
                - 数据类型 ： torch.Tensor
                - 形状 ： [batch_size, latent_dim]
                - 内容 ：潜在空间表示 z
                - 含义 ：表型特征在潜在空间中的编码表示
                - 具体示例 ：如果 latent_dim = 500 ，则形状为 [32, 500]
                ### 第二个返回值（赋值给 loss）
                - 数据类型 ： torch.Tensor （标量）
                - 内容 ：VAE 的总损失
                - 计算公式 ： total_loss = reconstruction_loss + kl_divergence
                - 组成部分 ：
                - reconstruction_loss ：重构损失，衡量重构数据与原始数据的差异
                - kl_divergence ：KL散度损失，正则化潜在空间分布
                """
                _, loss = self.vae(ph_features)

            # 反向传播触发，根据损失来计算梯度
            loss.backward()
            # 更新 self.vae 模型内部的权重和偏置参数
            self.vae_optimizer.step()

            if epoch % 100 == 0:
                print(
                    "Epoch: {},\tlr: {:.5f},\tloss: {:.5f}".format(epoch, self.vae_optimizer.param_groups[0]['lr'],
                                                                   loss.item()))
            if best_loss > loss:
                best_loss = loss
                best_epo = epoch
                # - 功能 ：检查是否设置了模型保存路径
                # - 逻辑 ：只有当用户指定了保存路径时才进行模型保存操作
                # - 来源 ： self.args.ckpt_path 来自配置参数，默认为 './save_model/{model}_{dataset}_{atlas}/'
                if (self.args.ckpt_path != ''):
                    if not os.path.exists(self.args.ckpt_path):
                        # - 功能 ：保存VAE模型的状态字典到磁盘
                        # - 参数说明 ：
                        # - self.vae.state_dict() ：获取VAE模型的所有可学习参数
                        # - self.vae_save_path ：保存路径，格式为 "{ckpt_path}/fold{fold}_pretrain.pth
                        os.makedirs(self.args.ckpt_path)
                    torch.save(self.vae.state_dict(), self.vae_save_path)
                    # print("Epoch:{} {} Saved vae to:{}".format(epoch, "\u2714", self.vae_save_path))
# 当VAE损失在100个连续epoch内没有显著改善时，自动停止训练
            self.early_stopping(loss, self.vae)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        print("\r\n => Fold {} best pretrain vae loss {:.5f}, epoch {}\n".format(self.fold, best_loss, best_epo))

    def create_rp_graph(self, affinity_graphs):
        """
        第一个返回值： self.rp_graph
        - 数据类型 ： torch.Tensor
        - 形状 ： [num_subjects, num_subjects]
        - 内容 ：融合后的奖励-惩罚图
        - 数值范围 ： [0, 1] （经过 sigmoid 激活）
        - 含义 ：表示受试者之间的自适应连接权重
        
        第二个返回值： self.value
        - 数据类型 ： torch.Tensor （标量）
        - 内容 ：总体价值评估
        - 计算公式 ：所有评分标准的价值之和
        - 含义 ：衡量当前图结构的整体质量
        - 用途 ：后续用于奖励损失计算
        """
        rp_graph, value = self.rp_attention(affinity_graphs)

        return rp_graph, value

    def cal_graph_loss(self, img_embed, ph_embed):
        L = torch.diagflat(torch.sum(self.fused_graph, -1)) - self.fused_graph
        # smoothness loss
        # 消融实验
        img_smh_loss = self.cal_smh_loss(img_embed, L)
        ph_smh_loss = self.cal_smh_loss(ph_embed, L)

        # degree loss
        deg_loss = self.cal_deg_loss()

        # graph regularization
        # 消融实验
        img_loss = self.args.smh * img_smh_loss + self.args.deg * deg_loss
        ph_loss = self.args.smh * ph_smh_loss + self.args.deg * deg_loss

        # reward regularization
        reward_loss = self.cal_reward_loss()

        # 消融实验
        graph_loss = self.img_weight * img_loss + self.ph_weight * (ph_loss + reward_loss)
        #消融实验
        # graph_loss =  self.ph_weight * (ph_loss + reward_loss)

        return graph_loss

    def cal_smh_loss(self, embed, L):
        smh_loss = torch.trace(torch.mm(embed.T, torch.mm(L, embed)) / torch.prod(
            torch.tensor(self.fused_graph.shape, dtype=torch.float)))
        return smh_loss

    def cal_deg_loss(self):
        one = torch.ones(self.fused_graph.size(-1)).to(self.device)
        deg_loss = torch.sum(torch.mm(self.fused_graph, one.unsqueeze(-1) + 1e-5).log()) / self.fused_graph.shape[
            -1]
        return deg_loss

    def cal_reward_loss(self):
        reward_loss = self.args.val * (1 / (self.value + 1e-5))
        return reward_loss

    
    def forward(self, img_features, ph_features, affinity_graphs):
        """
        img_features : 图像特征张量 形状： [num_subjects, node_dim]
        ph_features : 表型特征张量 形状： [num_subjects, num_phenotypic_features]
        affinity_graphs : 亲和图张量  形状： [num_score, 3, num_subjects, num_subjects]
        """

        # 移除掉图像模块
        img_features = img_features.to(self.device)
        ph_features = ph_features.to(self.device)
        affinity_graphs = affinity_graphs.to(self.device)


        # ph_features111 : 表型特征张量 形状： [num_subjects, num_phenotypic_features]
        if self.args.module1 == 1:
            if self.args.dataset == 'TY':
                # print('TY dataset')
                ph_features111 = ph_features[:, :3].clone()
            else:
                # print('Other dataset,')
                ph_features111 = ph_features

        # reconstruction of phenotypic features
        # self.load_pretrain ：布尔值，在 CSDAPGNet 类的 __init__ 方法中被设置为 True
        if self.load_pretrain:
            # - 模型参数加载 ：将预训练的 VAE 模型参数加载到 self.vae 中
            # - 参数冻结 ：将 VAE 模型的所有参数设置为不可训练状态（ requires_grad = False ）
            self.load_vae()

        """
        第一个返回值（赋值给 ph_features ）
        - 数据类型 ： torch.Tensor
        - 形状 ： [batch_size, latent_dim]
        - 内容 ：表型特征在潜在空间中的编码表示 z
        - 维度 ： latent_dim = args.node_dim （默认为 500）
        - 含义 ：通过 VAE 编码器-重参数化过程得到的高维特征表示
        - 具体示例 ：如果 latent_dim = 500 ，则输出形状为 [32, 500] 

        第二个返回值（用 _ 忽略）
        - 数据类型 ： torch.Tensor （标量）
        - 内容 ：VAE 的总损失值
        - 计算公式 ： total_loss = reconstruction_loss + kl_divergence
        - 组成部分 ：
        - reconstruction_loss ：重构损失，衡量重构数据与原始数据的差异
        - kl_divergence ：KL散度损失，正则化潜在空间分布
        """
        ph_features, _ = self.vae(ph_features)

        
        if self.args.module3 == 1:
            img_features_enhanced, ph_features_enhanced = self.cross_modal_attention(
                img_features, ph_features
            )
            # 更新特征变量
            img_features = img_features_enhanced
            ph_features = ph_features_enhanced



        # construct the affinity graph
        self.rp_graph, self.value = self.create_rp_graph(affinity_graphs)

        # feature fusion and calculate the similarity of fused multimodal features
        """
        dim=1
        - 参数含义 : 指定拼接的维度
        - 维度说明 : 在第1维（特征维度）上进行拼接
        - PyTorch约定 : dim=0为样本维度，dim=1为特征维度
        fused_embed
        - 数据类型 : torch.Tensor
        - 形状 : [num_subjects, 2 * node_dim]
        - 默认形状 : [num_subjects, 1000] (500 + 500)
        - 内容结构 :
        ```
        fused_embed = [img_features | ph_features]
        ```
        - 数据含义 : 每行包含一个被试的完整多模态特征表示
        """
        # 消融实验 移除图像模块
        fused_embed = torch.cat((img_features, ph_features), dim=1)
        # fused_embed = ph_features

        self.fused_sim = cal_feature_sim(fused_embed)

        # compute adaptive reward population graph
        # self.fused_graph ：自适应融合图 形状：[num_subjects, num_subjects]
        # 含义：结合了特征相似性和表型信息的最终图结构
        # 作用：既保留了多模态特征的相似性信息，又融入了基于表型数据的自适应权重
        # fused_graph[i][j] = fused_sim[i][j] × rp_graph[i][j] 
        self.fused_graph = self.fused_sim * self.rp_graph
        """
        功能 ：将密集邻接矩阵转换为稀疏图表示格式

        输入 ：

        - self.fused_graph ：密集邻接矩阵，形状[num_subjects, num_subjects]
        输出 ：

        - fused_index ：边索引矩阵
        - 数据类型：torch.Tensor (long)
        - 形状：[2, num_edges]
        - 内容：每列表示一条边的起点和终点索引
        - 格式：[[source_nodes], [target_nodes]]
        - 示例：[[0, 1, 2], [1, 2, 0]] 表示边 (0→1), (1→2), (2→0)

        - fused_attr ：边权重向量
        - 数据类型：torch.Tensor (float)
        - 形状：[num_edges]
        - 内容：对应每条边的权重值
        - 来源：原密集矩阵中非零元素的值
        """
        fused_index, fused_attr = dense_to_sparse(self.fused_graph)
        # print('fused_index:'+ str(fused_index))
        # print('fused_index.shape:'+ str(fused_index.shape))
        # print('fused_attr:'+ str(fused_attr))
        # print('fused_attr.shape:'+ str(fused_attr.shape))
        # **新增：ACL模块生成边重要性权重Q**
        if self.args.module1 == 1:
            Q_weights = self.acl_module( ph_features111, fused_index)
            enhanced_attr = self.alpha_fusion * fused_attr + (1 - self.alpha_fusion) * Q_weights
        # print('Q_weights:'+ str(Q_weights))
        # print('Q_weights.shape:'+ str(Q_weights.shape))
        # **融合原有权重和ACL权重**
            
        # print('self.alpha_fusion:'+ str(self.alpha_fusion))
        # print('enhanced_attr:'+ str(enhanced_attr))
        
        

        """
        功能 : 将一维张量重塑为二维张量
        参数解释 :
         -1 : 表示该维度的大小由PyTorch自动推断
          1 : 表示第二个维度的大小固定为1
        """
        
        if self.args.module1 == 1:
            enhanced_attr = enhanced_attr.view(-1, 1)
        else:
            fused_attr = fused_attr.view(-1, 1)

        # edge dropout 
        """
        - self.training : 检查模型是否处于训练模式
        - 只有在训练时才应用dropout，推理时不使用
        - 这是PyTorch的标准做法
        - self.edge_drop > 0 : 检查边缘dropout率是否大于0
        - 根据配置文件，默认值为0.3（30%的边会被随机丢弃）
        - 如果设置为0，则不进行边缘dropout

        ### 输出结果 第一个返回值： fused_index （更新后的边索引）
        - 数据类型 ： torch.Tensor (long类型)
        - 形状 ： [2, num_remaining_edges]
        - 内容 ：经过随机丢弃后剩余的边索引
        - 特点 ： num_remaining_edges ≈ (1 - edge_drop_rate) × num_edges 第二个返回值： fused_mask （边掩码）
        - 数据类型 ： torch.Tensor (bool类型)
        - 形状 ： [num_edges]
        - 内容 ：布尔掩码，指示原始边中哪些被保留
        - 值含义 ：
        - True ：对应的边被保留
        - False ：对应的边被丢弃
        - 用途 ：用于同步更新边属性 fused_attr

        # 假设原始边索引
        fused_index = torch.tensor([[0, 1, 2, 0], 
                                [1, 2, 0, 2]])  # 4条边

        # 经过30%的edge dropout后可能得到
        fused_index_new = torch.tensor([[0, 2, 0], 
                                    [1, 0, 2]])  # 保留3条边

        fused_mask = torch.tensor([True, False, True, True])  # 第2条边被丢弃
        """
        if self.training and (self.edge_drop > 0):
            fused_index, fused_mask = dropout_edge(fused_index)
            if self.args.module1 == 1:
                enhanced_attr = enhanced_attr[fused_mask]
            else:
                fused_attr = fused_attr[fused_mask]
            

#    img_features (图像特征) 默认形状 : [num_subjects, 500] (node_dim默认为500)
#    ph_features (表型特征)  默认形状 : [num_subjects, 500]
        """
        img_embed (图像嵌入表示)
        - 数据类型 : torch.Tensor
        - 形状 : [num_subjects, args.out]
        - 默认形状 : [num_subjects, 16]
        - 内容 : 经过图U-Net处理后的图像特征嵌入
        - 特点 : 融合了图结构信息的高级特征表示 
        ph_embed (表型嵌入表示)
        - 数据类型 : torch.Tensor
        - 形状 : [num_subjects, args.out]
        - 默认形状 : [num_subjects, 16]
        - 内容 : 经过图U-Net处理后的表型特征嵌入
        - 特点 : 融合了图结构信息的高级特征表示
        """
       
        # 消融实验，移除图片模块
        if self.args.module1 == 1:
            img_embed = self.img_unet(img_features, fused_index, enhanced_attr)
            ph_embed = self.ph_unet(ph_features, fused_index, enhanced_attr)
             
        else:
            img_embed = self.img_unet(img_features, fused_index, fused_attr)
            ph_embed = self.ph_unet(ph_features, fused_index, fused_attr)
            # 使用增强后的边权重
         
        # 添加fil模块-module2
        if self.args.module2 == 1 and self.args.module1 == 1:
            # FIL模块调用 - 这是关键的调用点
            # 步骤2：创建DGL图对象
            num_nodes = self.fused_graph.shape[0]
            g = dgl.graph((fused_index[0], fused_index[1]), num_nodes=num_nodes)
            
            # 步骤3：添加边权重
            g.edata['weight'] = enhanced_attr
        # 输入：
        # - g: DGL图对象
        # - h: 初始节点特征 [num_nodes, input_dim]
        # - edge_features: 边权重特征 [num_edges]
        # 输出：
        # - h: FIL处理后的节点特征 [num_nodes, hidden_dim]
            if self.args.dataset == 'TY':
                img_embed = self.FIL(g,img_embed,enhanced_attr)
                ph_embed = self.FIL(g,ph_embed,enhanced_attr)
            else:
                img_embed = self.FIL(g,img_embed,enhanced_attr)







        """
        ## 输出结果
        ### self.joint_embed
        - 数据类型 ： torch.Tensor
        - 形状 ： [num_subjects, out_channels] （例如 [32, 256] ）
        - 内容 ：融合后的多模态联合特征表示
        - 用途 ：后续用于分类任务的最终特征
        - 所有的线性变换都保持维度不变（ self.channel = args.out = 16 ）
        - 逐元素相乘和相加操作不改变张量形状
        - 因此 joint_embed 的形状为 [num_subjects, 16]
        ### self.img_weight
        - 数据类型 ： torch.Tensor （标量）
        - 数值范围 ： [0, 1]
        - 内容 ：图像模态的重要性权重
        - 用途 ：表示图像特征在融合中的贡献度
        ### self.ph_weight
        - 数据类型 ： torch.Tensor （标量）
        - 数值范围 ： [0, 1]
        - 内容 ：表型模态的重要性权重
        - 用途 ：表示表型特征在融合中的贡献度
        - 约束 ： img_weight + ph_weight = 1 （由 softmax 保证）
        """
        # 消融实验：用表型嵌入替代图像嵌入，但是我不会用，只是占个位置
        # img_embed = ph_embed.clone()  # 使用表型嵌入的副本作为图像嵌入

        # obtain modality-joint representation and update weights of imaging and phenotypic data
        self.joint_embed, self.img_weight, self.ph_weight = self.mm_attention(img_embed, ph_embed)

        # 这行代码调用了 `cal_graph_loss` 方法，计算图正则化损失，用于约束图神经网络的学习过程，
        # 确保学到的特征表示在图结构上具有平滑性和合理性。
        # computing graph loss
        graph_loss = self.cal_graph_loss(img_embed, ph_embed)
        # 这行代码调用了分类器 `clf` 对融合后的多模态特征进行最终的分类预测，是整个 CSDA-PGNet 模型的输出层。
        # 批归一化 和归一化不一样，反正就是进行了一些好的操作，不用管
        # 下面这个没经过softmax
        # softmax处理后
        # - 定义 : 将logits转换为概率分布
        # - 数值范围 : [0, 1]，且所有类别概率之和为1

      


        outputs = self.clf(self.joint_embed)


        

        return outputs, graph_loss

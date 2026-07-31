import numpy as np


def create_reward_penalty_graph(ph_dict, labels, train_idx, val_idx, test_idx, args):
    """
    ### 功能
    该函数根据表型数据和标签信息，为不同的被试构建三种类型的图结构：

    - 奖励图 (Reward Graph) ：表示具有相同标签和相似表型的被试之间的连接
    - 惩罚图 (Penalty Graph) ：表示具有不同标签但相似表型的被试之间的连接
    - 激励图 (Motivation Graph) ：表示标签未知但表型相似的被试之间的连接
    ### 输入参数
    - ph_dict (字典)：包含表型数据的字典，键为表型名称（如 'AGE_AT_SCAN', 'SEX', 'SITE_ID'），值为对应被试的表型数据
    - labels (NumPy数组)：被试的标签数据，通常是二分类标签（如 0 或 1）
    - train_idx (列表/数组)：训练集被试的索引
    - val_idx (列表/数组)：验证集被试的索引
    - test_idx (列表/数组)：测试集被试的索引
    - args (对象)：包含配置参数的对象，特别是 args.scores （表型评分列表）和 args.ages （年龄表型名称）
    ### 输出
    - final_graph (NumPy数组)：形状为 (len(scores), 3, num_nodes, num_nodes) 的四维数组
    - 第一维：表型特征的数量
    - 第二维：3种图类型（奖励图、惩罚图、激励图）
    - 第三、四维：被试数量 × 被试数量的邻接矩阵
    """
    """
    ## 三种图结构的初始化
    ### 3. reward_graph = np.zeros((len(scores), num_nodes, num_nodes))
    - 功能 ：初始化奖励图矩阵
    - 形状 ： (表型特征数, 被试数, 被试数)
    - 示例形状 ： (3, 871, 871) （3个表型特征，871个被试）
    - 含义 ：
    - 奖励条件 ：当两个被试 i 和 j 具有 相同标签 且 相似表型 时， reward_graph[l, i, j] += 1
    - 目的 ：量化具有相同诊断结果和相似表型特征的被试之间的关系
    - 应用 ：用于强化模型对正确分类的学习
    ### 4. penalty_graph = np.zeros((len(scores), num_nodes, num_nodes))
    - 功能 ：初始化惩罚图矩阵
    - 形状 ： (表型特征数, 被试数, 被试数)
    - 示例形状 ： (3, 871, 871)
    - 含义 ：
    - 惩罚条件 ：当两个被试 i 和 j 具有 不同标签 但 相似表型 时， penalty_graph[l, i, j] += 1
    - 目的 ：识别具有相似表型但不同诊断结果的被试对
    - 应用 ：用于惩罚模型的错误分类倾向
    ### 5. motivation_graph = np.zeros((len(scores), num_nodes, num_nodes))
    - 功能 ：初始化激励图矩阵
    - 形状 ： (表型特征数, 被试数, 被试数)
    - 示例形状 ： (3, 871, 871)
    - 含义 ：
    - 激励条件 ：当被试 i 和 j 的 标签未知 （如训练集与验证集/测试集之间）但 表型相似 时， motivation_graph[l, i, j] += 1
    - 目的 ：建立训练集与验证集/测试集之间基于表型相似性的连接
    - 应用 ：用于半监督学习，利用未标记数据的表型信息
    """
    scores = args.scores   # ['SITE_ID', 'SEX', 'AGE_AT_SCAN']
    num_nodes = len(labels) # 等于被试（受试者）的总数

    # reward graph: subjects i and j have the same label and score, r(i,j) + 1
    # penalty_graph: subjects i and j have different labels but the same scores, p(i,j) + 1
    # motivation_graph: the labels of subjects i and j are unknown and their scores are the same, m(i,j) + 1
    # reward_graph- 功能 ：初始化奖励图矩阵- 形状 ： (表型特征数, 被试数, 被试数)- 示例形状 ： (3, 871, 871) （3个表型特征，871个被试）
    # penalty_graph - 功能 ：初始化惩罚图矩阵 - 形状 ： (表型特征数, 被试数, 被试数) - 示例形状 ： (3, 871, 871)
    # motivation_graph - 功能 ：初始化激励图矩阵 - 形状 ： (表型特征数, 被试数, 被试数) - 示例形状 ： (3, 871, 871)
    reward_graph = np.zeros((len(scores), num_nodes, num_nodes))
    penalty_graph = np.zeros((len(scores), num_nodes, num_nodes))
    motivation_graph = np.zeros((len(scores), num_nodes, num_nodes))

    for l, score in enumerate(scores):
        label_dict = ph_dict[score]
        if score in [args.ages, 'FIQ']:
#  对于年龄（ AGE_AT_SCAN ）和智商（ FIQ ）等连续型数据：
#  相似性判断 ：使用 abs(float(label_dict[i]) - float(label_dict[j])) < 2
#  判断两个被试的表型值是否相似（差值小于2）。
            for i in train_idx:
                # 训练集内部连接 ：
                for j in train_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[j]))
                        if val < 2 and (labels[i] == labels[j]):
                            reward_graph[l, i, j] += 1    # 奖励：相似表型 + 相同标签
                        elif val < 2 and (labels[i] != labels[j]):
                            penalty_graph[l, i, j] += 1   # 惩罚：相似表型 + 不同标签
                    except ValueError:  # missing label
                        pass
                # 训练集 ↔ 验证集/测试集 ：如果表型相似，在激励图中建立双向连接
                for k in val_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[k]))
                        if val < 2:
                            motivation_graph[l, i, k] += 1
                            motivation_graph[l, k, i] += 1
                    except ValueError:  # missing label
                        pass
                for v in test_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[v]))
                        if val < 2:
                            motivation_graph[l, i, v] += 1
                            motivation_graph[l, v, i] += 1
                    except ValueError:  # missing label
                        pass

            for i in val_idx:
                # 验证集内部 ：表型相似的被试建立连接 在激励图中建立双向连接
                for k in val_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[k]))
                        if val < 2:
                            motivation_graph[l, i, k] += 1
                    except ValueError:  # missing label
                        pass
                    # 验证集 ↔ 测试集 ：表型相似，在激励图中建立双向连接
                for v in test_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[v]))
                        if val < 2:
                            motivation_graph[l, i, v] += 1
                            motivation_graph[l, v, i] += 1
                    except ValueError:  # missing label
                        pass
# 测试集内部 ：表型相似建立连接 在激励图中建立双向连接
            for i in test_idx:
                for v in test_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[v]))
                        if val < 2:
                            motivation_graph[l, i, v] += 1
                    except ValueError:  # missing label
                        pass
# 对于性别（ SEX ）、站点（ SITE_ID ）等分类型数据：
# 相似性判断 ：使用 label_dict[i] == label_dict[j] 判断两个被试的表型值是否相同。
        else:
            for i in train_idx:
                for j in train_idx:
                    if (label_dict[i] == label_dict[j]) and (labels[i] == labels[j]):
                        reward_graph[l, i, j] += 1  # 奖励：相同表型 + 相同标签
                    elif (label_dict[i] == label_dict[j]) and (labels[i] != labels[j]):
                        penalty_graph[l, i, j] += 1 # 惩罚：相同表型 + 不同标签
                for k in val_idx: #对于分类型数据， 所有跨集合的连接都建立激励图连接 ，不考虑表型是否相同
                    # 这是因为分类型表型（如性别、站点）的信息对于未标记数据同样重要
                    motivation_graph[l, i, k] += 1
                    motivation_graph[l, k, i] += 1
                for v in test_idx:
                    motivation_graph[l, i, v] += 1
                    motivation_graph[l, v, i] += 1

            for i in val_idx:
                for k in val_idx:
                    motivation_graph[l, i, k] += 1
                for v in test_idx:
                    motivation_graph[l, i, v] += 1
                    motivation_graph[l, v, i] += 1

            for i in test_idx:
                for v in test_idx:
                    motivation_graph[l, i, v] += 1

        final_graph = np.stack((reward_graph, penalty_graph, motivation_graph), axis=1)
        return final_graph

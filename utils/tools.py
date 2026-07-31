import numpy as np
import scipy.io as sio
import torch
from nilearn import connectome
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import shutil
import csv
import os
import re
from scipy.spatial import distance
import warnings

# Ignore the specific FutureWarning from Nilearn
warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn.connectome.connectivity_matrices")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', save=False, trace_func=print):
        """
        ### 功能概述
`EarlyStopping` 是一个用于防止模型过拟合的早停机制类。当验证损失在指定的耐心期（patience）内没有改善时，它会自动停止训练过程。
        Args:
            patience (int):  验证损失没有改善后等待的轮数，默认为7
                            
            verbose (bool): 是否打印验证损失改善的消息，默认为False
                            Default: False
            delta (float): 被认为是改善的最小变化量，默认为0,改善的最小阈值
                           
            path (str): 保存检查点的路径，默认为'checkpoint.pt'
            save (bool): 是否保存模型检查点，默认为False                
            trace_func (function): 跟踪打印函数，默认为print
                           
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None    #当前最佳分数（负验证损失）
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.save = save
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score >= self.best_score + self.delta:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# - 文件路径构建 ：

# - 根据 args.data_folder （如 "data/ABIDE_aal/"）和 id_file （"id.txt"）构建完整路径
# - 最终路径： data/ABIDE_aal/id.txt
# - ID文件读取 ：

# - 如果 id.txt 文件存在，直接使用 np.genfromtxt() 读取
# - 如果文件不存在，先调用 create_id_file() 创建文件，然后读取
# - 数量限制 ：

# - 如果指定了 num_subjects ，则只返回前N个被试ID
# - 否则返回所有被试ID
def get_ids(args, id_file="id.txt", num_subjects=None):
    """Obtain the ID of subjects."""
    data_folder = args.data_folder
    id_path = os.path.join(data_folder, id_file)

    if os.path.exists(id_path):
        subject_IDs = np.genfromtxt(id_path, dtype=str)
    else:
        create_id_file(args)
        subject_IDs = np.genfromtxt(id_path, dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


def create_id_file(args, phenotypic_file="Phenotypic_V1_0b_preprocessed1.csv"):
    """Create the ID file."""
    data_folder = args.data_folder
    scores_dict = {}

    phenotype = os.path.join(data_folder, phenotypic_file)
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scores_dict[row[args.key]] = row[args.key]
    l = []
    for v in scores_dict.keys():
        l.append(v)
    fl = os.path.join(data_folder, "id.txt")
    with open(fl, "w") as f:
        for item in l:
            f.write("%s\n" % item)


def subject_connectivity(timeseries, subject, args, kind='correlation', save=True):
    atlas_name = args.atlas
    save_path = args.data_folder
    # print("Estimating %s matrix for subject %s" % (kind, subject))

    # calculate the function matrix according to the type of connection to be calculated
    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


def get_timeseries(subject_list, args):
    # stores a list of time series, the shape of each subject's time series is (timepoints x regions)
    timeseries = []

    data_folder = args.data_folder
    dataset = args.dataset
    atlas_name = args.atlas

    for i in range(len(subject_list)):
        print("\nStart reading timeseries files.\n")
        if dataset == "ABIDE":
            ro_file = [f for f in os.listdir(data_folder) if
                       f.endswith(subject_list[i] + '_rois_' + atlas_name + '.1D')]
        elif dataset == "ADHD":
            ro_file = [f for f in os.listdir(data_folder) if
                       f.endswith(subject_list[i] + '_rois_' + atlas_name + '.1D')]
        else:
            raise ValueError("No such dataset!")
        fl = os.path.join(data_folder, ro_file[0])
        print("\nReading timeseries file %s\n" % fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))
    print("Reading timeseries files finished!")

    return timeseries


def get_fc(file_name, variable, norm):

    """
    功能：从.mat文件中加载功能连接矩阵并进行归一化处理
    参数：
    - file_name : 字符串，.mat文件的完整路径（如 "data/ABIDE_aal/50003_aal_correlation.mat" ）
    - variable : 字符串，.mat文件中要提取的变量名（通常为 "connectivity" ）
    - norm : 布尔值，是否进行归一化处理
    返回：
    - matrix : numpy数组，形状为 (116, 116) 的功能连接矩阵，包含脑区间的相关系数值（范围-1到1）
    """
    matrix = sio.loadmat(file_name)[variable]   # 从.mat文件加载矩阵
    if norm:
        # 临时抑制numpy的警告信息
        # 输入：
# - divide='ignore' : 忽略除零警告
# - invalid='ignore' : 忽略无效值警告（如NaN、inf）
        with np.errstate(divide='ignore', invalid='ignore'): 
            # Fisher-Z normalization  在Fisher-Z变换过程中，相关系数为±1时会产生无穷值，此设置避免产生大量警告信息
            # 功能： 执行反双曲正切变换（Fisher-Z变换）目的 ：将有界的相关系数转换为无界的正态分布数据 
            # 相关系数的分布通常不是正态分布，特别是当真实相关性接近 ±1 时，分布会变得非常偏斜。
            # 输入： matrix : 相关系数矩阵，值域为 [-1, 1]
            # 输出： norm_matrix : Fisher-Z变换后的矩阵，值域为 (-∞, +∞)
            norm_matrix = np.arctanh(matrix)
            norm_matrix[norm_matrix == float('inf')] = 0
        return norm_matrix
    else:
        return matrix


def get_networks(subject_list, args, norm=True, kind='correlation'):

    data_folder = args.data_folder   # 数据文件夹路径
    atlas_name = args.atlas         # 图谱名称（如'aal'）
    variable = args.variable        # .mat文件中的变量名
    dataset = args.dataset         # 数据集名称（'ABIDE'或'ADHD'）

    graphs = []

# - 功能 ：为每个被试构建.mat文件的完整路径
# - 文件命名规则 ： {subject_id}_{atlas_name}_{kind}.mat
# - 示例路径 ： data/ABIDE_aal/50003_aal_correlation.mat
# - 数据集支持 ：目前支持ABIDE和ADHD数据集
    # get the fc matrix
    for subject in subject_list:
        if dataset == 'ABIDE':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        elif dataset == 'ADHD':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        else:
            raise ValueError("No such dataset!")
        if norm:
            # 功能 ：调用 `get_fc` 函数加载和处理矩阵
            norm_matrix = get_fc(fl, variable, norm)
            graphs.append(norm_matrix)
        else:
            fc_matrix = get_fc(fl, variable, norm)
            graphs.append(fc_matrix)

    graphs = np.array(graphs, dtype=np.float32)

    return graphs


def get_upper_triangle_networks(subject_list, args, norm=True, kind='correlation'):
    """
    ## 输入参数
    - self.ids : 被试ID列表，通过读取 id.txt 文件获得
    - norm=True (默认): 是否对功能连接矩阵进行Fisher-Z标准化
    - kind='correlation' (默认): 连接类型，使用相关性计算
    ## 输出结果
    - 返回值 ： numpy.ndarray
    - 形状 ： (n_subjects, n_features)
    - n_subjects ：被试数量
    - n_features ：上三角元素数量（如AAL图谱为6670）
    - 数据类型 ： float32
    - 内容 ：每行代表一个被试的功能连接特征向量
    """
    data_folder = args.data_folder
    atlas_name = args.atlas
    variable = args.variable
    dataset = args.dataset

    all_networks = []

    for subject in subject_list:
        if dataset == 'ABIDE':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        elif dataset == 'ADHD':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        else:
            raise ValueError("No such dataset!")
        norm_matrix = get_fc(fl, variable, norm=norm)
        all_networks.append(norm_matrix)

    all_networks = np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    vec_networks = [mat[idx] for mat in all_networks]
    matrix = np.vstack(vec_networks)

    return matrix


def get_subject_score(subject_list, score, args, phenotypic_file="Phenotypic_V1_0b_preprocessed1.csv"):
    """Obtain phenotypic information of subjects.
    
    ## 输入参数
    - subject_list : 被试ID列表，包含需要获取标签的被试ID
    - score : 标签名称，指定要获取的表型变量名
    - args : 配置参数对象，包含数据集路径等信息
    - phenotypic_file : 表型数据文件名，默认值为 "Phenotypic_V1_0b_preprocessed1.csv"
    ## 输出结果
    - 返回值 ： 字典类型
    - 键 ： 被试ID（字符串）
    - 值 ： 对应标签值（字符串）
    - 示例 ：
        ```
        {
            '0050002': '1',     # 患者
            '0050003': '2',     # 对照组
            '0050004': '-1'     # 缺失值
        }
        ```
    """
    data_folder = args.data_folder
    scores_dict = {}


    

 # 检查文件扩展名，支持CSV和Excel
    if args.use_excel_features:
        phenotype = os.path.join(data_folder, 'processed_standard_data.xlsx')
        # Excel文件处理
        import pandas as pd
        df = pd.read_excel(phenotype)
        
        # 取出每一行
        for _, row in df.iterrows():
            subject_id = str(row[args.key])  # 确保ID为字符串
            if subject_id in subject_list:
                subject_score = row[score]
                # 处理缺失值
                if pd.isna(subject_score) or subject_score == '' or subject_score == 'N/A':
                    scores_dict[subject_id] = '-1'
                else:
                    scores_dict[subject_id] = str(subject_score)
    else:
        phenotype = os.path.join(data_folder, phenotypic_file)
        # 原有的CSV文件处理逻辑
        with open(phenotype) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                subject_id = row[args.key]
                if subject_id in subject_list:
                    subject_score = row[score]
                    # filling missing values with -1
                    if subject_score == '' or subject_score == 'N/A' or subject_score == 'nan':
                        scores_dict[subject_id] = '-1'
                    else:
                        scores_dict[subject_id] = subject_score
    return scores_dict


def cal_feature_sim(features, tensor=True, self_loop=True):
    """
        提供相似度矩阵\n
        形状 ： [num_subjects, num_subjects]\n
        值域 ： (0, 1] （由于使用了 torch.exp ，结果总是正数，最大值为 1）
    """
    if torch.is_tensor(features):
        # calculate the correlation coefficient distance
        """
        这行代码使用 PyTorch 的 torch.cdist 函数计算输入特征张量 features 中所有样本对之间的 欧几里得距离 ，生成一个距离矩阵 dist 。
        p=2 ：距离度量参数
        - 指定使用 L2 范数（欧几里得距离）
        - p=1 表示曼哈顿距离， p=2 表示欧几里得距离
        dist ：PyTorch 张量 (torch.Tensor)
        - 数据类型 ：与输入 features 相同
        - 形状 ： [num_subjects, num_subjects]
        - 内容 ：距离矩阵，其中 dist[i][j] 表示第 i 个样本与第 j 个样本之间的欧几里得距离
        - 特性 ：
            - 对称矩阵： dist[i][j] = dist[j][i]
            - 对角线元素为 0： dist[i][i] = 0 （样本与自身的距离为 0）
            - 所有元素非负： dist[i][j] ≥ 0
        """
        dist = torch.cdist(features, features, p=2)
        # calculate sigma
        # sigma = torch.mean(dist) 这行代码的功能是计算输入距离矩阵 dist 中所有元素的平均值，
        # 并将结果存储在 sigma 变量中。这个 sigma 值在后续的高斯核函数计算中作为尺度参数使用。
        sigma = torch.mean(dist)
        # calculate feature similarity
        """
        功能：提供相似度矩阵
        ## 输出结果
        - feature_sim ：PyTorch 张量 (torch.Tensor)
        - 数据类型 ：与输入 dist 和 sigma 相同
        - 形状 ： [num_subjects, num_subjects]
        - 内容 ：特征相似度矩阵，其中 feature_sim[i][j] 表示第 i 个样本与第 j 个样本之间的相似度得分
        - 值域 ： (0, 1] （由于使用了 torch.exp ，结果总是正数，最大值为 1）
        - 特性 ：
            - 对称矩阵： feature_sim[i][j] = feature_sim[j][i]
            - 对角线元素为 1： feature_sim[i][i] = 1 （样本与自身的相似度最高）
            - 距离越小，相似度越高；距离越大，相似度越低
        """
        feature_sim = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        # - self_loop=True（默认） ：保留对角线元素，每个样本与自身的相似度为1.0
        # - self_loop=False ：移除自环，将对角线元素设为0
        if not self_loop:
            # 创建原矩阵的副本，避免修改原始数据
            feature_sim_no_self_loop = feature_sim.clone()
            # 将对角线元素填充为0
            feature_sim_no_self_loop.fill_diagonal_(0)
            feature_sim = feature_sim_no_self_loop
            # - tensor=True（默认） ：返回PyTorch张量格式，用于深度学习模型
            # - tensor=False ：转换为NumPy数组格式，用于传统机器学习或数据分析
        if not tensor:
            feature_sim = feature_sim.numpy()
        return feature_sim
    else:
        distv = distance.pdist(features, metric='correlation')
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
        if not self_loop:
            np.fill_diagonal(feature_sim, 0)
        if tensor:
            feature_sim = torch.tensor(feature_sim, dtype=torch.float32)
        return feature_sim


def ordinal_encoding(data_dict):
    """
    Map string data to integer, filling missing values with -1.
    ### 功能详解 1. 字符串到整数的映射
    - 为每个唯一的非空字符串值分配一个从0开始的整数编码
    - 使用 value_to_int 字典维护映射关系
    - 按照值在数据中的首次出现顺序进行编码 2. 缺失值处理
    - 将 None 值和空字符串 '' 统一编码为 -1
    - 确保缺失数据的一致性处理 3. 编码过程
    - 第一遍遍历 ：建立唯一值到整数的映射关系
    - 第二遍遍历 ：根据映射关系生成最终的编码字典
    ### 输入参数
    - data_dict : 字典类型
    - 键：被试ID（字符串）
    - 值：原始标签值（字符串类型，如 '1', '2', 'Male', 'Female' 等）
    - 示例： {'0050002': '1', '0050003': '2', '0050004': '', '0050005': '1'}
    ### 输出结果
    - 返回值 : 字典类型
    - 键：被试ID（字符串，与输入相同）
    - 值：编码后的整数标签
    - 缺失值（ None 或空字符串 '' ）被编码为 -1
    - 其他唯一值按出现顺序从 0 开始递增编码
    - 示例： {'0050002': 0, '0050003': 1, '0050004': -1, '0050005': 0}
    """
    value_to_int = {}
    int_list = []
    curr_int = 0

    for v in data_dict.values():
        if v is None or v == '':
            int_list.append(-1)
        elif v not in value_to_int:
            value_to_int[v] = curr_int
            curr_int += 1
            int_list.append(value_to_int[v])
        else:
            int_list.append(value_to_int[v])

    int_dict = {}
    for k, v in data_dict.items():
        if v is None or v == '':
            int_dict[k] = -1
        else:
            int_dict[k] = value_to_int[v]

    return int_dict


def feature_selection(matrix, labels, train_ind, fnum, tensor=True):
    """
    ## 功能
    feature_selection 函数使用递归特征消除（RFE）方法结合岭回归分类器进行特征选择，
    从原始的高维特征矩阵中选择最重要的特征子集，以降低数据维度并提高模型性能。

    ## 输入参数
    - x ( torch.Tensor 或 numpy.ndarray ): 原始特征矩阵，形状为 (num_subjects, num_features) ，包含所有被试的功能连接特征
    - y ( torch.Tensor 或 numpy.ndarray ): 标签向量，形状为 (num_subjects,) ，包含所有被试的分类标签（如正常/患病）
    - train_ind ( list 或 numpy.ndarray ): 训练样本的索引列表，用于指定哪些样本用于训练特征选择器
    - opt.node_dim ( int ): 特征选择后保留的特征数量，即目标维度
    ## 输出结果
    - x ( torch.Tensor ): 降维后的特征矩阵，形状为 (num_subjects, opt.node_dim) ，包含所有被试的选定特征
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator=estimator, n_features_to_select=fnum, verbose=0, step=100)
    # 提取训练样本的特征和标签
    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    if tensor:
        x_data = torch.tensor(x_data, dtype=torch.float32)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


def move_files_to_main_directory(main_directory):
    # Traverse all contents in the main directory
    for root, dirs, files in os.walk(main_directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            new_location = os.path.join(main_directory, name)
            # Move files to the main directory
            shutil.move(file_path, new_location)
        for name in dirs:
            dir_path = os.path.join(root, name)
            # Attempt to remove empty directories
            try:
                os.rmdir(dir_path)
            except OSError:
                print(f"Directory is not empty or an error occurred: {dir_path}")

def replace_non_floats(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        words = lines[i].split()
        for j in range(len(words)):
            try:
                float(words[j])
            except ValueError:
                words[j] = re.sub(r'\S', ' ', words[j])

        lines[i] = ' '.join(words) + '\n'

    with open(file_name, 'w') as f:
        f.writelines(lines)


def print_result(opt, n_folds, accs, sens, spes, aucs, f1,label_name):
    print(f"\n=== 标签 {label_name} 的详细结果 ===")
    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.var(accs)))
    print("=> Average test sensitivity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.var(sens)))
    print("=> Average test specificity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.var(spes)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.var(aucs)))
    print("=> Average test F1-score in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(f1), np.var(f1)))
    print("{} Saved model to:{}".format("\u2714", opt.ckpt_path))


def save_result(opt, n_folds, accs, sens, spes, aucs, f1,label_name):
    result_path = f'./result/{opt.model}_{opt.dataset}_{opt.atlas}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    save_path = os.path.join(result_path, 'result.txt')
    with open(save_path, 'a') as f:
        print("========================================================", file=f)
        print(f"\n=== 标签 {label_name} 的详细结果 ===")
        print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.var(accs)), file=f)
        print("=> Average test sensitivity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.var(sens)), file=f)
        print("=> Average test specificity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.var(spes)), file=f)
        print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.var(aucs)), file=f)
        print("=> Average test F1-score in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(f1), np.var(f1)), file=f)
        print("========================================================", file=f)







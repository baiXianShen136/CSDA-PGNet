import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.tools import (get_ids, get_timeseries, get_networks, get_upper_triangle_networks, get_subject_score,
                         ordinal_encoding, subject_connectivity, move_files_to_main_directory, replace_non_floats)
from nilearn import datasets
from opt import OptInit
import shutil
import glob

class MyDataloader:

    def __init__(self, args, features=None, labels=None, ph_dict=None, ph_data=None):
# - self.data_folder ：数据文件夹路径（如 data/ABIDE_aal ）
# - self.tensor_folder ：保存预处理张量的文件夹路径
# - self.seed ：随机种子，用于数据分割的可重复性
# - self.score_list ：表型数据评分列表（如年龄、性别、站点等）
# - self.ids ：被试ID列表，通过读取 id.txt 文件获得
        self.args = args
        self.data_folder = self.args.data_folder
        self.tensor_folder = os.path.join(self.args.data_folder, 'save_tensor')
        self.seed = args.seed
        self.score_list = args.scores
# - 脑网络的功能连接矩阵的上三角部分
# - 通过 get_upper_triangle_networks() 方法生成
# - 每个被试的功能连接特征被展平为一维向量
# - 形状通常为 [num_subjects, num_features] ，其中 num_features 是连接矩阵上三角部分的元素数量
        self.features = features
# 含义 ：存储分类标签
# - 二分类标签：0 表示健康对照组，1 表示患者组
# - 通过 get_labels() 方法生成
# - 形状为 [num_subjects]
        self.labels = labels
#         含义 ：存储表型数据字典
#         数据内容 ：
# - 键：表型特征名称（如 'AGE_AT_SCAN', 'SEX', 'SITE_ID' 等）
# - 值：对应特征的数值数组
# - 包含所有被试的人口统计学和临床信息
        self.ph_dict = ph_dict
# 含义 ：存储表型数据矩阵
# 数据类型 ：NumPy 数组，dtype 为 float32
# 数据内容 ：
# - 将 ph_dict 中的所有特征组合成矩阵形式
# - 形状为 [num_subjects, num_phenotypic_features]
# - 每行代表一个被试，每列代表一个表型特征
        self.ph_data = ph_data
        try:
            # 这行代码的作用是获取所有被试（受试者）的ID列表。
            self.ids = self.get_subject_IDs()
            print(f'self.ids:{self.ids}')
        except FileNotFoundError:
            self.ids = None
# 自己写的从excel加载向量
    def load_excel_features(self):
        """
        从Excel文件加载图片特征
        输入：Excel文件路径（行=受试者，列=特征）
        输出：numpy数组，形状为 [num_subjects, num_features]
        """
        import pandas as pd
        
        # Excel文件路径（需要根据您的实际路径修改）
        excel_path = os.path.join(self.data_folder, 'processed_standard_data.xlsx')
        
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        
        # 选择以'clip_ear'开头的列作为特征
        clip_ear_columns = [col for col in df.columns if col.startswith('clip_ear')]
        if not clip_ear_columns:
            raise ValueError("未找到以'clip_ear'开头的列")
       # 提取特征数据（只包含以'clip_ear'开头的列）
        features = df[clip_ear_columns].values.astype(np.float32)
        
        print(f"Loaded picture features from Excel: {features.shape}")

        return features


    def load_data(self, save=False):
        """
        输入参数：
            - save (bool, 默认False): 是否保存加载的数据为tensor文件
            输出：
            - 返回四个变量： features , labels , ph_dict , ph_data
        """

         # 如果脑区间特征在Excel中，使用以下代码替换原有的特征加载逻辑
          # 这里我们要自己写一下脑部特征的向量输入，我先留着
        if self.args.use_excel_features:
            print("\nLoading features....")
            # 从Excel加载图片特征
            self.features = self.load_excel_features()
        else:
    #       这个好像没啥用，没用到self.fcs
            print("\nLoading features....")
            self.fcs = self.get_networks(self.ids, norm=True, kind='correlation')
            # - 功能 ：加载或生成上三角特征向量
            # - 逻辑 ：
            # - 如果预保存的tensor文件存在且不强制重新保存，则直接加载
            # - 否则调用 get_upper_triangle_networks() 提取功能连接矩阵的上三角部分
            # - 输出 ： self.features 包含展平的上三角特征向量
            f_name = 'up_triangles.pt'
            f_load_path = os.path.join(self.tensor_folder, f_name)
            if os.path.exists(f_load_path) and (not save):
                self.features = self.load_tensor(f_load_path)
            else:
                self.features = self.get_upper_triangle_networks(self.ids)
        




#         ## 输入参数
# - l_load_path (str): 预保存标签tensor文件的完整路径
# ## 输出结果
# - 返回值 ：加载的PyTorch张量对象
# - 赋值给 ： self.labels
# - 数据内容 ：二分类标签数据
#   - 0 ：健康对照组（Healthy Controls）
#   - 1 ：患者组（Patient Group）
# - 数据类型 ： torch.Tensor ，底层数据类型为 int32
# - 数据形状 ： [num_subjects]
#   - num_subjects ：被试数量
#   - 例如：如果有871个被试，形状为 [871]
# tensor([0, 1, 0, 1, 0, 0, 1, 1, 0, ...])  # 0=健康对照组, 1=患者组
        print("Loading labels....")
        l_name = 'labels' + '.pt'
        l_load_path = os.path.join(self.tensor_folder, l_name)
        if os.path.exists(l_load_path) and (not save):
            self.labels = self.load_tensor(l_load_path)
        else:
            if self.args.use_excel_features:
                # 从Excel文件中读取标签
                import pandas as pd
                import numpy as np
                # 假设Excel文件路径和标签列名
                excel_path = os.path.join(self.args.data_folder, 'processed_standard_data.xlsx')  # 请替换为实际文件名
                df = pd.read_excel(excel_path)
                 # 直接按顺序读取标签列（假设标签列名为'label'或'DX_GROUP'）
                labels_list = df[self.args.labels].tolist()
                self.labels = np.array(labels_list, dtype=np.int32)
                print(f'self.labels:{self.labels}')
            else:
                self.labels = self.get_labels(self.ids, binary=True)




        print("Loading phenotypic data....")
#  这四行代码负责构建表型数据（phenotypic data）文件的名称和完整路径，用于数据的保存和加载。
#  它们为表型字典和表型数据矩阵创建了对应的文件路径。

#  输入
# - self.score_list : 表型评分列表，来自配置参数 args.scores ，通常包含 ['SITE_ID', 'SEX', 'AGE_AT_SCAN']
# - self.tensor_folder : 张量文件保存目录路径，通常为 data/ABIDE_aal/save_tensor
# 最终结果： 'ph_dict_SITE_ID_SEX_AGE_AT_SCAN.pt' 'ph_data_SITE_ID_SEX_AGE_AT_SCAN.pt'
        dict_name = 'ph_dict_' + str('_'.join(self.score_list)) + '.pt'
        data_name = 'ph_data_' + str('_'.join(self.score_list)) + '.pt'
        dict_load_path = os.path.join(self.tensor_folder, dict_name)
        data_load_path = os.path.join(self.tensor_folder, data_name)
        if os.path.exists(dict_load_path) and os.path.exists(data_load_path) and (not save):
            self.ph_dict = self.load_tensor(dict_load_path)
            self.ph_data = self.load_tensor(data_load_path)
        else:
            self.ph_dict, self.ph_data = self.get_phenotypic_data(self.ids, self.score_list)

        print("Data has loaded!\n")

        if save:
            self.save_tensor(self.features, f_name, self.tensor_folder)
            self.save_tensor(self.labels, l_name, self.tensor_folder)
            self.save_tensor(self.ph_dict, dict_name, self.tensor_folder)
            self.save_tensor(self.ph_data, data_name, self.tensor_folder)
            print("Data has saved!")
        return self.features, self.labels, self.ph_dict, self.ph_data

# 输入 ：

# - get_subject_IDs() 方法接受两个可选参数：
#   - id_file="id.txt" ：ID文件名，默认为 "id.txt"
#   - num_subjects=None ：限制被试数量，默认为 None（使用所有被试）
# 输出 ：

# - self.ids ：一个包含所有被试ID的NumPy字符串数组
# - 例如： ['50003', '50004', '50005', ...]
    def get_subject_IDs(self, id_file="id.txt", num_subjects=None):
        subject_IDs = get_ids(self.args, id_file, num_subjects)
        return subject_IDs

    def get_timeseries(self, subject_list):
        timeseries = get_timeseries(subject_list, self.args)
        return timeseries

    def calculate_connectivity(self, timeseries, subject, kind='correlation', save=True):
        connectivity = subject_connectivity(timeseries, subject, self.args, kind=kind, save=save)
        return connectivity

    def get_networks(self, subject_list, norm=True, kind='correlation'):
        """
        功能 ：加载功能连接矩阵
        输入参数：
        - subject_list : 被试ID列表
        - args : 配置参数对象（包含数据路径、图谱名称等）
        - norm : 是否进行Fisher-Z标准化（默认True）
        - kind : 连接类型（默认'correlation'，相关性）
        输出：self.fcs 存储完整的功能连接矩阵
        """
#         输入参数：

# - subject_list : 被试ID列表
# - args : 配置参数对象（包含数据路径、图谱名称等）
# - norm : 是否进行Fisher-Z标准化（默认True）
# - kind : 连接类型（默认'correlation'，相关性）
# 输出：
# - graphs : numpy数组，形状为 (n_subjects, n_rois, n_rois) ，包含所有被试的功能连接矩阵
        networks = get_networks(subject_list, self.args, norm, kind)
        return networks

    def get_upper_triangle_networks(self, subject_list, norm=True):
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
        networks = get_upper_triangle_networks(subject_list, self.args, norm)
        return networks

    def get_sub_scores(self, subject_list, score, encode=True):
        """
        ## 功能
        这行代码的主要功能是 获取指定被试列表的标签信息 ，具体包括：
        - 从表型数据文件（通常是CSV文件）中读取被试的标签数据
        - 对标签数据进行序数编码（ordinal encoding）处理,若encode=False，就不序数编码
        - 返回一个字典，包含每个被试ID及其对应的编码后标签值
        ## 输入
        - subject_list ：被试ID列表，类型为列表或数组，包含需要获取标签信息的被试ID
        - 例如： ['50003', '50004', '50005', ...]
        - self.args.labels ：标签列名，类型为字符串，指定要从表型数据文件中提取的标签字段名
        - 在ABIDE数据集中，通常为 'DX_GROUP' ，表示诊断组别
        ## 输出
        - information ：字典类型，键为被试ID（字符串），值为编码后的标签值（整数）
        - 数据结构： {'被试ID': 编码值, ...}
        - 例如： {'50003': 0, '50004': 1, '50005': 0, ...}
        - 编码规则：
            - 健康对照组（通常原始值为1）→ 编码为0
            - 患者组（通常原始值为2）→ 编码为1
            - 缺失值或无效值 → 编码为-1
        """
        args = self.args
        if encode:
            scores = ordinal_encoding(get_subject_score(subject_list, score, args))
        else:
            scores = get_subject_score(subject_list, score, args)
        return scores

    def get_labels(self, subject_list, binary=True):
        """
        ## 功能
        这行代码的主要功能是：

        - 标签获取 ：调用 `get_labels` 方法来获取数据集中所有被试的分类标签
        - 二值化处理 ：将多分类标签转换为二分类标签（健康对照 vs 患者）
        - 标签存储 ：将处理后的标签数据存储在 self.labels 属性中，供后续模型训练使用
        ## 输入
        - self.ids ：被试ID列表，类型为 NumPy 字符串数组，包含所有需要处理的被试标识符
        
        - 例如： ['50003', '50004', '50005', ...]
        - 这个列表通过 self.get_subject_IDs() 方法从 id.txt 文件中读取获得
        - binary=True ：布尔参数，控制是否进行二值化处理
        
        - True ：将多分类标签转换为二分类（健康对照=0，所有患者亚型=1）
        - False ：保持原始的多分类标签
        ## 输出
        - self.labels ：NumPy 数组，包含所有被试的分类标签
        - 数据类型 ： np.int32
        - 形状 ： [num_subjects] ，其中 num_subjects 是被试总数
        - 内容 ：
            - 当 binary=True 时：
            - 0 ：健康对照组（Healthy Controls）
            - 1 ：患者组（包括所有患者亚型）
            - 当 binary=False 时：保持原始标签值（可能包含多个患者亚型）
        ## 示例
        假设原始标签为： [0, 1, 2, 0, 3, 1] （0=健康，1-3=不同患者亚型）

        - 当 binary=True 时，输出： [0, 1, 1, 0, 1, 1]
        - 当 binary=False 时，输出： [0, 1, 2, 0, 3, 1]
        """
        information = self.get_sub_scores(subject_list, self.args.labels)
        # - 提取字典值 ： information.values() 获取字典中所有的值（标签编码）
        # - 转换为列表 ： list(information.values()) 将字典值转换为Python列表
        # - 创建NumPy数组 ： np.array(...) 将列表转换为NumPy数组
        # - 指定数据类型 ： dtype=np.int32 确保所有元素都是32位整数类型
        values = np.array(list(information.values()), dtype=np.int32)
        if binary:
            # Healthy controls are encoded as 0, and all subtype patients are encoded as 1.
            values[values > 1] = 1
        labels = values
        return labels

    def get_phenotypic_data(self, subject_list, score_list):
        """
        ## 输入
        - self.ids ( numpy.ndarray )：被试ID列表
        
        - 类型：字符串数组
        - 内容：所有需要处理的被试标识符
        - 示例： ['50003', '50004', '50005', ...]
        - 来源：通过 self.get_subject_IDs() 方法从 id.txt 文件中读取
        - self.score_list ( list )：表型特征列表
        
        - 类型：字符串列表
        - 内容：需要提取的表型特征名称
        - 示例： ['SITE_ID', 'SEX', 'AGE_AT_SCAN']
        - 说明：每个元素对应表型数据文件中的一个列名
        ## 输出
        该方法返回两个变量：

        ### 1. self.ph_dict (字典类型)
        - 数据结构 ： {特征名: numpy数组, ...}
        - 键 ：表型特征名称（字符串）
        - 值 ：对应特征的数据数组
        - 示例 ：
        ```
        {
            'SITE_ID': array([1, 2, 1, 3, ...], dtype=int32),
            'SEX': array([1, 2, 1, 2, ...], dtype=int32),
            'AGE_AT_SCAN': array([25.3, 18.7, 22.1, ...], 
            dtype=float32)
        }
        ```
        ### 2. self.ph_data (PyTorch张量)
        - 数据类型 ： torch.Tensor (float32)
        - 形状 ： [num_subjects, num_phenotypic_features]
        - 内容 ：所有表型特征组合成的矩阵
        - 示例形状 ： [871, 3] （871个被试，3个表型特征）
        - 数据排列 ：每行代表一个被试，每列代表一个表型特征
        """
        args = self.args
        ph_data = []
        ph_dict = {}

# - 循环处理每个表型特征 ：遍历 score_list 中的每个特征名
# - 条件编码 ：
# - 年龄数据： encode=False ，保持连续数值
# - 其他数据： encode=True ，进行分类编码
# - 数据类型转换 ：
# - 年龄：转换为 float32
# - 其他：转换为 int32 ，最终统一为 float32
# - 维度调整 ：使用 np.swapaxes(ph_data, 0, 1) 将数据从 [特征数, 被试数] 转置为 [被试数, 特征数]  0 1 指对 0维度，和1维度进行交换
# - 双格式返回 ：同时返回字典格式（便于特征名索引）和矩阵格式（便于模型输入）
        for score in score_list:
            if score == args.ages:
                information = self.get_sub_scores(subject_list, score, encode=False)
                values = np.array(list(information.values()), dtype=np.float32)
                ph_data.append(values)
                ph_dict[score] = values
            else:
                information = self.get_sub_scores(subject_list, score, encode=True)
                values = np.array(list(information.values()), dtype=np.int32)
                ph_data.append(values)
                ph_dict[score] = values

        ph_data = np.array(ph_data).astype(np.float32)
        ph_data = np.swapaxes(ph_data, 0, 1)
        return ph_dict, ph_data

    def fetch_abide(self, derivatives):
        data_folder = self.args.data_folder
        num_subjects = self.args.num_subjects

        datasets.fetch_abide_pcp(data_dir=data_folder, n_subjects=num_subjects, global_signal_regression=False,
                                 pipeline='cpac', band_pass_filtering=True, derivatives=derivatives)
        move_files_to_main_directory(data_folder)

    def process_abide(self, kind='correlation'):
        self.ids = self.get_subject_IDs()
        time_series = self.get_timeseries(self.ids)
        connectivities = []
        for i in range(len(self.ids)):
            connectivities.append(self.calculate_connectivity(time_series[i], self.ids[i], kind=kind))
            print(f"subject_{self.ids[i]}'s {kind} connectivity has been calculated!")
        print("all finished!")
        connectivities = np.array(connectivities, dtype=np.float32)
        return connectivities

    def process_adhd200(self, kind='correlation', filter=True):
        data_folder = self.args.data_folder
        altas = self.args.atlas
        if filter:
            prefix = 'sfnwmrda'
        else:
            prefix = 'snwmrda'

        self.ids = self.get_subject_IDs()

        for id_ in self.ids:
            file_pattern = os.path.join(data_folder, f'*/{prefix}{id_}*rest_1_{altas}_TCs.1D')
            files = glob.glob(file_pattern)

            for file_name in files:
                replace_non_floats(file_name)
                shutil.move(file_name, os.path.join(data_folder, f'{id_}_rois_{altas}.1D'))
                print(f"{file_name} has processed!")

                subdir_name = os.path.dirname(file_name)
                shutil.rmtree(subdir_name)

        time_series = self.get_timeseries(self.ids)
        connectivities = []
        for i in range(len(self.ids)):
            connectivities.append(self.calculate_connectivity(time_series[i], self.ids[i], kind=kind))

        connectivities = np.array(connectivities, dtype=np.float32)
        return connectivities

    def data_split(self, n_folds, val_ratio=0):
        """
        ## 功能
        data_split 方法负责将数据集进行分层交叉验证分割，支持两种模式：
        1. 标准K折交叉验证 ：当 val_ratio=0 时，只分割训练集和测试集
        2. 三分割模式 ：当 val_ratio>0 时，分割为训练集、验证集和测试集
        ## 输入
        - self : MyDataloader 类的实例，包含已加载的特征数据 ( self.features ) 和标签数据 ( self.labels )
        - n_folds (整数): 交叉验证的折数，例如 5 表示5折交叉验证
        - val_ratio (浮点数，默认为0): 验证集占训练集的比例，例如 0.1 表示验证集占训练集的10%
        ## 输出
        - cv_splits (列表): 包含每一折数据分割索引的列表
        - 当 val_ratio=0 时：返回 [(train_index, test_index), ...] 格式的元组列表
        - 当 val_ratio>0 时：返回 [(train_index, val_index, test_index), ...] 格式的元组列表
        """



        """
        skf.split(self.features, self.labels):
        - 分层采样 ：确保每个折（fold）中各类别（健康对照组和患者组）的比例与原始数据集保持一致
        - 交叉验证 ：将数据集分成 K 个互不重叠的子集，每次使用其中一个作为测试集，其余作为训练集
        - 索引生成 ：返回每个折的训练集和测试集的样本索引，而不是实际的数据
        ## 输入
        - self.features ( torch.Tensor )：功能连接特征数据
        - 形状： [num_subjects, num_features]
        - 内容：脑区间功能连接矩阵的上三角特征向量
        - 示例形状： [871, 4005] （871个被试，4005个特征）
        - 作用： split() 方法使用特征数据的第一维度（样本数量）来确定如何分割数据

        - self.labels ( torch.Tensor )：标签数据
        - 形状： [num_subjects]
        - 内容：二值化标签（0=健康对照组，1=患者组）
        - 示例： [0, 1, 1, 0, 1, 0, ...]
        - 作用：用于分层采样，确保每个折中各类别比例保持一致
        ## 输出
        skf.split() 返回一个 生成器对象 ，每次迭代(一共k个元组)产生一个包含两个元素的元组：

        - train_index ( numpy.ndarray )：训练集样本的索引数组
        - test_index ( numpy.ndarray )：测试集样本的索引数组
        ## 示例：
        Fold 1:\n
        训练集索引: [1, 2, 4, 5, 6, 7, 9, 10, 11, 12]...\n
        测试集索引: [0, 3, 8, 13, 18, 23, 28, 33, 38, 43]...\n
        训练集大小: 80\n
        测试集大小: 20\n
        训练集标签分布: [48, 32]   48个健康对照，32个患者\n
        测试集标签分布: [12, 8]    12个健康对照，8个患者\n
        ## 使用同一组索引提取不同数据
        train_features = self.features[train_index]      提取特征数据
        train_labels = self.labels[train_index]          提取标签数据
        train_ph_data = self.ph_data[train_index]        提取表型数据
        train_ids = self.ids[train_index]                提取被试ID
        """
        # 创建分层K折交叉验证对象，确保每折中各类别比例保持一致
        # n_splits (int): K折交叉验证的折数 - 表示将数据集分成多少个子集
        # shuffle：表示在进行K折交叉验证之前，会先将数据集进行 随机打乱  ，如果设置了随机种子，那么每次随机的结果就都一样，那么每次随机打乱的结果都一样，
        
        skf = StratifiedKFold(n_splits=n_folds, random_state=self.seed, shuffle=True)
        if val_ratio == 0:
            cv_splits = list(skf.split(self.features, self.labels))
        else:
            cv_splits = []
            for train_index, test_index in skf.split(self.features, self.labels):
                """
                ## 功能
                train_test_split 是 scikit-learn 库中的数据分割函数，用于将数据集进一步分割成训练集和验证集。
                ## 输入
                - train_index ( numpy.ndarray )：来自 K 折交叉验证的训练集索引
                - 类型：一维整数数组
                - 内容：原始数据集中用于训练的样本索引
                - 示例： [0, 2, 3, 5, 7, 8, 9, 11, ...] （80个索引，假设总共100个样本的5折交叉验证）
                - 来源： skf.split(self.features, self.labels) 生成的训练集索引

                - test_size=val_ratio (浮点数)：验证集占训练集的比例
                - 类型：浮点数（0.0 到 1.0 之间）
                - 含义：从训练集中分出多少比例作为验证集
                - 示例： val_ratio=0.1 表示将训练集的 10% 作为验证集
                - 计算：如果训练集有 80 个样本，验证集将有 8 个样本，新训练集将有 72 个样本
                - random_state=self.seed (整数)：随机种子
                - 类型：整数
                - 作用：确保每次运行时分割结果相同，保证实验的可重现性
                - 来源： self.seed 是 MyDataloader 类中设置的随机种子

                - stratify=self.labels[train_index] ( torch.Tensor 或 numpy.ndarray )：分层依据
                - 类型：一维数组，包含训练集样本对应的标签
                - 内容：训练集中每个样本的类别标签（0=健康对照组，1=患者组）
                - 作用：确保分割后的训练集和验证集中各类别比例与原始训练集保持一致
                - 示例：如果 train_index（索引数组） 对应的标签分布是 60% 健康对照组，40% 患者组，那么分割后的新训练集和验证集都会保持这个比例
                ## 输出
                train_test_split 返回两个数组：
                - train_index (更新后的训练集索引)：从原始训练集中进一步筛选出的训练样本索引
                - val_index (验证集索引)：从原始训练集中分出的验证样本索引
                """
                train_index, val_index = train_test_split(train_index, test_size=val_ratio, random_state=self.seed,
                                                          stratify=self.labels[train_index])
                cv_splits.append((train_index, val_index, test_index))
        return cv_splits

    def save_tensor(self, tensor, name, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, name)
        torch.save(tensor, save_path)

    def load_tensor(self, load_path):
        """
        ## 功能
        tensor = torch.load(load_path) 是 PyTorch 框架中用于加载预保存张量数据的核心语句。在这个上下文中，它属于 `load_tensor` 方法，主要功能是：
        torch.load() 会完全恢复保存时的张量形状和内容。也就是张量的形状是根据保存为pt时，是什么样子，然后转为张量时，就是什么形状的。
        - 数据加载 ：从磁盘上的 .pt 文件中加载预先保存的 PyTorch 张量对象
        - 缓存机制 ：避免重复计算，直接加载已处理好的数据
        - 数据恢复 ：将序列化的张量数据反序列化为可用的 PyTorch 张量对象
        ## 输入
        - load_path (字符串类型)：预保存张量文件的完整路径
        - 在 load_data 方法的上下文中，这个路径通过以下方式构建：
            - 功能连接特征： f_load_path = os.path.join(self.tensor_folder, f_name) ，其中 f_name = 'up_triangles.pt'
            - 标签数据： l_load_path = os.path.join(self.tensor_folder, l_name) ，其中 l_name = 'labels.pt'
            - 表型字典： dict_load_path = os.path.join(self.tensor_folder, dict_name)
            - 表型数据： data_load_path = os.path.join(self.tensor_folder, data_name)
        - 完整路径示例： data/ABIDE_aal/save_tensor/up_triangles.pt
        ## 输出
        - tensor (PyTorch 张量对象)：从文件中加载的张量数据
        - 数据类型 ：根据保存时的数据类型而定，通常为 torch.Tensor
        - 数据内容 ：根据加载的文件类型不同而不同：
            - 如果是功能连接特征文件：包含功能连接矩阵的上三角特征向量
            - 如果是标签文件：包含二值化的标签数据（0表示健康对照，1表示患者）
            - 如果是表型数据文件：包含表型特征矩阵或字典
        - 数据形状 ：
            - 功能连接特征： [num_subjects, num_features] ，其中 num_features 通常为 num_rois * (num_rois - 1) / 2
            - 标签数据： [num_subjects]
            - 表型数据： [num_subjects, num_phenotypic_features]
        """
        tensor = torch.load(load_path)
        return tensor


if __name__ == "__main__":
    settings = OptInit(dataset="ABIDE", atlas="aal")
    settings.args.data_folder = rf"../data/{settings.args.dataset}_{settings.args.atlas}/"
    opt = settings.initialize()
    dl = MyDataloader(opt)
    y = dl.get_labels(dl.ids)
    ts = dl.get_timeseries(dl.ids)


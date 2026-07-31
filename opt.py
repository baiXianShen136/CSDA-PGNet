import datetime
import argparse
import random
import numpy as np
import torch


def choose_dataset(dataset):
    data_dict = {}
    if dataset == "ABIDE":
        data_dict['num_subjects'] = 871 # 您的被试数量
        data_dict['num_classes'] = 2  # 您的分类数量
        data_dict['key'] = 'SUB_ID' # 被试ID列名
        data_dict['labels'] = 'DX_GROUP'
        data_dict['ages'] = 'AGE_AT_SCAN'
        data_dict['genders'] = 'SEX'
        data_dict['sites'] = 'SITE_ID'
        data_dict['variable'] = 'connectivity'
        data_dict['scores'] = [data_dict['sites'], data_dict['genders'], data_dict['ages']]
        data_dict['use_excel_features'] = False  # 新增：标识使用Excel特征
    elif dataset == "ADHD":
        data_dict['num_subjects'] = 582
        data_dict['num_classes'] = 2
        data_dict['key'] = 'ScanDir ID'
        data_dict['labels'] = 'DX'
        data_dict['ages'] = 'Age'
        data_dict['genders'] = 'Gender'
        data_dict['sites'] = 'Site'
        data_dict['variable'] = 'connectivity'
        data_dict['scores'] = [data_dict['sites'], data_dict['genders'], data_dict['ages']]
        data_dict['use_excel_features'] = False  # 新增：标识使用Excel特征
    elif dataset == "TY":
        data_dict['num_subjects'] = 126
        # data_dict['num_subjects'] = 196
        # data_dict['num_subjects'] = 163  # 您的被试数量
        data_dict['num_classes'] = 2
        data_dict['key'] = '编号'
        data_dict['labels'] = '输出_冠状动脉粥样硬化心脏病'
        data_dict['ages'] = '年龄'
        data_dict['genders'] = '性别'
        data_dict['xiongmen'] = '胸闷'
        data_dict['yatong'] = '压痛'
        # 新增的表征特征
        data_dict['buwei'] = '部位'
        data_dict['fanwei'] = '范围'
        data_dict['xingzhi'] = '性质'
        data_dict['huodongxing'] = '活动性'
        data_dict['chengdu'] = '程度'
        data_dict['fanying'] = '反应'
        data_dict['chixushijian'] = '持续时间'
        data_dict['xiuxihuanjie'] = '休息能否缓解'

        data_dict['variable'] = 'connectivity'
        # 基础表征特征
        base_scores = [data_dict['ages'],data_dict['genders'],data_dict['xiongmen'],data_dict['yatong'],
        data_dict['buwei'],
        data_dict['fanwei'],
        data_dict['xingzhi'],
        data_dict['huodongxing'],
        data_dict['chengdu'],
        data_dict['fanying'],
        data_dict['chixushijian'],
        data_dict['xiuxihuanjie']
        ]
         # 动态获取症状列
        symptom_columns = get_symptom_columns()
        data_dict['scores'] = base_scores + symptom_columns
        data_dict['use_excel_features'] = True  # 新增：标识使用Excel特征
    else:
        print("plase input correct content!")
        raise ValueError
    return data_dict

def get_symptom_columns():
    """
    动态获取Excel文件中以'症状_'开头的列名
    """
    import pandas as pd
    import os
    
    # Excel文件路径（需要根据实际情况调整）
    excel_path = 'data/TY/processed_standard_data.xlsx'
    
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
            # 获取所有以'症状_'开头的列名
            symptom_columns = [col for col in df.columns if col.startswith('症状_')]
            return symptom_columns
        except Exception as e:
            print(f"读取Excel文件时出错: {e}")
            return []
    else:
        print(f"Excel文件不存在: {excel_path}")
        return []


def choose_atlas(atlas):
    """
    num_rois 直接决定了功能连接矩阵的大小：

    - 矩阵维度 ： num_rois × num_rois （如AAL图谱为 116×116）
    - 数据来源 ：从 .mat 文件中加载，包含脑区间的相关系数
    - 处理位置 ： `utils/tools.py` 中的 `get_fc` 函数
    """
    if atlas == 'aal':
        num_rois = 116
    elif atlas == 'ho':
        num_rois = 111
    elif atlas == 'ty':
        num_rois = 123   #随便写的，用不到   上面的num_rois 也是用不到
    return atlas, num_rois


class OptInit:
    def __init__(self, model=None, dataset="ABIDE", atlas="aal"):
        """
        model parameter settings
        作用 :创建配置初始化对象，设置模型、数据集和脑图谱参数，输出 : OptInit对象，包含所有默认配置参数
        参数 :
        - model : 字符串，模型名称（如 "CSDA-PGNet" ）
        - dataset : 字符串，数据集名称（如 "ABIDE" ）
        - atlas : 字符串，脑图谱名称（如 "aal" ）
        """
        data_dict = choose_dataset(dataset)
        atlas, num_rois = choose_atlas(atlas)
        self.parser = argparse.ArgumentParser()
        # data
        self.parser.add_argument('--data_folder', default=rf'data/{dataset}_{atlas}', type=str, help='data folder')
        # self.parser.add_argument('--data_folder', default='data/TY', type=str, help='data folder')
        
        self.parser.add_argument('--atlas', type=str, default=atlas,
                                 help='atlas for network construction (node definition)')
        self.parser.add_argument('--num_rois', type=int, default=num_rois, help='number of brain regions')
        self.parser.add_argument('--num_classes', type=int, default=data_dict['num_classes'], help='number of classes')
        self.parser.add_argument('--num_subjects', type=int, default=data_dict['num_subjects'],
                                 help='number of subjects')
        self.parser.add_argument('--key', type=str, default=data_dict['key'], help='key values for image data')
        self.parser.add_argument('--labels', type=str, default=data_dict['labels'], help='the title of labels')
        self.parser.add_argument('--ages', type=str, default=data_dict['ages'], help='the title of ages')
        self.parser.add_argument('--genders', type=str, default=data_dict['genders'], help='the title of genders')

#       这是TY特有的基础特征
        self.parser.add_argument('--xiongmen', type=str, default=data_dict.get('xiongmen',None), help='the title of xiongmen')
        self.parser.add_argument('--yatong', type=str, default=data_dict.get('yatong',None), help='the title of yatong')
        self.parser.add_argument('--buwei', type=str, default=data_dict.get('buwei',None), help='the title of buwei')
        self.parser.add_argument('--fanwei', type=str, default=data_dict.get('fanwei',None), help='the title of fanwei')
        self.parser.add_argument('--xingzhi', type=str, default=data_dict.get('xingzhi',None), help='the title of xingzhi')
        self.parser.add_argument('--huodongxing', type=str, default=data_dict.get('huodongxing',None), help='the title of huodongxing')
        self.parser.add_argument('--chengdu', type=str, default=data_dict.get('chengdu',None), help='the title of chengdu')
        self.parser.add_argument('--fanying', type=str, default=data_dict.get('fanying',None), help='the title of fanying')
        self.parser.add_argument('--chixushijian', type=str, default=data_dict.get('chixushijian',None), help='the title of chixushijian')
        self.parser.add_argument('--xiuxihuanjie', type=str, default=data_dict.get('xiuxihuanjie',None), help='the title of xiuxihuanjie')
        # 这是TY特有的动态获取 症状_xxx 的特征
        # 动态添加症状列参数
        if dataset == "TY":
            symptom_columns = get_symptom_columns()
            for symptom_col in symptom_columns:
                # 为每个症状列添加参数
                param_name = f'--{symptom_col.replace("症状_", "").replace("_", "")}'
                self.parser.add_argument(param_name, type=str, default=symptom_col, 
                                       help=f'the title of {symptom_col}')

        self.parser.add_argument('--use_excel_features', type=bool, default=data_dict['use_excel_features'], help='the title of use_excel_features')
        

        # 我缝了一些模块，然后以这些模块为基础，添加一些参数，为0就是不使用，为1就是使用
        self.parser.add_argument('--module1', type=int, default=0, help='加了一些模块，如果用就是1，默认为0')
        self.parser.add_argument('--module2', type=int, default=0, help='加了一些模块，如果用就是1，默认为0')
        # self.parser.add_argument('--module3', type=int, default=0, help='加了一些模块，如果用就是1，默认为0')
        self.parser.add_argument('--module3', type=int, default=0, help='加了一些模块，如果用就是1，默认为0')
        self.parser.add_argument('--module4', type=int, default=0, help='加了一些模块，如果用就是1，默认为0')






        # self.parser.add_argument('--sites', type=str, default=data_dict['sites'], help='the title of sites')
        self.parser.add_argument('--scores', default=data_dict['scores'], type=list, help='phenotypic scores')
        self.parser.add_argument('--variable', type=str, default=data_dict['variable'],
                                 help='variable name of .mat file')
        self.parser.add_argument('--dataset', default=dataset, type=str, help='name of dataset')

        # hyper parameters
        self.parser.add_argument('--model', default=model, type=str, help='name of model')
        self.parser.add_argument('--node_dim', type=int, default=500, help='dimension of node features after rfe')
        self.parser.add_argument('--img_depth', default=2, type=int, help='depth of the img_unet')
        self.parser.add_argument('--ph_depth', default=3, type=int, help='depth of the ph_unet')
        self.parser.add_argument('--hidden', type=int, default=128, help='hidden channels of the unet')
        self.parser.add_argument('--out', type=int, default=16, help='out channels of the unet')
        self.parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')
        self.parser.add_argument('--edge_drop', default=0.3, type=float, help='ratio of edge dropout')
        self.parser.add_argument('--pool_ratios', default=0.8, type=float,
                                 help='pooling ratio to be used in the Graph_Unet')
        self.parser.add_argument('--smh', type=float, default=1, help='graph_loss_smooth')
        self.parser.add_argument('--deg', type=float, default=1e-4, help='graph_loss_degree')
        self.parser.add_argument('--val', type=float, default=1e-2, help='graph_loss_value')

        # train parameter
        self.parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')


        # self.parser.add_argument('--train', default=True, type=bool, help='train(default) or test')
        self.parser.add_argument('--train', default=True, type=bool, help='train(default) or test')


        self.parser.add_argument('--seed', type=int, default=911, help='random state')
        self.parser.add_argument('--early_stop', type=int, default=100, help='early stop patience')
        self.parser.add_argument('--lr', default=1e-4, type=float, help='initial model learning rate')
        self.parser.add_argument('--vae_lr', default=1e-3, type=float, help='initial vae learning rate')
        self.parser.add_argument('--wd', default=5e-4, type=float, help='initial weight decay')
        self.parser.add_argument('--epoch', default=500, type=int, help='number of epochs for training')
        # self.parser.add_argument('--folds', default=6, type=int, help='cross validation folds')
        self.parser.add_argument('--folds', default=10, type=int, help='cross validation folds')

        # train setting
        self.parser.add_argument('--log_save', type=bool, default=False, help='save log or not')
        self.parser.add_argument('--model_save', type=bool, default=True, help='save model or not')
        self.parser.add_argument('--result_save', type=bool, default=True, help='save result or not')
        self.parser.add_argument('--print_freq', default=5, type=int, help='print frequency')
        self.parser.add_argument('--ckpt_path', type=str, default=rf'./save_model/{model}_{dataset}_{atlas}',
                                 help='checkpoint path to save trained models')

        args = self.parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.args = args

    def print_args(self):
        # self.args.printer args
        # 输入 ：self.args.__dict__.items() - 获取args对象的所有属性和值
        # 输出 ：以 参数名:参数值 的格式打印每个配置项
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train == 1 else 'eval'
        print('===> Phase is {}.'.format(phase))

# - 调用 self.set_seed(self.args.seed) ：设置随机种子 确保实验结果的可重复性
# - 默认种子值为 911（在构造函数中设置）
    def initialize(self):
        self.set_seed(self.args.seed)
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)  # 设置Python随机数种子
        np.random.seed(seed)   # 设置NumPy随机数种子
        torch.manual_seed(seed)  # 设置PyTorch CPU随机数种子
        torch.cuda.manual_seed(seed)  # 设置当前GPU随机数种子
        torch.cuda.manual_seed_all(seed)   # 设置所有GPU随机数种子
        torch.backends.cudnn.deterministic = True   #确保CUDNN确定性 作用 ：强制cuDNN使用确定性算法 作用 ：禁用cuDNN的自动算法选择优化
        torch.backends.cudnn.benchmark = False    # 禁用CUDNN基准测试    作用 ：禁用cuDNN的自动算法选择优化

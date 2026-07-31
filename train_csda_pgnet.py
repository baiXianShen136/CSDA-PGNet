import torch
import numpy as np
import os
from opt import OptInit
from utils.tools import (EarlyStopping, feature_selection, print_result, save_result)
from utils.metrics import accuracy, auc, metrics
from utils.mydataloader import MyDataloader
from model.csda_pgnet import CSDAPGNet
from model.rp_graph import create_reward_penalty_graph
from tensorboardX import SummaryWriter


def train():
    print("  Number of training samples %d" % len(train_ind))
    print("  Number of validation samples %d" % len(val_ind))
    print("  Start training...")
    acc = 0
    correct = 0
    best_loss = 1e10
    best_epo = 0
    for epoch in range(opt.epoch):
        # 作用 ：将模型设置为训练模式
        model.train()  


        # 作用 ：清零所有参数的梯度
        optimizer.zero_grad()
        # 确保梯度计算开启 ：在上下文块内，所有张量操作都会进行梯度跟踪和计算
        with torch.set_grad_enabled(True):
            outputs, graph_loss = model(x, ph_features, affinity_graphs)
        outputs_train = outputs[train_ind]
        # 组成部分： 分类损失 ：衡量预测标签与真实标签的差异 图损失 ：包括平滑性损失、度损失和奖励损失，用于正则化图结构
        loss = loss_fn(outputs_train, labels[train_ind]) + graph_loss
# - 日志路径 ： ./log/CSDA-PGNet_ABIDE_aal_log/{fold}/
# - 作用 ：将训练指标写入TensorBoard格式的日志文件
        if opt.log_save:
            writer.add_scalar("train\tloss", loss.item(), epoch)

#  计算出损失函数后， 该函数用于 执行反向传播算法，计算损失函数相对于模型所有可学习参数的梯度。
        loss.backward()
        # 功能： 根据计算得到的梯度更新模型参数。
        optimizer.step()

        """
        ## 示例
        假设有3个训练样本：

        - 预测输出： [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]
        - 真实标签： [1, 0, 1]
        - 预测类别： [1, 0, 1] （通过argmax得到）
        - 正确预测： [True, True, True] → [1.0, 1.0, 1.0]
        - correct_train = 3.0
        - acc_train = 1.0 （100%准确率）
        """
        correct_train, acc_train = accuracy(outputs[train_ind].detach().cpu().numpy(), y[train_ind])
        if opt.log_save:
            writer.add_scalar("train\tacc", acc_train, epoch)

# model.eval() 是 PyTorch 中用于将模型设置为 评估模式
        model.eval()
        with torch.set_grad_enabled(False):
            outputs, graph_loss = model(x, ph_features, affinity_graphs)
        loss_val = loss_fn(outputs[val_ind], labels[val_ind]) + graph_loss
        if opt.log_save:
            writer.add_scalar("val\tloss", loss_val.item(), epoch)
        outputs_val = outputs[val_ind].detach().cpu().numpy()
        correct_val, acc_val = accuracy(outputs_val, y[val_ind])
        if opt.log_save:
            writer.add_scalar("val\tacc", acc_val, epoch)

        """
        功能 ：获取当前优化器的学习率
        详细解释 ：

        - optimizer.state_dict() ：返回优化器的状态字典，包含所有参数组和状态信息
        - ["param_groups"] ：访问参数组列表，优化器可以有多个参数组，每个组可以有不同的学习率
        - [0] ：获取第一个参数组（通常只有一个参数组）
        - ["lr"] ：提取该参数组的当前学习率值
        """
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        """
        功能 ：计算验证集的敏感性、特异性和F1分数
        详细解释 ：

        - outputs_val ：验证集的模型预测输出，形状为 [num_val_samples, num_classes]
        - y[val_ind] ：验证集的真实标签
        - `metrics` 函数返回三个指标：
        - 敏感性（Sensitivity） ：真阳性率，计算公式为 TP / (TP + FN)
        - 特异性（Specificity） ：真阴性率，计算公式为 TN / (TN + FP)
        - F1分数 ：精确率和召回率的调和平均，计算公式为 2 * precision * recall / (precision + recall)
        用途 ：评估模型在验证集上的分类性能，特别适用于医学诊断等需要平衡敏感性和特异性的任务
        
        """
        val_sen, val_spe, val_f1 = metrics(outputs_val, y[val_ind])
        """
        详细解释 ：

        - `auc` 函数计算ROC曲线下的面积
        - 首先使用 softmax 将模型输出转换为概率分布
        - 提取正类（类别1）的预测概率： pos_probs = softmax(preds, axis=1)[:, 1]
        - 使用 roc_auc_score 计算AUC值
        """
        val_auc = auc(outputs_val, y[val_ind])
        if epoch % opt.print_freq == 0:
            print(
                "Epoch: {},\tlr: {:.5f},\ttrain loss: {:.5f},\ttrain acc: {:.5f},\teval loss: {:.5f},\teval acc: {:.5f} ,"
                "\teval spe: {:.5f}\teval_sen: {:.5f}".format(epoch, lr, loss.item(), acc_train.item(),
                                                              loss_val.item(), acc_val.item(), val_spe, val_sen))
        if best_loss > loss_val:
            best_loss = loss_val    # 更新最佳验证损失
            best_epo = epoch      # 记录最佳性能对应的训练轮次
            acc = acc_val     # 更新当前fold的最佳准确率
            correct = correct_val  # 更新当前fold的正确预测数量
            aucs[fold] = val_auc
            sens[fold] = val_sen
            spes[fold] = val_spe
            f1[fold] = val_f1
            if (opt.ckpt_path != '') and opt.model_save:
                if not os.path.exists(opt.ckpt_path):
                    os.makedirs(opt.ckpt_path) # 创建保存目录
                torch.save(model.state_dict(), fold_model_path)  # 保存模型参数
                print("Epoch:{} {} Saved model to:{}".format(epoch, "\u2714", fold_model_path))

        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    accs[fold] = acc  # 作用 ：将当前fold的最佳验证准确率保存到数组中
    corrects[fold] = correct

    print("\r\n => Fold {} best val_loss {:.5f}, val_acc {:.5f}, epoch {}\n".format(fold, best_loss, acc, best_epo))


def evaluate():
    """
    该函数的主要作用是：
    1. 加载训练好的模型权重
    2. 在测试集上进行推理
    3. 计算各种性能指标
    4. 输出测试结果
    """
    import pandas as pd
    import os
    
    print("  Number of testing samples %d" % len(test_ind))
    print("  Start testing...")
    model.load_state_dict(torch.load(fold_model_path), strict=False)
    model.eval()
    outputs, _ = model(x, ph_features, affinity_graphs)
    outputs_test = outputs[test_ind].detach().cpu().numpy()


    # ============= 新增：提取特征用于t-SNE =============
    # 提取融合特征（推荐使用joint_embed）
    Z_joint = model.joint_embed.detach().cpu().numpy()  # shape: (N, 16)

     # 保存特征矩阵和标签
    tsne_dir = f"./tsne_features/{current_label.replace('输出_', '')}"
    os.makedirs(tsne_dir, exist_ok=True)

    # 保存所有样本的特征和标签
    np.save(os.path.join(tsne_dir, f"fold_{fold}_Z_joint.npy"), Z_joint)
    np.save(os.path.join(tsne_dir, f"fold_{fold}_y.npy"), y)

    # 保存测试集样本（用于后续可视化）
    # np.save(os.path.join(tsne_dir, f"fold_{fold}_Z_test.npy"), Z_joint[test_ind])
    # np.save(os.path.join(tsne_dir, f"fold_{fold}_y_test.npy"), y[test_ind])

    print(f"  t-SNE特征已保存到: {tsne_dir}")
    print(f"  特征矩阵形状: Z={Z_joint.shape}, y={y.shape}")


#   自己的
    # 获取测试集的被试ID
    test_subject_ids = dl.ids[test_ind]
    
    # 创建保存目录
    result_dir = f"./results/{current_label.replace('输出_', '')}"
    os.makedirs(result_dir, exist_ok=True)

     # 准备保存的数据
    results_data = {
        '被试ID': test_subject_ids,
        '真实标签': y[test_ind],
        '预测标签': np.argmax(outputs_test, 1),
        '预测正确': (y[test_ind] == np.argmax(outputs_test, 1)).astype(int)
    }
    # 创建DataFrame并保存
    df_results = pd.DataFrame(results_data)
    csv_file = os.path.join(result_dir, f"fold_{fold}_predictions.csv")
    df_results.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"  预测结果已保存到: {csv_file}")

# 自己的


    corrects[fold], accs[fold] = accuracy(outputs_test, y[test_ind])
    sens[fold], spes[fold], f1[fold] = metrics(outputs_test, y[test_ind])
    aucs[fold] = auc(outputs_test, y[test_ind])
    print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))


def print_all_results_summary(all_results):
    """
    打印所有标签的汇总结果
    """
    print("\n所有标签的性能对比:")
    print(f"{'标签':<15} {'准确率':<10} {'敏感性':<10} {'特异性':<10} {'AUC':<10} {'F1分数':<10}")
    print("-" * 70)
    
    for label, results in all_results.items():
        acc_mean = np.mean(results['accs'])
        sen_mean = np.mean(results['sens'])
        spe_mean = np.mean(results['spes'])
        auc_mean = np.mean(results['aucs'])
        f1_mean = np.mean(results['f1'])
        
        print(f"{label:<15} {acc_mean:.5f} {sen_mean:.5f} {spe_mean:.5f} {auc_mean:.5f} {f1_mean:.5f}")


def save_all_results_summary(all_results):
    """
    保存所有标签的汇总结果
    """
    import pandas as pd
    import os
    
    # 创建汇总数据
    summary_data = []
    for label, results in all_results.items():
        summary_data.append({
            '标签': label,
            '准确率_均值': np.mean(results['accs']),
            '准确率_标准差': np.std(results['accs']),
            '敏感性_均值': np.mean(results['sens']),
            '敏感性_标准差': np.std(results['sens']),
            '特异性_均值': np.mean(results['spes']),
            '特异性_标准差': np.std(results['spes']),
            'AUC_均值': np.mean(results['aucs']),
            'AUC_标准差': np.std(results['aucs']),
            'F1分数_均值': np.mean(results['f1']),
            'F1分数_标准差': np.std(results['f1'])
        })
    
    # 保存汇总结果
    result_dir = "./results/summary"
    os.makedirs(result_dir, exist_ok=True)
    
    df_summary = pd.DataFrame(summary_data)
    summary_file = os.path.join(result_dir, "all_labels_summary.csv")
    df_summary.to_csv(summary_file, index=False, encoding='utf-8')
    
    # 保存详细的JSON格式
    import json
    json_file = os.path.join(result_dir, "all_labels_detailed.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    print(f"\n汇总结果已保存到:")
    print(f"  CSV汇总: {summary_file}")
    print(f"  JSON详细: {json_file}")

    
if __name__ == "__main__":
 
    # 定义要循环处理的标签列表
    # label_list = ['输出_不稳定性心绞痛', '输出_冠状动脉粥样硬化心脏病', '输出_缺血性心肌病',
    #  '输出_左心室室壁瘤', '输出_冠状动脉支架植入后状态']   
    # label_list = ['输出_不稳定性心绞痛', '输出_冠状动脉粥样硬化心脏病', '输出_缺血性心肌病',
    #  '输出_冠状动脉支架植入后状态', '输出_高血压','输出_糖尿病'] 
    # label_list = ['输出_不稳定性心绞痛', '输出_冠状动脉粥样硬化心脏病', '输出_缺血性心肌病',
    #  '输出_冠状动脉支架植入后状态'] 
    label_list = ['输出_冠状动脉粥样硬化心脏病']
    # label_list = ['DX_GROUP']
      # 存储所有标签的结果
    all_results = {} 

       # 为每个标签创建循环
    for current_label in label_list:    

        # settings = OptInit(model="CSDA-PGNet", dataset="ABIDE", atlas="aal")
        # fold  data_folder id.txt顺序和你编号中顺序一致就行得改   --fold 6  --data_folder  data/TY
        settings = OptInit(model="CSDA-PGNet", dataset="TY", atlas="ty")



        # modify existing parameters
        # settings.args.train = False
        # settings.args.img_depth = 2
        # settings.args.ph_depth = 3
        # settings.args.node_dim = 2500
        # settings.args.pool_ratios = 0.1
        # settings.args.scores = [settings.args.sites]
        # settings.args.smh = 1e-1
        # 作用 : 设置计算设备，优先使用GPU 输入 : 检查CUDA可用性 输出 : 设备对象（cuda:0或cpu）
        # 这里实际上是又设置了一次，所以以后改的话就用这里改
        settings.args.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        print(f'settings.args.device:{settings.args.device}')

         # 关键步骤：动态修改标签
        settings.args.labels = current_label

       


        # initialize parser
        # 作用 : 初始化配置并打印所有参数  opt 变量接收返回的 self.args 对象 这个对象包含了所有模型训练和数据处理所需的配置参数
        opt = settings.initialize()
        # 是一个用于打印所有配置参数的实用函数。
        settings.print_args()

        # 输入 这是一个配置对象，来自 `opt.py` 中的 OptInit 类实例，包含了所有模型和数据处理的配置参数
        # 输出 dl ： `mydataloader.py` 中 MyDataloader 类的实例对象
        # 作用: 这行代码创建了一个数据加载器对象，用于处理 CSDA-PGNet 模型所需的多模态数据。
        dl = MyDataloader(opt)

    # 用于加载多模态数据，包括功能连接特征、标签和表型数据。 
    # save=False ：布尔值参数，控制是否保存加载的数据到磁盘  True ：加载后将数据保存为 .pt 文件
    # return self.features, self.labels, self.ph_dict, self.ph_data
        x, y, ph_dict, ph_data = dl.load_data(save=False)
        print(f'x:{x},x.shape:{x.shape}')
        print(f'y:{y},y.shape:{y.shape}')
        # print(f'ph_dict:{ph_dict},ph_dict.shape:{ph_dict.shape}')
        print(f'ph_data:{ph_data},ph_data.shape:{ph_data.shape}')

    # - torch.tensor(y, dtype=torch.long) : 将NumPy数组转换为PyTorch长整型张量
    # - .to(opt.device) : 将张量移动到指定计算设备
    #  这行代码的主要功能是 将NumPy数组转换为PyTorch张量并设置为浮点数类型 ，用于深度学习模型的表型数据处理。
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        # （NumPy数组格式）转换为PyTorch张量
        ph_features = torch.from_numpy(ph_data).float()

        # k-fold cross validation
        n_folds = opt.folds
        cv_splits = dl.data_split(n_folds, val_ratio=0.1)
        corrects = np.zeros(n_folds, dtype=np.int32)
        accs = np.zeros(n_folds, dtype=np.float32)
        sens = np.zeros(n_folds, dtype=np.float32)
        spes = np.zeros(n_folds, dtype=np.float32)
        f1 = np.zeros(n_folds, dtype=np.float32)
        aucs = np.zeros(n_folds, dtype=np.float32)
        times = np.zeros(n_folds, dtype=np.float32)

        for fold in range(n_folds):
            print("\r\n========================== Fold {} ==========================".format(fold))
            train_ind = cv_splits[fold][0] # 训练集索引
            val_ind = cv_splits[fold][1]  # 验证集索引  
            test_ind = cv_splits[fold][2]   # 测试集索引

            # feature_selection
            x = feature_selection(x, y, train_ind, opt.node_dim)

    # torch.from_numpy() 函数用于将 NumPy 数组转换为 PyTorch 张量
            affinity_graphs = torch.from_numpy(
                create_reward_penalty_graph(ph_dict, y, train_ind, val_ind, test_ind, opt)).float()
            model = CSDAPGNet(opt, fold).to(opt.device)
  

            print(model)

            # record training time  ### 输出
    # - 返回一个 torch.cuda.Event 对象，可以用于记录GPU操作的时间点
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

    # torch.nn.CrossEntropyLoss() 就是 PyTorch 里用于“多分类单标签”任务的标准损失函数
    # 自动对模型输出应用softmax函数，将logits转换为概率分布
            loss_fn = torch.nn.CrossEntropyLoss()
            """
            1. model.parameters()

            - 类型 : 生成器对象
            - 功能 : 返回模型中所有可训练参数的迭代器
            - 内容 : 包括权重矩阵、偏置向量等所有需要梯度更新的参数
            2. lr=opt.lr

            - 参数名 : 学习率（learning rate）
            - 默认值 : 根据 `opt.py` 配置，默认为 1e-4
            - 功能 : 控制参数更新的步长大小
            - 影响 : 学习率过大可能导致训练不稳定，过小则收敛缓慢
            3. weight_decay=opt.wd

            - 参数名 : 权重衰减（L2正则化）
            - 默认值 : 根据配置，默认为 5e-4
            - 功能 : 防止过拟合，通过在损失函数中添加权重的L2范数惩罚项
            - 计算 : loss = original_loss + weight_decay * ||weights||²
            ### 输出
            - 返回值 : torch.optim.Adam 对象
            - 功能 : 可用于执行参数更新的优化器实例
            """
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

            early_stopping = EarlyStopping(patience=opt.early_stop, verbose=True)
            # 这行代码用于构建每个fold（折）的模型保存路径，是K折交叉验证中模型检查点文件的路径生成。
            # opt.ckpt_path : 模型检查点的基础保存目录 默认值为 ./save_model/{model}_{dataset}_{atlas}/
            fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)
            # 作用 ：检查配置参数中是否启用了日志保存功能 默认值为 False opt.log_save
            if opt.log_save:
                # ：创建TensorBoard日志写入器，用于记录训练过程中的各种指标
                writer = SummaryWriter(f"./log/{opt.model}_{opt.dataset}_{opt.atlas}_log/{fold}")
            if opt.train == 1:
                # Recording start time # 记录开始时间
                start.record()
                # pretraining vae  表型特征

                # model.train_vae(ph_features)
                # 用多卡
                model.train_vae(ph_features)

                # training model
                train()
                # Recording end time # 记录结束时间
                end.record()
                # Synchronizing GPU and CPU
                torch.cuda.synchronize()
                time = start.elapsed_time(end) / 1000
                print(f"Training time of {fold}-fold data: {time} s\n")
                times[fold] = time
                # evaluating model
                evaluate()
            elif opt.train == 0:
                evaluate()
                print(f"img_weight: {model.img_weight}")
                print(f"ph_weight: {model.ph_weight}")
                print(f"scores_weight: {opt.scores}:{model.rp_attention.weights.data}")
        # 保存当前标签的结果
        current_results = {
            'label': current_label,
            'accs': accs.copy(),
            'sens': sens.copy(),
            'spes': spes.copy(),
            'aucs': aucs.copy(),
            'f1': f1.copy(),
            'times': times.copy() if opt.train == 1 else None
        }
        all_results[current_label] = current_results

        print("\r\n========================== Finish ==========================")
        if opt.train == 1:
            print("=> Average training time in {}-fold CV: {:.5f} s".format(n_folds, np.mean(times)))
        print_result(opt, n_folds, accs, sens, spes, aucs, f1,current_label)
        save_result(opt, n_folds, accs, sens, spes, aucs, f1,current_label)
        # 打印所有标签的汇总结果
    print("\r\n========================== 所有标签汇总结果 ==========================")
    print_all_results_summary(all_results)
    save_all_results_summary(all_results)

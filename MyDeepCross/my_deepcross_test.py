import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from Utils.tool import auc, acc, accuracy_score, paint, paint_curve, F1, roc, test_patho
from DeepCross.trainer import Trainer
from DeepCross.network import DeepCross
# from Utils.criteo_loader import getTestData, getTrainData
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import pandas as pd

deepcross_config = \
    {
        'deep_layers': [512, 256, 128, 64],  # 设置Deep模块的隐层大小
        'num_cross_layers': 4,  # cross模块的层数
        'num_epoch': 700,
        'batch_size': 64,
        'lr': 0.00001,
        'l2_regularization': 0.0001,
        'device_id': 0,
        'use_cuda': True,
        'nonpatho_path': r'D:\desktop\randomForest\data\chrom_freq1500.csv',
        'patho_path': r'D:\desktop\randomForest\data\plas_freq1500.csv',
        'test_size': 0.01
    }


def DataPreprocess():
    # 只考虑稠密的
    # import data
    # patho = pd.read_csv(r'D:\desktop\randomForest\data\k=8_patho_freq.csv')
    # nonpatho = pd.read_csv(r'D:\desktop\randomForest\data\k=8_non_freq.csv')
    patho = pd.read_csv(deepcross_config['patho_path'])
    names = []
    for i in range(patho.shape[1]):
        names.append('F' + str(i))
    patho.columns = names
    nonpatho = pd.read_csv(deepcross_config['nonpatho_path'])
    nonpatho.columns = names
    print('read csv over')

    train_df = pd.concat((patho, nonpatho))
    print(train_df.shape)
    labels = np.concatenate((np.ones(patho.shape[0]), np.zeros(nonpatho.shape[0])))
    # labels = np.concatenate((np.ones(patho.shape[0]), np.zeros(nonpatho.shape[0])))

    print(len(labels))
    scaler = StandardScaler().fit(train_df)
    # np.savetxt(r'D:\bio4\pathtest20\2\data2\scaler.csv', scaler, delimiter=",")  # 不能正常十进制储存

    # 这一步再用scaler中的均值和方差来转换data，使data数据标准化
    scaled = scaler.transform(train_df)
    X = scaled

    # 进行数据合并，为了同时对train和test数据进行预处理
    # data_df = pd.concat((train_df, test_df))

    # del data_df['Id']
    #
    # print(data_df.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=deepcross_config['test_size'], random_state=414)
    print(y_train[0:5])
    return train_df, X_train, X_test, y_train, y_test


def getTrainData(df):
    # df = pd.read_csv(filename)
    print(df.columns)

    # C开头的列代表稀疏特征，I开头的列代表的是稠密特征
    dense_features_col = [col for col in df.columns]

    # 这个文件里面存储了稀疏特征的最大范围，用于设置Embedding的输入维度
    # fea_col = np.load(feafile, allow_pickle=True)
    # sparse_features_col = []
    # for f in fea_col[1]:
    #     sparse_features_col.append(f['feat_num'])
    #
    # data, labels = df.drop(columns='Label').values, df['Label'].values

    return dense_features_col


def eval_model(y_test, y_pred, test_acc, test_roc, test_f1):
    # if config.use_cuda:
    #     deepCrossing.loadModel(map_location=lambda storage, loc: storage.cuda(deepcrossing_config['device_id']))

    # y_pred_list.append(y_pred)
    # print("Test Data CTR Predict...\n ", y_pred.view(-1))
    # torch.Tensor.cpu(y_test, y_pred)
    accuracy = acc(y_test, y_pred, 'test')
    roc_score = roc(y_test, y_pred)
    print('roc_score', roc_score)
    f1_ = F1(y_test, y_pred)
    test_acc.append(accuracy)

    test_roc.append(roc_score)

    test_f1.append(f1_)

    # return test_acc, test_roc, test_f1


if __name__ == "__main__":
    ####################################################################################
    # DeepCross 模型
    ####################################################################################
    test_acc = []
    test_roc = []
    test_f1 = []
    y_pred_list = []
    train_df, X_train, X_test, y_train, y_test = DataPreprocess()
    # X_train = X_train.to_numpy()
    # X_test = X_test.to_numpy()
    dense_features_cols = getTrainData(train_df)

    train_dataset = Data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    # training_data, training_label, dense_features_col, sparse_features_col = getTrainData(deepcross_config['train_file'], deepcross_config['fea_file'])
    # train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())
    # test_data = getTestData(deepcross_config['test_file'])
    # test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    deepCross = DeepCross(deepcross_config, dense_features_cols)
    deepCross = deepCross.cuda()

    print(deepCross)

    # summary = SummaryWriter('../TrainedModels' + time.strftime("%Y-%m-%d", time.localtime()))

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=deepCross, config=deepcross_config)
    trainer.use_cuda()
    test_data, y_test = torch.tensor(X_test).cuda(), torch.tensor(y_test).cuda()
    # 训练
    # trainer.train(train_dataset)
    train_loss = []
    for epoch in range(deepcross_config['num_epoch']):
        print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
        data_loader = DataLoader(dataset=train_dataset, batch_size=deepcross_config['batch_size'], shuffle=True)
        # 训练一个轮次
        # self._train_an_epoch(data_loader, epoch_id=epoch)
        trainer._train_an_epoch(data_loader, epoch, train_loss)

        # 保存模型
        # trainer.save()

        ####################################################################################
        # 模型测试阶段
        ####################################################################################
    #     deepCross.eval()
    #     y_pred_probs = deepCross(torch.tensor(test_data).float())
    #     y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    #     eval_model(y_test, y_pred, test_acc, test_roc, test_f1)
    #
    # paint(train_loss, test_acc, test_roc, test_f1)
    # # print(y_pred_list[-1])
    # paint_curve(y_test, y_pred_probs)

    #### test patho ####
    deepCross.eval()
    test_patho(deepCross)
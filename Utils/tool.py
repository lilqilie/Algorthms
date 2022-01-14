from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

from torch.utils.data import DataLoader
def testDataPreprocess():
    test_path = r'D:\bio4\nonsplit LOO\freq_patho.csv'
    # 只考虑稠密的
    # import data
    # patho = pd.read_csv(r'D:\desktop\randomForest\data\k=8_patho_freq.csv')
    # nonpatho = pd.read_csv(r'D:\desktop\randomForest\data\k=8_non_freq.csv')
    patho = pd.read_csv(test_path)
    names = []
    for i in range(patho.shape[1]):
        names.append('F' + str(i))
    patho.columns = names
    # nonpatho = pd.read_csv(config.nonpatho_path)
    # nonpatho.columns = names
    print('read csv over')

    # train_df = pd.concat((patho, nonpatho))
    print(patho.shape)
    # labels = np.concatenate((np.zeros(patho.shape[0]), np.ones(nonpatho.shape[0])))
    labels = (np.ones(patho.shape[0]))

    print(len(labels))
    scaler = StandardScaler().fit(patho)
    # # np.savetxt(r'D:\bio4\pathtest20\2\data2\scaler.csv', scaler, delimiter=",")  # 不能正常十进制储存
    #
    # scaler.transform(data)
    # 这一步再用scaler中的均值和方差来转换data，使data数据标准化
    scaled = scaler.transform(patho)
    X = scaled
    df_X = patho
    # 进行数据合并，为了同时对train和test数据进行预处理
    # data_df = pd.concat((train_df, test_df))

    # del data_df['Id']
    #
    # print(data_df.columns)
    # X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=414)
    # print(y_train[0:5])
    return df_X, X, labels

def test_patho(Model):
    accuracy_ = []
    train_loss = []
    train_acc = []
    train_auc = []
    train_f1 = []
    test_acc = []
    test_roc = []
    test_f1 = []
    df_X, X, labels = testDataPreprocess()
    # dense_features_cols = getTrainData(df_X)
    # Model.eval()

        #     deepCrossing.loadModel(map_location=lambda storage, loc: storage.cuda(deepcrossing_config['device_id']))

    test_data, y_test = torch.tensor(X).cuda(), torch.tensor(labels).cuda()
    # else:
    #     deepCrossing.loadModel(map_location=torch.device('cpu'))
    # if self._config['use_cuda'] is True:

    y_pred_probs = Model(torch.tensor(test_data).float())
    print(y_pred_probs)
    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    # print("Test Data CTR Predict...\n ", y_pred.view(-1))
    # torch.Tensor.cpu(y_test, y_pred)
    accuracy = acc(y_test, y_pred, 'test')
    # roc_score = roc(y_test, y_pred)
    # print('roc_score', roc_score)
    f1_ = F1(y_test, y_pred)
    paint_curve(y_test, y_pred_probs)



def paint(train_loss, test_acc, test_roc, test_f1):
    # acc = history.history['accuracy']  # 训练集准确率
    # val_acc = history.history['val_accuracy']  # 测试集准确率
    # loss = history.history['loss']  # 训练集损失
    # val_loss = history.history['val_loss']  # 测试集损失
    #  打印acc和loss，采用一个图进行显示。
    #  将acc打印出来。

    plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)  # 将图像分为一行两列，将其显示在第一列
    # # plt.plot(train_acc, label='Training Accuracy')
    # plt.plot(test_acc, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    plt.subplot(1, 2, 1)  # 将其显示在第二列
    plt.plot(train_loss, label='Training Loss')
    # plt.plot(test_loss, label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)  # 将其显示在第三列
    plt.plot(test_roc, label='Roc-auc score', color='black', linestyle='dotted')
    plt.plot(test_f1, label='F1 score')
    plt.plot(test_acc, label='Accuracy')

    # plt.plot(test_loss, label='Validation Loss')
    plt.title('Validation eval')
    plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    plt.legend()
    plt.show()


def roc(y_ture, y_pred):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()
    roc = roc_auc_score(y_test, y_pred)
    return roc


def F1(y_ture, y_pred):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()
    f1 = f1_score(y_test, y_pred)
    print('test f1 score: %f' % f1)

    return f1


def acc(y_ture, y_pred, mood):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()

    acc = accuracy_score(y_test, y_pred)
    print('%s accuracy: %f' % (mood, acc))
    return acc


def paint_curve(y_test, y_pred):
    y_test = y_test.cpu()
    # y_pred = y_pred.cpu()
    y_pred = y_pred.cpu().detach().numpy()
    # auc
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Validation ROC Curve')
    plt.plot(fpr, tpr, 'b', label='DEEP CROSS(area=%0.3f)' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()

    # precision and recall
    plt.subplot(1, 2, 2)
    plt.title('Validation Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')

    # y_true和y_scores分别是gt label和predict score

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    area = auc(recall, precision)

    plt.plot(precision, recall, 'g', label='DEEP CROSS(area=%0.3f)' % area)
    plt.legend()
    plt.show()
    # plt.savefig('p-r.png')


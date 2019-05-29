import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def describe_missing(df, include=None, exclude=None):
    """
        用来描述datafame的Object类型与非object类型数据的基本情况与缺失率
        df: dataframe-like，要描述的数据
        include: list-like，选择的数据类型
        exclude: list-like，排除的数据类型
    """
    return df.select_dtypes(include=include, exclude=exclude).describe().T \
        .assign(missing_pct=df.apply(lambda x: (len(x) - x.count()) / len(x)))


def plot_pie(data_series, title=None, startangle=0, pctdistance=0.7,
             labeldistance=1.5, autopct='%1.0f%%', font_size='medium',
             title_font_size=20, explode=None, shadow=True):
    """
    参数说明：
    data_series: pd.Series类型,包含数据和index
    title：图片名称
    startangle:开始的角度，3点钟的位置为0，逆时针，12点的位置为90°
    pctdistance:比例数据的位置，1代表在圆的边上，0代表在圆心上
    labeldistance:label的位置
    autopct:
    font_size:字体的大小
    explode:是否分裂，是一个数组类型的，元素个数与index个数相同
    """
    from matplotlib import font_manager as fm
    # 设置绘图区域大小
    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
    # 把两个图割开,方便后边分别设置
    ax1, ax2 = axes.ravel()

    patches, texts, autotexts = ax1.pie(data_series.values, labels=data_series.index, autopct=autopct,
                                        shadow=shadow, startangle=startangle, pctdistance=pctdistance, explode=explode)

    ax1.axis('equal')
    # 重新设置字体大小
    proptease = fm.FontProperties()
    proptease.set_size(font_size)
    # font size include: ‘xx-small’,x-small’,'small’,'medium’,‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)

    ax1.set_title(title, loc='center', fontsize=title_font_size)

    # ax2 只显示图例（legend）
    ax2.axis('off')
    ax2.legend(patches, data_series.index, loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_learning_curve(estimator, X, y, title='学习曲线', ylim=None, cv=None, n_jobs=-1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    train_size:X 中用来作样本的比例，如果是0.1，则0.1*X用来作全部样本（包括训练集和测试集）
    cv：用于交叉验证的参数，cv=5代表把X随机分为5份，其中四个作为训练集，一个作为测试集
    """
    """
        画出data在某模型上的learning curve.
        参数解释
        ----------
        estimator : 你用的分类器。
        title : 表格的标题。
        X : 输入的feature，numpy类型
        y : 输入的target vector
        ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
        cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
        n_jobs : 并行的的任务数(默认1)
        """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def plot_confusion_matrix(y_test, y_predict, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_test, y_predict)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def print_recall_precision_f1(y_true, y_pred):
    print('precision score: %.4f' % precision_score(y_true, y_pred))
    print('recall score: %.4f' % recall_score(y_true, y_pred))
    print('f1 score: %.4f' % f1_score(y_true, y_pred))


def print_roc_auc_score(y_test, y_score):
    print('AUC score:', roc_auc_score(y_test, y_score[:, 1]))


def plot_auc_curve(y_test, y_score):
    """
    绘制auc曲线
    :param y_test:
    :param y_score:
    :return:
    """
    # fpr,tpr,thresholds 分别为假正率、真正率和阈值
    tpr, fpr, thresholds = roc_curve(y_test, y_score[:, 0])
    # 计算auc的值
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

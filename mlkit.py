import itertools
import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
import seaborn as sns
# metrics
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
# stats
from scipy.stats import skew, norm


# from scipy.special import boxcox1p
# from scipy.stats import boxcox_normmax


def describe_missing(df, include=None, exclude=None, missing_only=True,
                     sort=True, ascending=False):
    """
    用来描述DataFrame的Object类型与非object类型数据的基本情况与缺失率
    :param df: DataFrame-like，要描述的数据
    :param include: list-like or str 选择的数据类型,like: float,int
            numerical features: 'number' or np.number
            categorical features: 'object' or 'O'
    :param exclude: list-like，排除的数据类型: object
    :param missing_only: only return missing percent > 0 columns description
    :param sort: 是否根据缺失率排序，默认是根据missing_pct降序排列
    :param ascending: 默认为降序
    :return: df
    """

    df = df.select_dtypes(include=include, exclude=exclude).describe().T \
        .assign(missing_pct=df.apply(lambda x: (len(x) - x.count()) / len(x)))

    if missing_only:
        df = df[df['missing_pct'] > 0]

    if sort:
        return df.sort_values(by='missing_pct', ascending=ascending)
    else:
        return df


def specific_dtype_column_names(df, dtype):
    """
    获取指定类型的列名
    :param df:
    :param dtype: 'number','object','bool'
    :return:
    """
    return df.select_dtypes(include=dtype).columns.values


def plot_univariate_dist(df, col_name):
    """
    plot univariate distribution for regression task, to see the numeric target column distribution.
    :param df: DataFrame
    :param col_name: column name, str
    """
    sns.set_style('white')
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(df[col_name], color='b')
    ax.xaxis.grid(False)
    ax.set(ylabel='Frequecy')
    ax.set(xlabel=col_name)
    ax.set(title=f'{col_name} distribution')
    sns.despine(trim=True, left=True)
    (mu, sigma) = norm.fit(df[col_name])
    # print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu,sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(
        mu, sigma)], loc='best')
    plt.show()


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
    explode:是否分裂，是一个数组类型的，元素个数与index个数相同,比如一个一个Series有四个数据，
            [0.05,0,0,0]表示第一个数分裂的间距大小为0.05,其它的三个不分裂。
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


def plot_learning_curve(estimator, X, y, title='Learning Curve', ylim=((1.01, 0.7)), cv=5, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymax, ymin), optional
        Defines maximum and minimum yvalues plotated.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        使用n_jobs=-1的时候，有问题无法得到结果，使用默认的n_jobs=-1

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.gca().invert_yaxis()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# def plot_confusion_matrix(y_test, y_predict, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     cm = confusion_matrix(y_test, y_predict)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()


def print_recall_precision_f1(y_true, y_pred, pos_label=1):
    """
    二分类：默认是计算1类别的recall，precision，f1值
    pos_label：指定要计算的分类
    """
    print('precision score: %.4f' % precision_score(y_true, y_pred, pos_label=pos_label))
    print('recall score: %.4f' % recall_score(y_true, y_pred, pos_label=pos_label))
    print('f1 score: %.4f' % f1_score(y_true, y_pred, pos_label=pos_label))


def print_roc_auc_score(y_test, y_score):
    print('AUC score: %.4f' % roc_auc_score(y_test, y_score[:, 1]))


def plot_auc_curve(y_test, y_score):
    """
    绘制auc曲线
    :param y_test:二分类中的真实值，比如： [1,0,...,0,0]，shape=[n_samples]
    :param y_score: 预测分类的概率，有的classifier有predict_pro方法，得到是n_samples * n_classes的矩阵,第一列是预测为0的概率，
    第二列是预测为1的概率。所以取了第二列
    :return:
    """
    # fpr,tpr,thresholds 分别为假正率、真正率和阈值
    if y_score.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
    else:
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
    # 计算auc的值,面积
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_gridsearch_cv(results, param_name, scoring, x_min, x_max, y_min, y_max, save=False, saves='MyFigure.png'):
    """
    绘制单因素gridsearch cv的结果
    result: gridsearch.cv_results_
    param_name: 要搜索的参数
    x_min: x轴最小值
    x_max: x轴最大值
    y_min: y轴最小值
    y_max: y轴最大值
    scoring: dict,scorer可以指定正类的，以-1作为positive class,gridsearch也可以有多个评分标准
             比如：scoring = {'AUC':'roc_auc','Recall':make_scorer(recall_score,pos_label=-1)}
    """
    plt.figure(figsize=(10, 8))
    plt.title('GridSearchCV for ' + param_name, fontsize=24)

    plt.xlabel(param_name, fontsize=14)
    plt.ylabel("score", fontsize=14)
    plt.grid()

    ax = plt.axes()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    pad = 0.005
    X_axis = np.array(results["param_" + param_name].data, dtype=float)

    # ('train','--')
    sample = 'test'
    style = '--'
    for scorer, color in zip(sorted(scoring), ['b', 'k']):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]

        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label='%s (%s)' % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index]] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        ax.annotate('%0.4f' % best_score,
                    (X_axis[best_index], best_score + pad))

    plt.legend(loc='best')
    plt.grid('off')
    plt.tight_layout()

    if save:
        plt.savefig(saves, dpi=100)

    plt.show()


def print_clustering_scores(labels_true, labels_pred):
    """
    以下scores都在[0,1]之间，越大越好
    """
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_pred))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels_pred))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels_pred,
                                               average_method='arithmetic'))
    print("Fowlkes and Mallows Index: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels_pred))


def print_silhoutte_scores(X, labels_pred):
    """
    打印轮廓系数，完全是非监督的方式
    Silhouetee coefficient在[-1,1]，越大越好
    DB score: 
    """
    print('Silhouette coefficient:{:.4f}'.format(metrics.silhouette_score(X, labels_pred, metric='euclidean')))
    print('Davies Bouldin score:{:.4f}'.format(metrics.davies_bouldin_score(X, labels_pred)))
    print('Calinski Harabasez score:{:.4f}'.format(metrics.calinski_harabasz_score(X, labels_pred)))

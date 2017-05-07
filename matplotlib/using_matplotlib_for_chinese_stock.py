import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_ohlc
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold
import datetime

# 用来正常显示中文标签，此方法对我无用
# plt.rcParams['font.sans-serif'] = ['SimHei']
# 需下载字体，放入/usr/share/fonts
# 修改配置文件matplotlibrc，路径可由matplotlib.matplotlib_fname()查看
# 最后删除/Users/jiajing/.matplotlib／fontList.py3k.cache 

# 开始时间
start = datetime.datetime(2016, 1, 1)
# 结束时间
end = datetime.date.today()

# 上证50成分股
symbol_dict = {
    "600000": "浦发银行",
    "600010": "包钢股份",
    "600015": "华夏银行",
    "600016": "民生银行",
    "600018": "上港集团",
    "600028": "中国石化",
    "600030": "中信证券",
    "600036": "招商银行",
    "600048": "保利地产",
    "600050": "中国联通",
    "600089": "特变电工",
    "600104": "上汽集团",
    "600109": "国金证券",
    "600111": "北方稀土",
    "600150": "中国船舶",
    "600256": "广汇能源",
    "600406": "国电南瑞",
    "600518": "康美药业",
    "600519": "贵州茅台",
    "600583": "海油工程",
    "600585": "海螺水泥",
    "600637": "东方明珠",
    "600690": "青岛海尔",
    "600837": "海通证券",
    "600887": "伊利股份",
    "600893": "中航动力",
    "600958": "东方证券",
    "600999": "招商证券",
    "601006": "大秦铁路",
    "601088": "中国神华",
    "601166": "兴业银行",
    "601169": "北京银行",
    "601186": "中国铁建",
    "601288": "农业银行",
    "601318": "中国平安",
    "601328": "交通银行",
    "601390": "中国中铁",
    "601398": "工商银行",
    "601601": "中国太保",
    "601628": "中国人寿",
    "601668": "中国建筑",
    "601688": "华泰证券",
    "601766": "中国中车",
    "601800": "中国交建",
    "601818": "光大银行",
    "601857": "中国石油",
    "601901": "方正证券",
    "601988": "中国银行",
    "601989": "中国重工",
    "601998": "中信银行"}

symbols, names = np.array(list(symbol_dict.items())).T
# 读太多股票程序会卡住，这里简单读20只股票，作为例子
simple_symbols = symbols[:20]

quotes = [quotes_historical_yahoo_ohlc(symbol+".ss", start, end, asobject=True) for symbol in simple_symbols]

# 用astype转换数组元素的类型，open_price.dtype可查看元素类型
# 用q.open取某股所有开盘价，也可用q['open']
open_price = np.array([q.open for q in quotes]).astype(np.float)
close_price = np.array([q.close for q in quotes]).astype(np.float)
# 计算每日价格浮动，一行是一只股票
variation = close_price - open_price

# 指定一个模型
edge_model = covariance.GraphLassoCV()
# 标准化
# covariance（协方差），两个序列与其均值之差的乘积、再求和、再取平均，用来表明两个序列是同向变化（值为正，一个序列中变大，另一个也变大），还是反向变化（值为负）
# correlation（相关系数），取值在-1和+1之间，取0表示不相关，取-1表示负相关，取+1表示正相关。
# 相关系数=协方差/两个序列标准差之积
# 矩阵转置.T，一列是一只股票
X = variation.copy().T
# axis=0，列，用来计算数据X每个特征（每只股票）的方差
X /= X.std(axis=0)
# 训练数据
edge_model.fit(X)

# 使用Affinity Propagation算法
_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

# 打印聚类结果
for i in range(n_labels + 1):
	# 同一类的打印出来
	print('Cluster %i: %s' % ((i + 1), ','.join(names[labels == i])))

# 指定一个模型，用来找最佳画点位置
# LocallyLinearEmbedding是一种非线性降维算法
# n_components : number of coordinates for the manifold.
# n_neighbors : number of neighbors to consider for each point.
node_position_model = manifold.LocallyLinearEmbedding(n_components=2, eigen_solver='dense', n_neighbors=6)
# 训练模型，降维得到2*20的array，即每只股的X, Y轴坐标
embedding = node_position_model.fit_transform(X.T).T

plt.figure(1, facecolor='w', figsize=(7, 5))
# 清空画布
plt.clf()
# axes相当于子图，axis指坐标
# A given figure can contain many Axes, but a given Axes object can only be in one Figure.
# The Axes contains two (or three in the case of 3D) Axis objects (be aware of the difference between Axes and Axis)
ax = plt.axes([0., 0., 1., 1.])
# 关掉坐标
plt.axis('off')

# 20*20的偏相关矩阵
partial_correlations = edge_model.precision_.copy()
# np.diag返回矩阵对角线上的元素
# np.sqrt对矩阵中各元素开方
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
# np.newaxis的功能是在行（放前面）或列（放后面）上增加维度
# arr.shape可用来查看维度
partial_correlations *= d[:, np.newaxis]
# triu函数返回的是矩阵的上三角矩阵，tril返回下三角
# Return a copy of a matrix with the elements below the k-th diagonal zeroed.
# k = 0 (the default) is the main diagonal, k < 0 is below it and k > 0 is above.
# 根据相关性，设置显示阈值，返回20*20的bool型矩阵
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Marker size is scaled by 参数s
# 参数c can be a sequence of N numbers to be mapped to colors using the 参数cmap
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)
# numpy.where(condition[, x, y])
# If both x and y are specified, the output array contains elements of x where condition is True, and elements from y elsewhere.
# If only condition is given, return the tuple condition.nonzero(), the indices where condition is True.
# 返回值为True的坐标，start_idx为连线起点股票index, end_idx为连线终点index
start_idx, end_idx = np.where(non_zero)

# 每行为一条连线起终点两只股票的坐标
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
# 每一条连线的相关性权重值
values = np.abs(partial_correlations[non_zero])
# 参数zorder，控制绘图顺序，为0相当于放在底层
# 参数norm, scales data, typically into the interval [0, 1]. 
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    # np.argmin返回的是最小值的索引
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    # 调整位置，与最小间距的点错开，设置text的错开位置
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    # bbox用来给标签加个框，facecolor是版面色，edgecolor是边框色，alpha是透明度，
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

# ptp():peak to peak. Range of values (maximum - minimum) along an axis.
plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.title(u'上证50成分股')
plt.show()













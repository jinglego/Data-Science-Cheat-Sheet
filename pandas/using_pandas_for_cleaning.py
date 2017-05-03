import numpy as np
import pandas as pd
# 得到level, number两列数据
data1 = pd.DataFrame({'level': ['a', 'b', 'c', 'd'], 'number1': [1, 3, 5, 7]})
data2 = pd.DataFrame({'level': ['a', 'b', 'c', 'e'], 'number2': [2, 3, 6, 10]})

# 相同列中相同的行才保留，不同的列也保留，相当于inner join
print(pd.merge(data1, data2))
# 相当于left outer join
print(pd.merge(data1, data2, how='left'))
# 相当于right outer join
print(pd.merge(data1, data2, how='right'))
# 相当于full outer join
print(pd.merge(data1, data2, how='outer'))

# 针对列名不同的合并，可用left_on, right_on来指定，两列都会保留
data3 = pd.DataFrame({'level1': ['a', 'b', 'c', 'd'], 'number1': [1, 3, 5, 7]})
data4 = pd.DataFrame({'level2': ['a', 'b', 'c', 'e'], 'number2': [2, 3, 6, 10]})
print(pd.merge(data3, data4, left_on='level1', right_on='level2'))

# 对相同的列，左边有值的显示左边的，左边没值的用右边对应的补全
data5 = pd.DataFrame({'level': ['a', 'b', 'c', 'd'], 'number': [1, 3, 5, np.nan]})
data6 = pd.DataFrame({'level': ['a', 'b', 'c', 'e'], 'number': [2, np.nan, 6, 10]})
print(data5.combine_first(data6))

data7 = pd.DataFrame(np.arange(12).reshape(3,4),
              columns=['a','b','c','d'],
              index=['wang','li','zhang'])
# 将DataFrame的列转化成行，原来的列索引成为行的层次索引
print(data7.stack())
# 将DataFrame的列索引转化成行索引，原来的行索引成为行的层次索引，与stack互逆
print(data7.unstack())

data8 = pd.DataFrame({'a': [1, 3, 3, 4], 'b': [1, 3, 3, 5]})
# 判断该行与前一行是否重复
print(data8.duplicated())
# 删除重读的行
print(data8.drop_duplicates())

data9 = pd.DataFrame({'a': [1, 3, 3, 4], 'b': [1, 3, 3, 5]})
# 一次性替换某个范围的数据为某值
print(data9.replace([1, 4], np.nan))

data10 = [11, 15, 18, 20, 25, 26, 27, 24]
bins = [15, 20, 25]
# 数据分段（箱），判断在给定bins中哪个范围内，在分段内的显示分段如(15, 20]，不在分段内的数据显示为nan值
print(pd.cut(data10, bins))
# 显示为各个分段的标签名labels
print(pd.cut(data10, bins, labels=["small", "big"]))
# 显示所在分段排序标签，如-1, 0, 1
print(pd.cut(data10, bins).codes)
# 显示每个分段中值的个数
print(pd.cut(data10, bins).value_counts())

data11 = pd.DataFrame(np.arange(12).reshape(4,3))
# 随机生成要采样的行号，有顺序，例如[2, 0, 1]
samp=np.random.permutation(3)
# 按照顺序采样
print(data11.take(samp))



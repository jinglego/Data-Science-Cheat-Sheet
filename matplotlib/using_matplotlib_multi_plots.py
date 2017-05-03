import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

df = pd.read_excel("/Users/jiajing/My Data Analysis/sample-salesv3.xlsx")
df.head()

# groupby: Splitting the data into groups based on some criteria. Applying a function to each group independently. Combining the results into a data structure.
# agg: 聚合操作，包括np.sum, np.mean, np.std等，也可用字符串作为参数。传入列表，可对不同列应用不同的聚合函数分别进行操作。
# sort_values: 对DataFrame进行排序，by表示用哪一列，ascending为False表示降序
# reset_index: 对DataFrame重设index
top_10 = (df.groupby('name')['ext price', 'quantity'].agg({'ext price': 'sum', 'quantity': 'count'})
          .sort_values(by='ext price', ascending=False))[:10].reset_index()
# rename: 对DataFrame的列名进行重命名，inplace表示在原有实例上进行操作
top_10.rename(columns={'name': 'Name', 'ext price': 'Sales', 'quantity': 'Purchases'}, inplace=True)
# 用plt.style.available查看可选择的表格风格
plt.style.use('ggplot')
# kind="bar"表垂直柱状图、kind="barh"表水平柱状图
# top_10.plot(kind='barh', y="Sales", x="Name")

# 绘制多图，设置行数、列数、共享y轴刻度、图区大小，得到多个ax
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7, 4))
# 传入ax0，绘制第一个图
top_10.plot(kind='barh', y="Sales", x="Name", ax=ax0)
ax0.set_xlim([-10000, 140000])
ax0.set(title='Revenue', xlabel='Total Revenue', ylabel='Customers')
avg = top_10['Sales'].mean()
ax0.axvline(x=avg, color='b', label='Average', linestyle='--', linewidth=1)

# 传入ax1，绘制第二个图
top_10.plot(kind='barh', y="Purchases", x="Name", ax=ax1)
avg = top_10['Purchases'].mean()
ax1.set(title='Units', xlabel='Total Units', ylabel='')
ax1.axvline(x=avg, color='b', linestyle='--', linewidth=1)

# 设置总标题
fig.suptitle('2014 Sales Analysis', fontsize=14, fontweight='bold')

# 不显示legend（图例）
ax0.legend().set_visible(False)
ax1.legend().set_visible(False)

# 用fig.canvas.get_supported_filetypes()查看支持保存的文件类型
fig.savefig('/Users/jiajing/My Data Analysis/sales.png', transparent=False, dpi=80, bbox_inches="tight")

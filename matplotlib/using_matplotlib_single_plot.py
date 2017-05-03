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

fig, ax = plt.subplots(figsize=(10, 6))
# 传入ax参数，可用于后续操作
top_10.plot(kind='barh', y="Sales", x="Name", ax=ax)
# 设置x轴的数据范围
ax.set_xlim([-10000, 140000])
# ax.set_xlabel('Total Revenue')
# ax.set_ylabel('Customer')
ax.set(title='2014 Revenue', xlabel='Total Revenue', ylabel='Customer')
# 不显示legend（图例）
ax.legend().set_visible(False)

# 用于坐标刻度值格式转换的函数
def currency(x, pos):
	'The two args are the value and tick position'
	if x >= 1000000:
		return '${:1.1f}M'.format(x*1e-6)
	return '${:1.1f}K'.format(x*1e-3)
# 将转换函数传给FuncFormatter
formatter = FuncFormatter(currency)
# 对x轴的主刻度设置格式
ax.xaxis.set_major_formatter(formatter)


avg = top_10['Sales'].mean()
# 在图中加一条代表均值的标记线（b：蓝色，--：虚线），axvline中v是指垂直方向
ax.axvline(x=avg, color='b', label='Average', linestyle='--', linewidth=1)

for cust in [3, 5, 8]:
	# 设置标记文本的坐标和文字内容
	ax.text(115000, cust, "New Customer")

plt.show()
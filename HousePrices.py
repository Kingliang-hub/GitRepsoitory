# 加载 Python 库
import numpy as np
# 加载数据预处理模块
import pandas as pd
# 加载绘图模块
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.feature_extraction
from sklearn.feature_extraction import DictVectorizer

sns.set_style(style="darkgrid")

# 这里读取的数据是与项目文件同级目录下，或同一个文件夹中。
df = pd.read_csv("train.csv")
# 打印数据前五行
df_head = df.head()
# 打印数据描述
df_descirbe = df.describe()
# 打印数据描述
df_info = df.info()
# 判断是否存在null
df_isnull = df.isnull().any()
# 取数据前100行
# df_new=df.head(100)

# 删除带空值的列
# df_new = df.dropna(axis=1)
# 替换带空值的列为0
#df_new2 = df.fillna(value=0)
df_new2=df
# x_vars = df_new2.columns
# for label in df_new2.columns:
#     print(df_new2[label].dtypes)
#     if (df_new2[label].dtypes == 'int64' or df_new2[label].dtypes == 'float64'):
#         print(label)
#     else:
#         label_mapping = {label: idx for idx, label in enumerate(set(df_new2[label]))}
#         df_new2[label] = df_new2[label].map(label_mapping)

# x_vars=df_new2.columns[1:]
# print(x_vars)
# #分别分析所取的属性与价格的分布关系图
# x_vars=df_new2.columns[2:]
# #分别分析所取的属性与价格的分布关系图
# for x_var in x_vars:
#     #df.plot(kind='scatter',x=str(x_var),y='SalePcrice') #设置绘图的行和列
#     df_new2.plot(kind='scatter',x=x_var,y='SalePrice')
# plt.show()
# print('打印图片成功')
#删除原始数据中的索引 id
df_new2.drop(["Id"],axis=1,inplace=True)
#计算属性间的相关系数图
corr = df_new2.corr()
#绘制属性相关系数的热力图
plt.figure(figsize=(48,24))
sns.heatmap(corr,annot=True,cmap="RdBu")

plt.show()

plt.figure(figsize=(48,24))
#配置下三角热力图区域显示模式
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.set_style(style='white')
#对相关系数图进行三角显示
sns.heatmap(corr,annot=True,cmap='RdBu',mask=mask)
plt.show()

plt.figure(figsize=(48,24))
#配置强相关模式，相关系数大于 0.5
mask = np.zeros_like(corr[corr>=.5],dtype=np.bool)
# Create a msk to draw only lower diagonal corr map
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
#显示强相关模式的相关系数热力值，低于参考值的部分显示为白色，从而获取强相关项
sns.heatmap(corr[corr>=.5],annot=True,mask=mask,cbar=False)
plt.show()
#根据请相关系数图判断
#overallqual,Yearbuilt,
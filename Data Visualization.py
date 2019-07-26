#!/usr/bin/env python
# coding: utf-8

# #### Importing libraries needed for Visualization

# In[225]:


import matplotlib.pyplot as plt
import seaborn as sns
import pylab


# In[251]:


import numpy as np
import pandas as pd


# #### Line Plots

# In[34]:


a = np.random.randint(1,10,6)
b = np.random.randint(2,11,6)


# In[35]:


plt.plot(a, b)


# In[36]:


plt.plot(a, b, linewidth = 5)


# In[37]:


plt.plot(a, b, linewidth=5, color='m')


# In[38]:


plt.plot(a, b, linewidth=3, color='m', marker='D')


# In[39]:


plt.plot(a, b, linewidth=3, color='m', marker='D', markerfacecolor='b')


# In[40]:


plt.plot(a, b, linewidth=3, color='m', marker='D', markerfacecolor='b', markersize=15)


# In[41]:


plt.plot(a, b, linewidth=3, color='m', marker='D', markerfacecolor='b', markersize=15, linestyle='dotted')


# In[42]:


plt.plot(a, b, linewidth=3, color='m', marker='D', markerfacecolor='b', markersize=15)


# In[43]:


plt.plot(a, b, linewidth=3, color='m', marker='D', markerfacecolor='b', markersize=15, label='Graph One')
plt.plot(b, a, linewidth=3, color='b', marker='D', markerfacecolor='m', markersize=15, label='Graph Two')
plt.legend()


# In[44]:


x = [1,2,3]
y = [5,7,4]
x2 = [1,2,3]
y2 = [10,14,12]
plt.figure()
plt.plot(x,y, c='darkred', label='First Line')
plt.plot(x2, y2, c='blue', label='Second Line')
plt.xlabel('Values across x-axis')
plt.ylabel('Values across y-axis')
plt.title('Two Line Graphs one above other.')
plt.legend()
plt.show


# In[45]:


x = [1,2,3]
y = [5,7,4]
x2 = [1,2,3]
y2 = [10,14,12]
plt.figure(figsize=(8,8))
plt.plot(x,y, c='red', label='First Line', marker='*', mfc='blue', ms=20)
plt.plot(x2, y2, c='blue', label='Second Line', marker='s', mfc='lime', ms=10)
plt.xlabel('Values across x-axis')
plt.ylabel('Values across y-axis')
plt.title('Two Line Graphs one above other.')
plt.legend()
plt.show


# In[ ]:





# In[219]:


x = np.linspace(0, 10, 25)
y = x * x + 2
print(x)
print(y)


# In[220]:


print(np.array([x, y]).reshape(25, 2).reshape(2, 25))


# In[226]:


pylab.plot(x, y, 'b')


# In[227]:


pylab.subplot(1,2,1)
pylab.plot(x, y, 'r--')
pylab.subplot(1,2,2)
pylab.plot(y, x, 'g-*')


# In[228]:


fig = plt.figure()
axes = fig.add_axes([0.1,0.5,0.8,0.8])
axes.plot(x, y, 'r--')


# In[229]:


fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.9,0.9])
axes.plot(x, y, 'b--')


# In[230]:


fig, axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
    ax.plot(x, y, 'b--')


# In[231]:


fig, axes = plt.subplots(nrows=2, ncols=1)
for ax in axes:
    ax.plot(x, y, 'r--')


# In[232]:


fig = plt.figure(dpi=100)
# Left, Right, Width, Hieght
axes1 = fig.add_axes([0.1, 0.1, 0.9, 0.9])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.4])

axes1.plot(x, y, 'r-')
axes2.plot(x, y, 'g*')


# In[233]:


fig = plt.figure(figsize=(10, 9), dpi= 200)
fig.add_subplot()
plt.plot(x, y, 'r-')


# In[234]:


fig, axes = plt.subplots()
axes.plot(x, y, 'b--')
axes.set_title('Graph')
axes.set_xlabel('X')
axes.set_ylabel('Y')


# In[236]:


ax.legend(['Label 1', 'Label 2'])
fig, axes = plt.subplots(dpi=100)
axes.plot(x, x**2)
axes.plot(x, x**3)
axes.set_title('Graph')
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.legend(['Y = x**2', 'Y = x**3'], loc=2)


# In[237]:


fig, axes = plt.subplots(dpi = 100)
axes.plot(x, x**1.5, color = 'red', alpha = 0.5)
axes.plot(x, x+2, color = 'blue', alpha = 0.5)
axes.plot(x, x+4, color= 'green', alpha = 0.5)


# In[238]:


fig, axes = plt.subplots(dpi = 100)
axes.plot(x, x, color = 'blue', lw = 1)
axes.plot(x, x+2, color = 'blue', lw = 2)
axes.plot(x, x+4, color= 'blue', lw = 3)
axes.plot(x, x+6, color= 'blue', lw = 4)


# In[239]:


fig, axes = plt.subplots(dpi = 100)
axes.plot(x, x, color = 'red', lw = 1, linestyle = '-')
axes.plot(x, x+2, color = 'green', lw = 2, linestyle = '-.')
axes.plot(x, x+4, color= 'blue', lw = 3, linestyle = ':')
axes.plot(x, x+6, color= 'orange', lw = 4, linestyle = '--')


# In[240]:


ax.legend(['1', '5', '9', '13'])
fig, axes = plt.subplots(dpi = 110)
axes.plot(x, x+1, color = 'red', marker = 'o', markersize=12)
axes.plot(x, x+5, color = 'blue', marker = '1', markersize=9)
axes.plot(x, x+9, color= 'green', marker = 's', markersize=6)
axes.plot(x, x+13, color= 'orange', marker = '*', markersize=3)
axes.set_title('Graph')
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.legend(['12', '9', '6', '3'])


# In[241]:


ax.legend(['1', '5', '9', '13'])
fig, axes = plt.subplots(dpi = 100)
axes.plot(x, x+1, color = 'red', marker = 'o', markersize=12, markerfacecolor='k')
axes.plot(x, x+5, color = 'blue', marker = '1', markersize=9)
axes.plot(x, x+9, color= 'green', marker = 's', markersize=12, markerfacecolor='y')
axes.plot(x, x+13, color= 'orange', marker = '*', markersize=3)
axes.set_title('Graph')
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.legend(['12', '9', '6', '3'])


# In[242]:


fig , Axes = plt.subplots(1, 2, figsize=(10, 10))
Axes[0].plot(x, x **2, x, x ** 3, lw= 3)
Axes[0].grid(True)

Axes[1].plot(x, x **2, x, x ** 3, lw= 3)
Axes[1].grid(True)
Axes[1].set_xlim([1, 5])
Axes[1].set_ylim([0, 60])


# In[243]:


n = np.array([0, 1, 2, 3, 4, 5])

fig , ax = plt.subplots(1, 4, figsize=(16, 5))

ax[0].set_title('Scatter')
ax[0].scatter(x, x + 0.25 * np.random.randn(len(x)), color= 'orange')

ax[1].set_title('Step')
ax[1].step(n, n**2, lw = 3)

ax[2].set_title('Bar')
ax[2].bar(n, n**2)

ax[3].set_title('Fill Between')
ax[3].fill_between(x, x**2, x**3, color='orange')


# In[ ]:





# In[ ]:





# ### Scatter plot

# In[46]:


# Generating some random data with numpy
population = np.random.randint(0, 80, 30)
salary = np.random.randint(10,150,30)


# In[47]:


population


# In[48]:


salary


# In[49]:


plt.scatter(population, salary)


# In[50]:


plt.scatter(population, salary, c='r')


# In[51]:


plt.scatter(population, salary, c='r', s=100)


# In[52]:


plt.scatter(population, salary, c='r', s=100, marker='*')


# In[53]:


plt.scatter(population, salary, c='r', s=100, marker='*', label='Population Age vs. Salary')
plt.legend()


# In[54]:


plt.scatter(population, salary, c='red', s=100, marker='*', label='Population Age vs. Salary', alpha=0.5)
plt.xlabel('X')
plt.xlabel('Y')
plt.legend()


# In[208]:


plt.scatter(population, salary, c='red', s=400, marker='*', label='Population Age vs. Salary', 
            alpha=0.5, linewidths=15, edgecolors='orange')
plt.xlabel('X')
plt.xlabel('Y')
plt.legend()


# In[218]:


N = 30
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N)) ** 2
plt.scatter(x, y, s=area, c=colors, alpha=0.4)
plt.show()


# In[ ]:





# In[ ]:





# ### Pie Chart

# In[56]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities)


# In[57]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'])


# In[58]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0))


# In[59]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), startangle=105)


# In[60]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%')


# In[62]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3)


# In[63]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True) 


# In[64]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True, labeldistance=0.6)


# In[65]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True, labeldistance=0.6,
       radius=1.5)


# In[66]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True, labeldistance=0.6,
       radius=1.5, counterclock=True)


# In[67]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True, labeldistance=0.6,
       radius=1.5, counterclock=False)


# In[68]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True, labeldistance=0.6,
       radius=1.5, counterclock=True, frame=True)


# In[69]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True, labeldistance=0.6,
       radius=1.5, counterclock=True, frame=False)


# In[70]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True, labeldistance=0.6,
       radius=1.5, counterclock=True, frame=False, rotatelabels=True)


# In[71]:


# Generating Data
slices = [5, 4, 1, 3]
activities = ['Sleeping', 'Studying', 'Playing', 'Browsing']
plt.figure(figsize=(8, 8))
plt.pie(slices, labels=activities, colors=['r', 'b', 'orange', 'lime'], explode=(0, 0, 0.1, 0), 
        startangle=105, autopct='%1.1f%%', pctdistance=0.3, shadow=True, labeldistance=0.6,
       radius=1.5, counterclock=True, frame=False, rotatelabels=False)


# In[ ]:





# In[244]:


languages = ['Python', 'Java', 'Javascript', 'PHP', "C#", 'C++']
popularity = [22.5, 18.4, 13.6, 17.2, 11.1, 14.3]
colors = ['orange', 'blue', 'yellow', 'r', 'm', 'c']


# In[250]:


plt.figure(figsize=(10,10))
explode = (0.1, 0, 0, 0, 0, 0)
plt.pie(popularity, labels=languages, explode=explode, startangle=90, colors=colors,
       shadow=True, autopct='%1.1f%%')


# In[ ]:





# ### What is boxplot?

# 
# 
# - Draw a box plot to show distributions with respect to categories.
# - A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates     comparisons between variables or across levels of a categorical variable. 
# - The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be �outliers� using a method that is a function of the inter-quartile range.

# In[94]:


sns.boxplot('size', data=tips_data)


# In[96]:


sns.boxplot('total_bill', data=tips_data)


# In[97]:


sns.boxplot('day', 'total_bill', data=tips_data)


# In[100]:


sns.boxplot('day', 'total_bill', data=tips_data, hue='smoker', palette='spring')


# In[105]:


sns.boxplot('day', 'total_bill', data=tips_data, hue='time', linewidth=5)


# In[112]:


sns.boxplot(data=tips_data, orient='h')


# In[113]:


iris_data = sns.load_dataset("iris")
iris_data.sample(10)


# In[118]:


sns.boxplot(x='day', y='total_bill', data=tips_data)
sns.swarmplot(x='day',y= 'total_bill', data=tips_data, color='k')


# ### Bar Plots

# In[72]:


plt.figure(figsize=(8, 8))
x = [2,4,6,8,10]
y = [2,7,2,3,4]
plt.bar(x, y)
plt.xlabel('Values across x-axis')
plt.ylabel('Values across y-axis')
plt.title('Bar Graph')


# In[73]:


plt.figure(figsize=(8, 6))
x = [2,4,6,8,10]
y = [2,7,2,3,4]
plt.bar(x, y, color='red', width=0.5)
plt.xlabel('Values across x-axis')
plt.ylabel('Values across y-axis')
plt.title('Bar Graph')


# In[74]:


plt.figure(figsize=(8, 6))
x = [2,4,6,8,10]
y = [2,7,2,3,4]

x2 = [1,3,5,7,9]
y2 = [2,5,4,6,8]

plt.bar(x, y, label='First Bar', color='red', width=0.5)
plt.bar(x2, y2, label='Second Bar', color='b', width=0.2)
plt.legend()
plt.xlabel('Values across x-axis')
plt.ylabel('Values across y-axis')
plt.title('Bar Graph')


# In[75]:


# generating bar plots from real world data (tips)
import seaborn as sns
tips_data = sns.load_dataset('tips')
tips_data.head()


# In[76]:


sns.barplot(tips_data['day'], tips_data['tip'])


# In[77]:


sns.barplot(x = 'day', y = 'total_bill', data = tips_data)


# In[78]:


sns.barplot(x = 'day', y = 'total_bill', data = tips_data, hue='sex')


# In[79]:


sns.barplot(x = 'day', y = 'total_bill', data = tips_data, hue='sex', palette='spring')


# In[80]:


sns.barplot(x = 'day', y = 'total_bill', data = tips_data, hue='sex', palette='BuGn_d')


# In[81]:


sns.barplot(x = 'day', y = 'total_bill', data = tips_data, hue='smoker')


# In[82]:


sns.barplot(x = 'day', y = 'total_bill', data = tips_data, hue='smoker', palette='BuGn_d')


# In[83]:


sns.barplot(x = 'day', y = 'total_bill', data = tips_data, hue='smoker', palette='spring')


# In[84]:


sns.barplot(tips_data['day'], tips_data['tip'], palette='spring', order=['Sun', 'Thur', 'Sat', 'Fri'])


# In[85]:


sns.barplot(x="day", y="total_bill", data=tips_data,hue = 'sex' ,capsize = 0.2,palette = 'husl')
plt.legend()


# In[86]:


sns.barplot(x="day", y="total_bill", data=tips_data,hue = 'sex' ,capsize = 0.7,palette = 'husl')
plt.legend()


# In[87]:


sns.barplot("size", y="total_bill", data=tips_data, palette="Blues_d")


# In[88]:


sns.barplot("day", "total_bill", data=tips_data, linewidth=8.5, errcolor=".2", edgecolor=".8")


# In[89]:


sns.relplot('total_bill', 'tip', data= tips_data, color='r', hue='sex', size='size', col='sex')


# In[92]:


sns.barplot('size', 'tip', data=tips_data)


# ### What is heatmat?
# 
# 
# A heat map (or heatmap) is a graphical representation of data where the individual values contained in a matrix are represented as colors. It is a bit like looking a data table from above. It is really useful to display a general view of numerical data,
# 
# not to extract specific data point. It is quite straight forward to make a heat map, as shown on the examples below. However be careful to understand the underlying mechanisms. You will probably need to normalise your matrix, choose
# 
# a relevant colour palette, use cluster analysis and thus permute the rows and the columns of the matrix to place similar values near each other according to the clustering.
# 

# In[119]:


uniform_data = np.arange(1,17).reshape(4,4)
sns.heatmap(uniform_data)


# In[123]:


a = np.array([[1,2,3,4],[5,7,2,6],[8,1,4,9],[2,5,0,9]])
sns.heatmap(a)


# ### What is distplot:
# 

# - Flexibly plot a univariate distribution of observations.
# - This function combines the matplotlib hist function (with automatic calculation of a good default bin size) with the seaborn kdeplot() and rugplot() functions. It can also fit scipy.stats distributions and plot the estimated PDF over the data.

# In[138]:


num = np.random.randn(150)
sns.distplot(num)


# In[140]:


sns.distplot(num, color='lime')


# In[142]:


sns.distplot(num, color='red', hist=False)


# In[145]:


sns.distplot(num, hist=False, rug=True, color='k')


# In[151]:


sns.distplot(num, color='b', vertical=True, kde=True)


# In[153]:


sns.distplot(num, color='b', vertical=True, kde=False)


# In[ ]:





# ### Whatis PairGrid
# - Subplot grid for plotting pairwise relationships in a dataset.
# - http://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid
# 

# ### Draw a scatterplot for each pairwise relationship:
# 

# In[156]:


iris_data = sns.load_dataset("iris")
x = sns.PairGrid(iris_data)
x = x.map(plt.scatter)



# In[157]:


iris_data = sns.load_dataset("iris")
x = sns.PairGrid(iris_data)
x = x.map(plt.scatter, color='k')


# In[158]:


x = sns.PairGrid(iris_data)
x = x.map_diag(plt.hist, color='lime')


# In[160]:


x = sns.PairGrid(iris_data)
x = x.map_offdiag(plt.scatter, color= 'k')


# In[162]:


x = sns.PairGrid(iris_data)
x = x.map_diag(plt.hist, color='lime')
x = x.map_offdiag(plt.scatter, color= 'k')


# In[168]:


a = sns.PairGrid(iris_data, hue='species')
a = a.map_diag(plt.hist)
a = a.map_offdiag(plt.scatter)
a = a.add_legend()


# In[173]:


x = sns.PairGrid(iris_data,hue = 'species')
x = x.map_diag(plt.hist,histtype = 'step',linewidth =1)
x = x.map_offdiag(plt.scatter)
x = x.add_legend()


# In[176]:


x = sns.PairGrid(iris_data, vars=["sepal_length", "sepal_width"])
x =  x.map(plt.scatter, color='g')


# In[177]:


x = sns.PairGrid(iris_data,hue = 'species',vars = ['petal_length','petal_width'])
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x = x.add_legend()


# In[181]:


x = sns.PairGrid(iris_data)
x = x.map_diag(plt.hist, color='k')
x = x.map_upper(plt.scatter, color='b')
x = x.map_lower(sns.kdeplot, color='lime')
x = x.add_legend()


# In[186]:


g = sns.PairGrid(iris_data, hue="species", palette="Set1",
                 hue_kws={"marker": ["o", "s", "D"]})
g = g.map(plt.scatter, linewidths=1, edgecolor="w", s=40)
g = g.add_legend()


# In[ ]:





# ### What is violinplot?
# 
# 
# 
# Violinplots allow to visualize the distribution of a numeric variable for one or several groups. It is really close from a boxplot, but allows a deeper understanding of the density. 
# 
# Violins are particularly adapted when the amount of data is huge and showing individual observations gets impossible. Seaborn is particularly adapted to realize them through its violin function.
# 
# Violinplots are a really convenient way to show the data and would probably deserve more attention compared to boxplot that can sometimes hide features of the data.

# In[188]:


sns.violinplot('total_bill', data=tips_data)


# In[190]:


sns.violinplot('day', 'total_bill', data=tips_data)


# In[196]:


sns.violinplot('day', 'total_bill', data=tips_data, hue='sex', palette='muted')


# In[198]:


sns.violinplot('day', 'total_bill', data=tips_data, hue='sex', split=True, palette='spring')


# In[199]:


sns.violinplot('day', 'total_bill', data=tips_data, hue='sex', split=True, palette='spring', order=['Sat', 'Thur'])


# In[200]:


tips_data.sample(10)


# In[206]:


plt.figure(figsize=(8,8))
sns.violinplot('total_bill', 'day', data=tips_data, hue='sex', palette='spring')
plt.show()


# #### Lineplot and scatter plots together

# In[207]:


x = [1,2,3,4]
y = [10,20,25,30]
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(x, y, color='orange', linewidth=3) # line plot

ax.scatter([2,4,6], # scatter plot
           [5,15,25], 
           color='b',
           marker='*')
plt.title(r'$sigma_i=15$', fontsize=30)
# plt.savefig('foo2.png', transparent=True) # save the figure to your local directory
plt.show() # show the plot to you (Display)


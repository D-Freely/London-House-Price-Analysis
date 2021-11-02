import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
from bs4 import BeautifulSoup as bs

# data source - https://www.kaggle.com/justinas/housing-in-london

pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('.../housing_in_london_monthly_variables.csv', parse_dates=['date'])

num_area = df.area.nunique()
area_lst = df.area.unique().tolist()

df['year'] = df['date'].dt.year

# ------------------------------------------------------------------------------------------

df_ldn = df[df['borough_flag'].isin([1])].reset_index()
df_ldn = df_ldn[~df_ldn['year'].isin([2020])].reset_index()
ldn_borough_lst = df_ldn.area.unique().tolist()

r = requests.get('https://en.wikipedia.org/wiki/List_of_sub-regions_used_in_the_London_Plan', verify=False)
soup = bs(r.content, features="html.parser")

table = soup.find('table', {'class':'wikitable'})
columns = table.find_all('th')
column_names = [c.text.replace('\n', '').replace('[6]','').lower() for c in columns]

core_ldn = column_names[4:9]

table_rows = table.find('tbody').find_all('tr')

l = []
n_l = []
for tr in table_rows:
    td = tr.find_all('td')
    row = [tr.text.strip().lower() for tr in td]
    l.append(row)

l2 = filter(None, l)

for lst in l2:
    n_l.append(lst[0].split(', '))

central = n_l[0]
east = n_l[1]
north = n_l[2]
south = n_l[3]
west = n_l[4]

def group_area(borough):
    if borough in central:
        return 'central'
    elif borough in east:
        return 'east'
    elif borough in north:
        return 'north'
    elif borough in south:
        return 'south'
    elif borough in west:
        return 'west'
    else:
        return 'N/A'

def total_sold_area(core_area, year):
    df_total_area = df_ldn[df_ldn['core_ldn_area'].isin([core_area])]
    df_total_area_year = df_total_area[df_total_area['year'].isin([year])]
    total = df_total_area_year['houses_sold'].sum()
    return total

def weighted_avg(avg, n, sum_n):
    r = avg * (n / sum_n)
    return r

df_ldn['core_ldn_area'] = df_ldn.apply(lambda x: group_area(x['area']), axis=1)

df_ldn['weighted_avg_by_yr'] = df_ldn.apply(lambda x: weighted_avg(x['average_price'], x['houses_sold'], total_sold_area(x['core_ldn_area'], x['year'])), axis=1)

df_price_year = df_ldn.groupby(['core_ldn_area', 'year'])['weighted_avg_by_yr'].sum().reset_index()

df_linearR = df_price_year.groupby('year')['weighted_avg_by_yr'].mean().reset_index()

def lR_inputs_xy(x, y):
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    x_reshaped = x_array.reshape(-1, 1)
    y_reshaped = y_array.reshape(-1, 1)
    return x_reshaped, y_reshaped

def lR_predict_xinput(x):
    x_array = np.asarray(x)
    x_reshaped = x_array.reshape(-1, 1)
    return x_reshaped

linefitter = LinearRegression()
x1, y1 = lR_inputs_xy(df_linearR.year.tolist(), df_linearR.weighted_avg_by_yr.tolist())
linefitter.fit(x1, y1)

x2 = df_linearR.year.tolist()
x2.extend([2021, 2022, 2023, 2024, 2025])
x2_input = lR_predict_xinput(x2)

y1_predicted = linefitter.predict(x1)
y2_predicted = linefitter.predict(x2_input)

#--------------

def reformat(area):
    n_area = area[0].upper() + area[1:]
    return n_area

fig, ax = plt.subplots()

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

for area in core_ldn:
    df_price_area = df_price_year[df_price_year['core_ldn_area'].isin([area])].reset_index()
    ax.plot(df_price_area.year, df_price_area.weighted_avg_by_yr, label=reformat(area))

ax.plot(x2, y2_predicted, color='red', linestyle=':', linewidth=3)
ax.plot([2025, 2025, 1995], [1995, 644952.83, 644952.83], 'k-', lw=1,dashes=[2, 2])

ax.annotate(text='£644,953',
             xy=(2025, 644952.83),
             xytext=(0, 20),
             xycoords='data',
             textcoords='offset pixels',
             arrowprops=dict(arrowstyle="->", color='k', linestyle='--'),
             color='k')

ax.set_ylabel('$\it{Avg. House Price}$')
ax.yaxis.labelpad = 15
ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
ax.set_title('House Price Change - London sub-regions')
ax.legend(loc='upper left')

y_list = list(range(100000, 800001, 100000))
ax.set_yticks(y_list)
y_labels = ["£{:,}".format(y) for y in y_list]
ax.set_yticklabels(y_labels)

x_list = list(range(1995, 2026, 5))
ax.set_xticks(x_list)
ax.set_xticklabels(x_list)

# ------------------------------------------------------------------------------------------

fig, ax3 = plt.subplots()

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

df_crime = df_ldn.groupby(['core_ldn_area', 'year'])['no_of_crimes'].sum().reset_index()
df_crime.dropna(subset=['no_of_crimes'], inplace=True)

for area in core_ldn:
    df_area = df_crime[df_crime['core_ldn_area'].isin([area])].reset_index()
    ax3.plot(df_area.year, df_area.no_of_crimes, label=reformat(area))

ax3.axis([2001, 2019, 0, 300000])
ax3.set_ylabel('$\it{Crimes Committed}$')
ax3.yaxis.labelpad = 15
ax3.set_title('Total crime committed in London sub-regions')

y2_list = list(range(50000, 300001, 50000))
ax3.set_yticks(y2_list)
y2_labels = ['{:,}'.format(y) for y in y2_list]
ax3.set_yticklabels(y2_labels)

ax3.legend(loc='best')

# ------------------------------------------------------------------------------------------

def total_sold_year(year):
    df_total_year = df_ldn[df_ldn['year'].isin([year])]
    total = df_total_year['houses_sold'].sum()
    return total

df_ldn['weight_avg_2'] = df_ldn.apply(lambda x: weighted_avg(x['average_price'], x['houses_sold'], total_sold_year(x['year'])), axis=1)
df_plot = df_ldn.groupby('year')['weight_avg_2'].sum().reset_index()

df_sold = df_ldn.groupby('year')['houses_sold'].sum().reset_index()

fig, ax1 = plt.subplots()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

ax2 = ax1.twinx()

ax1.bar(df_sold['year'], df_sold['houses_sold'], color='RoyalBlue')
ax1.set_ylabel('$\it{Houses Sold (Total)}$', color='RoyalBlue')
ax1.yaxis.labelpad = 15
ax1.set_title('London Total Houses Sold vs Average Sale Price')
y_ax1 = list(range(25000, 175001, 25000))
ax1.set_yticks(y_ax1)
y_ax1_labels = ['{:,}'.format(y) for y in y_ax1]
ax1.set_yticklabels(y_ax1_labels)

ax2.plot(df_plot['year'], df_plot['weight_avg_2'], 'Tomato')
ax2.set_ylabel('$\it{Average House Sale Price (£)}$', color='Tomato')
ax2.yaxis.labelpad = 15
y_ax2 = list(range(100000, 500001, 100000))
ax2.set_yticks(y_ax2)
y_ax2_labels = ['£{:,}'.format(y) for y in y_ax2]
ax2.set_yticklabels(y_ax2_labels)

plt.show()

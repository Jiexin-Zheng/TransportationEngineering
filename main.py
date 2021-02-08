import xlrd
import datetime
import time
from xlrd import xldate_as_datetime
#import re
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize


def cum_prob_curve(data,bins,title,xlabel):
    '''
    绘制概率分布直方图和累计概率分布曲线
    '''
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter
    #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
    from matplotlib.pyplot import MultipleLocator
    fig= plt.figure(figsize=(8, 4),dpi=100)
    # 设置图形的显示风格
    plt.style.use('ggplot')
    # 中文和负号的正常显示
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    ax1 = fig.add_subplot(111)
    ##概率分布直方图
    a1,a2,a3=ax1.hist(data,bins =bins, alpha = 0.65,density=1,edgecolor='k')
    ##累计概率曲线
    #生成累计概率曲线的横坐标
    indexs=[]
    a2=a2.tolist()
    for i,value in enumerate(a2):
        if i<=len(a2)-2:
            index=(a2[i]+a2[i+1])/2
            indexs.append(index)
    #生成累计概率曲线的纵坐标
    def to_percent(temp,position):
        return '%1.0f'%(100*temp) + '%'
    dis=a2[1]-a2[0]
    freq=[f*dis for f in a1]
    acc_freq=[]
    for i in range(0,len(freq)):
        if i==0:
            temp=freq[0]
        else:
            temp=sum(freq[:i+1])
        acc_freq.append(temp)
    #这是双坐标关键一步
    ax2=ax1.twinx()
    #绘制累计概率曲线
    ax2.plot(indexs,acc_freq)
    #设置累计概率曲线纵轴为百分比格式
    ax2.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax1.set_xlabel(xlabel,fontsize=12)
    ax1.set_title(title,fontsize =12)
    #把x轴的刻度间隔设置为1，并存在变量里
    # x_major_locator=MultipleLocator(xlocator)
    # ax1.xaxis.set_major_locator(x_major_locator)
    ax1.set_ylabel('frequency',fontsize=12)
    ax2.set_ylabel("cumulative frequency",fontsize=12)
#    plt.savefig(pic_path,format='png', dpi=300)
    plt.show()


time_last = []
data = xlrd.open_workbook("data.xlsx")
data_sheet = data.sheets()[0]
time_original = data_sheet.col_values(0)
traffic_volume = []
density = []
speed = []


resArray = [[0]*4 for i in range(data_sheet.nrows - 1)]
for i in range(data_sheet.nrows - 1):
    line=data_sheet.row_values(i + 1)
    resArray[i] = line

time_start = []
time_end = []

for i in range(0 + 1,len(time_original)):
    time_last.append(xldate_as_datetime(time_original[i],0).strftime('%Y-%m-%d %H:%M:%S'))

for i in range(0,len(time_last)):
    resArray[i][0] = time_last[i]

time_reference  = pd.date_range(start = '12/05/2012',end = '01/01/2013',freq = '5min')

whole_padding = [[0]*4 for i in range(288 * 27)]

for i in range(0,len(whole_padding)):
    whole_padding[i][0] =  time_reference[i].strftime('%Y-%m-%d %H:%M:%S')

for i in range(0,len(resArray)):
    for j in range(0,len(whole_padding)):
        if whole_padding[j][0] == resArray[i][0]:
            whole_padding[j][1] = resArray[i][1]
            whole_padding[j][2] = resArray[i][2]
            whole_padding[j][3] = resArray[i][3]
#print(whole_padding)
time_start = pd.date_range(start = '12/05/2012',end = '12/31/2012')

for i in range(0,len(time_start)):
    time_end.append(time_start[i].strftime('%Y-%m-%d') + ' 00:00:00')


#原始数据每日有效组数统计
a = [[] for i in range(27)]

for j in range(0,len(time_end)):
    for i in range(0,len(time_last)):
        if j == len(time_end) - 1:
            if time_end[j] <= time_last[i]:
                a[j].append(time_last[i])
        else:
            if time_end[j] <= time_last[i] <= time_end[j + 1]:
                a[j].append(time_last[i])

#for i in range(27):
#    print(len(a[i]))


#补流量缺失值
for i in range(len(whole_padding)):
    if whole_padding[i][1] == 0:
        traffic_volume.append( (float(whole_padding[i - 2][1]) + float(whole_padding[i - 1][1]) + float(
        whole_padding[i + 1][1]) + float(whole_padding[i + 2][1])) / 4 )
    else:
        traffic_volume.append(float(whole_padding[i][1]))

#print(traffic_volume)

#补密度缺失值
for i in range(len(whole_padding)):
    if whole_padding[i][2] == 0:
        density.append( (float(whole_padding[i - 2][2]) + float(whole_padding[i - 1][2]) + float(
        whole_padding[i + 1][2]) + float(whole_padding[i + 2][2])) / 4 )
    else:
        density.append(float(whole_padding[i][2]))


#补车速缺失值
for i in range(len(whole_padding)):
    if whole_padding[i][3] == 0:
        speed.append( (float(whole_padding[i - 2][3]) + float(whole_padding[i - 1][3]) + float(
        whole_padding[i + 1][3]) + float(whole_padding[i + 2][3])) / 4 )
    else:
        speed.append(float(whole_padding[i][3]))


#日交通量列表获取
sum1 = 0
traffic_volume_day = []
for i in range(len(traffic_volume)):
    sum1 = sum1 + traffic_volume[i]
    if ((i + 1) % 288 == 0 and i >= 287):
        traffic_volume_day.append(sum1)
        sum1 = 0

for i in range(len(traffic_volume_day)):
    print(traffic_volume_day[i])

# fig = plt.figure(figsize = (12,4))
# ax = fig.add_subplot(111)
# ax.set(ylabel='daily traffic volume/veh', xlabel='time/day', title='daily traffic changing trend')
# ax.plot(time_start, traffic_volume_day, color='red', marker='+', linestyle='dashed')
# #ax.legend()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.tight_layout()
# plt.show()
# plt.savefig('./test.png')


#小时交通量列表获取
sum1 = 0
traffic_volume_hour = []
for i in range(len(traffic_volume)):
    sum1 = sum1 + traffic_volume[i]
    if ((i + 1) % 12 == 0 and i >= 11):
        traffic_volume_hour.append(sum1)
        sum1 = 0
# print(traffic_volume_hour)
# print(len(traffic_volume_hour))


#计算周平均日交通量
sum1 = 0
week_average_date_traffic = []
count = 0
for i in range(len(traffic_volume_day)):
    if(len(a[i]) > 200):
        sum1 = sum1 + traffic_volume_day[i]
        count = count + 1
    if(((i + 1) % 7 == 0 and i >= 6) or (i == len(traffic_volume_day))):
        week_average_date_traffic.append(sum1 / count)
        sum1 = 0
        count = 0

#print(week_average_date_traffic)

#计算月平均日交通量
sum1 = 0
month_average_date_traffic = 0
count = 0
for i in range(len(traffic_volume_day)):
    if(len(a[i]) > 200):
        sum1 = sum1 + traffic_volume_day[i]
        count = count + 1
month_average_date_traffic = sum1 / count
#print(month_average_date_traffic)


#计算日变系数
Kd = []
for i in range(len(traffic_volume_day)):
    if (len(a[i]) > 200):
        Kd.append(month_average_date_traffic/traffic_volume_day[i])
    else:
        Kd.append(0)

#print(Kd)

# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# ax1 = fig1.add_subplot(111)
# ax1.set(ylabel='Kd', xlabel='time/day', title='Kd changing trend')
# ax1.plot(time_start, Kd, color='red', marker='+', linestyle='dashed')
# plt.tight_layout()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.show()
# plt.savefig('./test1.pdf')



#获取每日高峰小时交通量
peak_hour_traffic = []
for i in range(len(traffic_volume_day)):
    if(len(a[i]) > 200):
        peak_hour_traffic.append(max(traffic_volume_hour[i * 24:(i + 1) * 24]))
    else:
        peak_hour_traffic.append(0)

# print(peak_hour_traffic)
# # print(len(peak_hour_traffic))
# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# ax1 = fig1.add_subplot(111)
# ax1.set(ylabel='peak hour traffic/veh', xlabel='time/day', title='peak hour traffic volume in every day')
# ax1.plot(time_start, peak_hour_traffic, color='red', marker='+', linestyle='dashed')
# plt.tight_layout()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.show()
# plt.savefig('./test1.pdf')



#计算高峰小时流率(5min)
peak_hour_flow_rate_5min = []
for i in range(len(traffic_volume_day)):
    if(len(a[i]) > 200):
        peak_hour_flow_rate_5min.append(max(traffic_volume[i * 288:(i + 1) * 288]))
    else:
        peak_hour_flow_rate_5min.append(0)

for i in range(len(peak_hour_flow_rate_5min)):
    peak_hour_flow_rate_5min[i] = peak_hour_flow_rate_5min[i] * 12

# print(peak_hour_flow_rate_5min)
# # print(len(peak_hour_flow_rate_5min))
#
# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# ax1 = fig1.add_subplot(111)
# ax1.set(ylabel='peak hour flow rate(5min)/veh', xlabel='time/day', title='peak hour flow rate(5min) in every day')
# ax1.plot(time_start, peak_hour_flow_rate_5min, color='red', marker='+', linestyle='dashed')
# plt.tight_layout()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.show()
# plt.savefig('./test1.pdf')


#计算高峰小时流率(15min)
peak_hour_flow_rate_15min = []

sum1 = 0
traffic_volume_15min = []
for i in range(len(traffic_volume)):
    sum1 = sum1 + traffic_volume[i]
    if ((i + 1) % 3 == 0 and i >= 2):
        traffic_volume_15min.append(sum1)
        sum1 = 0
#print(len(traffic_volume_15min))
#print(traffic_volume_15min)


for i in range(len(traffic_volume_day)):
    if(len(a[i]) > 200):
        peak_hour_flow_rate_15min.append(max(traffic_volume_15min[i * 96:(i + 1) * 96]))
    else:
        peak_hour_flow_rate_15min.append(0)

for i in range(len(peak_hour_flow_rate_15min)):
    peak_hour_flow_rate_15min[i] = peak_hour_flow_rate_15min[i] * 4

# print(peak_hour_flow_rate_15min)
# # print(len(peak_hour_flow_rate_15min))
#
#
# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# ax1 = fig1.add_subplot(111)
# ax1.set(ylabel='peak hour flow rate(15min)/veh', xlabel='time/day', title='peak hour flow rate(15min) in every day')
# ax1.plot(time_start, peak_hour_flow_rate_15min, color='red', marker='+', linestyle='dashed')
# plt.tight_layout()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.show()
# plt.savefig('./test1.pdf')



#计算高峰小时系数
peak_hour_factor_5min = []
peak_hour_factor_15min = []
for i in range(len(traffic_volume_day)):
    if(len(a[i]) > 200):
        peak_hour_factor_5min.append(peak_hour_traffic[i]/peak_hour_flow_rate_5min[i])
        peak_hour_factor_15min.append(peak_hour_traffic[i]/peak_hour_flow_rate_15min[i])
    else:
        peak_hour_factor_5min.append(0)
        peak_hour_factor_15min.append(0)
# print(peak_hour_factor_5min)
# print(peak_hour_factor_15min)
#
#
# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# ax1 = fig1.add_subplot(111)
# ax1.set(ylabel='peak hour factor(15min)', xlabel='time/day', title='peak hour factor(15min) in every day')
# ax1.plot(time_start, peak_hour_factor_15min, color='red', marker='+', linestyle='dashed')
# plt.tight_layout()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.show()
# plt.savefig('./test1.pdf')


# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# ax1 = fig1.add_subplot(111)
# ax1.set(ylabel='peak hour factor(5min)', xlabel='time/day', title='peak hour factor(5min) in every day')
# ax1.plot(time_start, peak_hour_factor_5min, color='red', marker='+', linestyle='dashed')
# plt.tight_layout()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.show()
# plt.savefig('./test1.pdf')
#


#计算高峰小时流量比
peak_hour_flow_ratio = []
for i in range(len(traffic_volume_day)):
    if(len(a[i]) > 200):
        peak_hour_flow_ratio.append(peak_hour_traffic[i]/traffic_volume_day[i])
    else:
        peak_hour_flow_ratio.append(0)
# print(peak_hour_traffic)
# print(len(peak_hour_flow_ratio))
# print(peak_hour_flow_ratio)
#
# #
# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# ax1 = fig1.add_subplot(111)
# ax1.set(ylabel='peak hour flow ratio', xlabel='time/day', title='peak hour flow ratio in every day')
# ax1.plot(time_start, peak_hour_flow_ratio, color='red', marker='+', linestyle='dashed')
# plt.tight_layout()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.show()
# plt.savefig('./test1.pdf')


#计算昼间流量比
traffic_volume_day12h = []
sum1 = 0
tmp = 0
count = 0
for i in range(len(traffic_volume_hour)):
    if((i + 1) >= (6 + tmp * 24) and (i + 1) <= (18 + tmp * 24)):
        sum1 = sum1 + traffic_volume_hour[i]
        count = count + 1
    if(count % 12 == 0 and count >= 12):
        traffic_volume_day12h.append(sum1)
        tmp = tmp + 1
        sum1 = 0
        count = 0
#
for i in range(len(traffic_volume_day12h)):
    if(len(a[i]) < 200):
        traffic_volume_day12h[i] = 0


day_flow_ratio = []
for i in range(len(traffic_volume_day)):
    if(len(a[i]) > 200):
        day_flow_ratio.append(traffic_volume_day12h[i]/traffic_volume_day[i])
    else:
        day_flow_ratio.append(0)

# print(traffic_volume_day)
#print(day_flow_ratio)

# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# ax1 = fig1.add_subplot(111)
# ax1.set(ylabel='day traffic volume ratio(12h)', xlabel='time/day', title='day traffic volume ratio(12h) in every day')
# ax1.plot(time_start, day_flow_ratio, color='red', marker='+', linestyle='dashed')
# plt.tight_layout()
# #ax.set_xticklabels(time_start,rotation = 20)
# plt.show()
# plt.savefig('./test1.pdf')


'''
test_date = '他的生日是2016-12-12 14:34,是个可爱的小宝贝.二宝的生日是2016-12-21 11:34,好可爱的.'

test_datetime = '他的生日是2016-12-12 14:34,是个可爱的小宝贝.二宝的生日是2016-12-21 11:34,好可爱的.'
s = "05"
#mat = re.search(r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2})", test_datetime)
for i in range(0 + 1,len(time_last) - 1):
    mat = re.search(r"(\d{4}-\d{1,2}-(%s)\s\d{1,2}:\d{1,2}:\d{1,2})"%s, time_last[i])
    print(mat)
# ('2016-12-12 14:34',)
#print(mat.group(0))
# 2016-12-12 14:34
'''

#获取原始数据
traffic_volume_original = []
density_original = []
speed_original = []
for i in range(len(resArray)):
    traffic_volume_original.append(resArray[i][1])
    density_original.append(resArray[i][2])
    speed_original.append(resArray[i][3])

#绘制地点车速频率分布图
# 绘制累计频率分布曲线
cum_prob_curve(speed_original,20,'speed frequency distribution','speed/(km/h)')

# 计算地点车速频率分布特征值
max_speed = max(speed_original)
min_speed = min(speed_original)
range_speed = max_speed - min_speed
mean_speed = np.mean(speed_original)
std_speed = np.std(speed_original)
sample_sum = len(speed_original)
std_speed = std_speed * pow(sample_sum/(sample_sum - 1),0.5)
# print(max_speed)
# print(min_speed)
# print(range_speed)
# print(mean_speed)
print(sample_sum)
print(std_speed)




#绘制速度-密度（占有率）、速度-流量、流量-密度（占有率）关系图

# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# plt.scatter(density_original, speed_original, color='red', marker='+')
# plt.title('speed - density ')
# plt.xlabel('density/%')
# plt.ylabel('speed/(km/h)')
# plt.show()
# plt.savefig('./test1.pdf')

# def f_1(x, A, B):
#     return A * x + B
#
# A1, B1 = optimize.curve_fit(f_1, density_original, speed_original)[0]
# x1 = np.arange(0, 60)
# y1 = A1 * x1 + B1
# print(A1)
# print(B1)
# plt.plot(x1, y1, "blue")
# plt.title('speed - density ')
# plt.xlabel('density/%')
# plt.ylabel('speed/(km/h)')
# plt.show()


# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# plt.scatter(traffic_volume_original, speed_original, color='red', marker='+')
# plt.title('speed - traffic volume ')
# plt.xlabel('traffic volume/veh')
# plt.ylabel('speed/(km/h)')
# plt.show()
# plt.savefig('./test1.pdf')


# def f_2(x1, A1, B1):
# #     return A1(x1 - pow(x1,2)/B1)
#      return (A1 * x1 ** 2 + B1 * x1)
# A1, B1 = optimize.curve_fit(f_2, traffic_volume_original, speed_original)[0]
# x1 = np.arange(0, 160)
# y1 = A1 * x1 ** 2 + B1 * x1
# print(A1)
# print(B1)
# plt.plot(x1, y1, "blue")
# plt.title('speed - traffic volume ')
# plt.xlabel('traffic volume/veh')
# plt.ylabel('speed/(km/h)')
# plt.show()


# plt.clf()
# fig1 = plt.figure(figsize = (12,4))
# plt.scatter(density_original, traffic_volume_original, color='red', marker='+')
# plt.title('traffic volume - density ')
# plt.xlabel('density/%')
# plt.ylabel('traffic volume/veh')
# plt.show()
# plt.savefig('./test1.pdf')


# def f_2(x1, A1, B1):
# #     return A1(x1 - pow(x1,2)/B1)
#      return (A1 * x1 ** 2 + B1 * x1)
# A1, B1 = optimize.curve_fit(f_2, density_original, traffic_volume_original)[0]
# x1 = np.arange(0, 160)
# y1 = A1 * x1 ** 2 + B1 * x1
# print(A1)
# print(B1)
# plt.plot(x1, y1, "blue")
# plt.title('traffic volume - density ')
# plt.xlabel('density/%')
# plt.ylabel('traffic volume/veh')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# rcParams['font.family'] = 'SimHei'
# config = {
#     "font.family": 'serif', # 衬线字体
#     "font.size": 12, # 相当于小四大小
#     "font.serif": ['SimSun'], # 宋体
#     "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
#     'axes.unicode_minus': False # 处理负号，即-号
# }
# rcParams.update(config)

# plt.figure(figsize=(20, 10), dpi=100)
# error_level = [0,1,2,3,4,5,6,7,8,9,10]
# cs = [23, 10, 38, 30, 36, 20, 28, 36, 16, 29, 15, 26, 30, 26, 38, 34, 33, 25, 28, 40, 28]
# cs = [17, 6, 12, 6, 10, 8, 11, 7, 15, 11, 6, 11, 10, 9, 16, 13, 9, 10, 12, 13, 14]
# cs = [16, 7, 8, 10, 10, 7, 9, 5, 9, 7, 12, 4, 11, 8, 10, 9, 9, 8, 8, 7, 10]
# plt.plot(error_level, scores, c='red', label="得分")
# plt.plot(error_level, rebounds, c='green', linestyle='--', label="篮板")
# plt.plot(error_level, assists, c='blue', linestyle='-.', label="助攻")
# plt.scatter(error_level, scores, c='red')
# plt.scatter(error_level, rebounds, c='green')
# plt.scatter(error_level, assists, c='blue')
# plt.legend(loc='best')
# plt.yticks(range(0, 50, 5))
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlabel("Error level α (years)", fontdict={'size': 16})
# plt.ylabel("Cumulative Score (%)", fontdict={'size': 16})
# plt.show()


'''对比实验'''
# CS
error_level = [1,2,3,4,5,6,7,8,9,10]
cs1 = [16.071, 30.357, 45.536, 51.786, 66.071, 73.214, 83.929, 85.714, 90.179, 91.964]  # Uncertainty
cs2 = [16.964, 38.393, 50.893, 62.5, 75.0, 80.357, 83.036, 87.5, 91.071, 93.75]  # 2D-slice-lstm
cs3 = [15.18, 39.29, 54.46, 66.96, 74.11, 80.36, 85.71, 91.96, 93.75, 96.43]  # TSAN
cs4 = [7.143, 10.714, 17.857, 26.786, 36.607, 42.857, 51.786, 58.929, 64.286, 66.071]  # 2D-slice-max
cs5 = [19.643, 36.607, 49.107, 58.929, 68.75, 76.786, 80.357, 85.714, 88.393, 90.179]  # 2D-slice-mean
#cs6 = [25.0, 41.964, 58.929, 73.214, 80.357, 85.714, 90.179, 91.071, 91.071, 93.75]  # Encoder+AggregationBlock+CosineSimilarity
cs6 = [23.464, 48.603, 66.480, 78.212, 87.151, 93.296, 96.089, 98.324, 98.324, 99.441]
plt.plot(error_level, cs1, label="Hepp et al.")
plt.plot(error_level, cs2, label="Lam et al.")
plt.plot(error_level, cs3, label="Cheng et al.")
plt.plot(error_level, cs4, label="2D-slice-max")
plt.plot(error_level, cs5, label="2D-slice-mean")
plt.plot(error_level, cs6, label="Our Method")
plt.scatter(error_level, cs1)
plt.scatter(error_level, cs2)
plt.scatter(error_level, cs3)
plt.scatter(error_level, cs4)
plt.scatter(error_level, cs5)
plt.scatter(error_level, cs6)
plt.xticks(range(1, 11, 1))
plt.xlabel("Error level α (years)")
plt.ylabel("Cumulative Score (%)")
plt.legend(loc='lower right')
plt.show()


# MCS
error_level = [1,2,3,4,5,6,7,8,9,10]
mcs1 = [8.036, 15.476, 22.991, 28.75, 34.97, 40.434, 45.87, 50.298, 54.286, 57.711] # Uncertainty
mcs2 = [8.482, 18.452, 26.562, 33.75, 40.625, 46.301, 50.893, 54.96, 58.571, 61.769] # 2D-slice-lstm
mcs3 = [7.59, 18.16, 27.23, 35.18, 41.67, 47.19, 52.01, 56.45, 60.18, 63.47] # TSAN
mcs4 = [3.572, 5.952, 8.928, 12.5, 16.518, 20.281, 24.219, 28.075, 31.697, 34.821] # 2D-slice-max
mcs5 = [9.822, 18.75, 26.339, 32.857, 38.839, 44.26, 48.772, 52.877, 56.429, 59.497] # 2D-slice-mean
#mcs6 = [12.5, 22.321, 31.473, 39.821, 46.577, 52.168, 56.92, 60.714, 63.75, 66.477] # Encoder+AggregationBlock+CosineSimilarity
mcs6 = [11.732, 24.022, 34.637, 43.352, 50.652, 56.744, 61.662, 65.735, 68.994, 71.762]
plt.plot(error_level, mcs1, label="Hepp et al.")
plt.plot(error_level, mcs2, label="Lam et al.")
plt.plot(error_level, mcs3, label="Cheng et al.")
plt.plot(error_level, mcs4, label="2D-slice-max")
plt.plot(error_level, mcs5, label="2D-slice-mean")
plt.plot(error_level, mcs6, label="Our Method")
plt.scatter(error_level, mcs1)
plt.scatter(error_level, mcs2)
plt.scatter(error_level, mcs3)
plt.scatter(error_level, mcs4)
plt.scatter(error_level, mcs5)
plt.scatter(error_level, mcs6)
plt.xticks(range(1, 11, 1))
plt.xlabel("Error level α (years)")
plt.ylabel("Mean Cumulative Score (%)")
plt.legend(loc='lower right')
plt.show()

'''多中心'''
# CS
# error_level = [1,2,3,4,5,6]
# cs1 = [23.464, 48.603, 66.480, 78.212, 87.151, 93.296]  # Encoder
# cs2 = [14.286, 33.036, 52.679, 69.643, 83.036, 86.607]  # Encoder+AggregationBlock
# cs3 = [22.707, 41.071, 60.699, 71.616, 79.039, 86.9]  # Encoder+CosineSimilarity
# plt.plot(error_level, cs1, c='red', linestyle=':',label="Spine")
# plt.plot(error_level, cs2, c='green', linestyle='--', label="Weifang_Spine")
# plt.plot(error_level, cs3, c='blue', linestyle='-.', label="Zhejiang_Spine")
# plt.scatter(error_level, cs1, c='red')
# plt.scatter(error_level, cs2, c='green')
# plt.scatter(error_level, cs3, c='blue')
# plt.xticks(range(1, 7, 1))
# plt.xlabel("Error level α (years)")
# plt.ylabel("Cumulative Score (%)")
# plt.legend(loc='lower right')
# plt.show()


# MCS
error_level = [1,2,3,4,5,6]
mcs1 = [11.732, 24.022, 34.637, 43.352, 50.652, 56.744] # Encoder
mcs2 = [7.143, 15.774, 25.0, 33.929, 42.113, 48.47] # Encoder+AggregationBlock
mcs3 = [11.354, 21.543, 31.332, 39.389, 45.997, 51.84] # Encoder+CosineSimilarity

plt.plot(error_level, mcs1, c='red', linestyle=':',label="Spine")
plt.plot(error_level, mcs2, c='green', linestyle='--', label="Weifang_Spine")
plt.plot(error_level, mcs3, c='blue', linestyle='-.', label="Zhejiang_Spine")
plt.scatter(error_level, mcs1, c='red')
plt.scatter(error_level, mcs2, c='green')
plt.scatter(error_level, mcs3, c='blue')
plt.xticks(range(1, 7, 1))
plt.xlabel("Error level α (years)")
plt.ylabel("Mean Cumulative Score (%)")
plt.legend(loc='lower right')
plt.show()









# '''消融实验'''
# # CS
# error_level = [1,2,3,4,5,6,7,8,9,10]
# cs1 = [15.179, 31.25, 50.0, 63.393, 75.893, 83.036, 85.714, 89.286, 91.071, 92.857]  # Encoder
# cs2 = [23.214, 39.286, 51.786, 66.964, 73.214, 77.679, 83.929, 88.393, 93.75, 93.75]  # Encoder+AggregationBlock
# cs3 = [26.786, 41.071, 55.357, 68.75, 75.893, 83.036, 85.714, 89.286, 91.071, 91.964]  # Encoder+CosineSimilarity
# cs4 = [25.0, 41.964, 58.929, 73.214, 80.357, 85.714, 90.179, 91.071, 91.071, 93.75]  # Encoder+AggregationBlock+CosineSimilarity
# plt.plot(error_level, cs1, c='red', linestyle=':',label="Encoder")
# plt.plot(error_level, cs2, c='black', linestyle='--', label="Encoder+AM")
# plt.plot(error_level, cs3, c='blue', linestyle='-.', label="Encoder+ASSL")
# plt.plot(error_level, cs4, c='green', label="Encoder+AAM+ASSL")
# plt.scatter(error_level, cs1, c='red')
# plt.scatter(error_level, cs2, c='black')
# plt.scatter(error_level, cs3, c='blue')
# plt.scatter(error_level, cs4, c='green')
# plt.xticks(range(1, 11, 1))
# plt.xlabel("Error level α (years)")
# plt.ylabel("Cumulative Score (%)")
# plt.legend(loc='lower right')
# plt.show()


# # MCS
# error_level = [1,2,3,4,5,6,7,8,9,10]
# mcs1 = [7.59, 15.476, 24.107, 31.964, 39.286, 45.536, 50.558, 54.861, 58.482, 61.607] # Encoder
# mcs2 = [11.607, 20.833, 28.572, 36.25, 42.411, 47.449, 52.009, 56.052, 59.822, 62.906] # Encoder+AggregationBlock
# mcs3 = [13.393, 22.619, 30.804, 38.393, 44.643, 50.128, 54.576, 58.433, 61.696, 64.448] # Encoder+CosineSimilarity
# mcs4 = [12.5, 22.321, 31.473, 39.821, 46.577, 52.168, 56.92, 60.714, 63.75, 66.477] # Encoder+AggregationBlock+CosineSimilarity
# plt.plot(error_level, mcs1, c='red', linestyle=':',label="Encoder")
# plt.plot(error_level, mcs2, c='black', linestyle='--', label="Encoder+AM")
# plt.plot(error_level, mcs3, c='blue', linestyle='-.', label="Encoder+ASSL")
# plt.plot(error_level, mcs4, c='green', label="Encoder+AM+ASSL")
# plt.scatter(error_level, mcs1, c='red')
# plt.scatter(error_level, mcs2, c='black')
# plt.scatter(error_level, mcs3, c='blue')
# plt.scatter(error_level, mcs4, c='green')
# plt.xticks(range(1, 11, 1))
# plt.xlabel("Error level α (years)")
# plt.ylabel("Mean Cumulative Score (%)")
# plt.legend(loc='lower right')
# plt.show()



# # -----crop与ori实验-----
# # CS
# x_data = [1,2,3,4,5,6,7,8,9,10]
# y1_data = [25.0, 41.964, 58.929, 73.214, 80.357, 85.714, 90.179, 91.071, 91.071, 93.75] # crop
# y2_data = [18.75, 34.821, 47.321, 58.036, 66.071, 75.893, 83.929, 88.393, 92.857, 94.643] # ori
# x1_width = range(0, len(x_data))
# x2_width = [i + 0.3 for i in x1_width]
# plt.bar(x1_width, y1_data, lw=0.5, width=0.3, label="健康人群")
# plt.bar(x2_width, y2_data, lw=0.5, width=0.3, label="患病人群")
# plt.xticks(np.arange(len(x_data))+0.15, x_data)
# plt.xlabel('Error level α (years)')
# plt.ylabel('Cumulative Score (%)')
# plt.legend(prop="SimHei", loc="upper left")
# plt.show()


# # MCS
# x_data = [1,2,3,4,5,6,7,8,9,10]
# y1_data = [12.5, 22.321, 31.473, 39.821, 46.577, 52.168, 56.92, 60.714, 63.75, 66.477] # crop
# y2_data = [9.375, 17.857, 25.223, 31.786, 37.5, 42.985, 48.103, 52.579, 56.607, 60.065] # ori
# x1_width = range(0, len(x_data))
# x2_width = [i + 0.3 for i in x1_width]
# plt.bar(x1_width, y1_data, lw=0.5, width=0.3, label="健康人群")
# plt.bar(x2_width, y2_data, lw=0.5, width=0.3, label="患病人群")
# plt.xticks(np.arange(len(x_data))+0.15, x_data)
# plt.xlabel('Error level α (years)')
# plt.ylabel('Mean Cumulative Score (%)')
# plt.legend(prop="SimHei", loc="upper left")
# plt.show()


# # -----123与45实验-----
# # CS
# x_data = [1,2,3,4,5,6,7,8,9,10]
# y1_data = [25.0, 41.964, 58.929, 73.214, 80.357, 85.714, 90.179, 91.071, 91.071, 93.75] # 123
# y2_data = [15.179, 26.786, 41.071, 54.464, 66.071, 74.107, 78.571, 85.714, 86.607, 91.071] # 45
# x1_width = range(0, len(x_data))
# x2_width = [i + 0.3 for i in x1_width]
# # plt.bar(x1_width, y1_data, lw=0.5, width=0.3, label="L1-L3")
# # plt.bar(x2_width, y2_data, lw=0.5, width=0.3, label="L4-L5")
# plt.bar(x1_width, y1_data, lw=0.5, width=0.3, label="健康人群")
# plt.bar(x2_width, y2_data, lw=0.5, width=0.3, label="患病人群")
# plt.xticks(np.arange(len(x_data))+0.15, x_data)
# plt.xlabel('Error level α (years)')
# plt.ylabel('Cumulative Score (%)')
# plt.legend(prop="SimHei", loc="upper left")
# plt.show()


# # MCS
# x_data = [1,2,3,4,5,6,7,8,9,10]
# y1_data = [12.5, 22.321, 31.473, 39.821, 46.577, 52.168, 56.92, 60.714, 63.75, 66.477] # 123
# y2_data = [7.59, 13.988, 20.759, 27.5, 33.928, 39.668, 44.531, 49.107, 52.857, 56.331] # 45
# x1_width = range(0, len(x_data))
# x2_width = [i + 0.3 for i in x1_width]
# # plt.bar(x1_width, y1_data, lw=0.5, width=0.3, label="L1-L3")
# # plt.bar(x2_width, y2_data, lw=0.5, width=0.3, label="L4-L5")
# plt.bar(x1_width, y1_data, lw=0.5, width=0.3, label="健康人群")
# plt.bar(x2_width, y2_data, lw=0.5, width=0.3, label="患病人群")
# plt.xticks(np.arange(len(x_data))+0.15, x_data)
# plt.xlabel('Error level α (years)')
# plt.ylabel('Mean Cumulative Score (%)')
# plt.legend(prop="SimHei", loc="upper left")
# plt.show()


# # -----123与45实验-----
# # CS
# x_data = [1,2,3,4,5,6,7,8,9,10]
# y1_data = [25.0, 41.964, 58.929, 73.214, 80.357, 85.714, 90.179, 91.071, 91.071, 93.75] # 123
# y2_data = [15.179, 26.786, 41.071, 54.464, 66.071, 74.107, 78.571, 85.714, 86.607, 91.071] # 45
# y3_data = [14.286, 33.036, 52.679, 69.643, 83.036, 86.607, 91.071, 92.857, 94.643, 95.536] # 12345
# x1_width = range(0, len(x_data))
# x2_width = [i + 0.3 for i in x1_width]
# x3_width = [i + 0.3 for i in x2_width]
# plt.bar(x1_width, y1_data, lw=0.5, width=0.3, label="L1-L3")
# plt.bar(x2_width, y2_data, lw=0.5, width=0.3, label="L4-L5")
# plt.bar(x3_width, y3_data, lw=0.5, width=0.3, label="L1-L5")
# plt.xticks(np.arange(len(x_data))+0.3, x_data)
# plt.xlabel('Error level α (years)')
# plt.ylabel('Cumulative Score (%)')
# plt.legend(prop="SimHei", loc="upper left")
# plt.show()


# # MCS
# x_data = [1,2,3,4,5,6,7,8,9,10]
# y1_data = [12.5, 22.321, 31.473, 39.821, 46.577, 52.168, 56.92, 60.714, 63.75, 66.477] # 123
# y2_data = [7.59, 13.988, 20.759, 27.5, 33.928, 39.668, 44.531, 49.107, 52.857, 56.331] # 45
# y3_data = [7.143, 15.774, 25.0, 33.929, 42.113, 48.47, 53.795, 58.135, 61.786, 64.854] # 12345
# x1_width = range(0, len(x_data))
# x2_width = [i + 0.3 for i in x1_width]
# x3_width = [i + 0.3 for i in x2_width]
# plt.bar(x1_width, y1_data, lw=0.5, width=0.3, label="L1-L3")
# plt.bar(x2_width, y2_data, lw=0.5, width=0.3, label="L4-L5")
# plt.bar(x3_width, y3_data, lw=0.5, width=0.3, label="L1-L5")
# plt.xticks(np.arange(len(x_data))+0.3, x_data)
# plt.xlabel('Error level α (years)')
# plt.ylabel('Mean Cumulative Score (%)')
# plt.legend(prop="SimHei", loc="upper left")
# plt.show()


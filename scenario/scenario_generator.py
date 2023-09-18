import random
import numpy as np
import matplotlib.pyplot as plt
import scipy


def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break 
    return item 


def curve(cur_y):
    curve_step = random_pick(list(range(8, 16)), [1 / 9] * 9)
    curve_strength = random_pick(list(range(15, 40)), [1 / 26] * 26)
    is_curve_append = random_pick([True, False], [0.5, 0.5])
    i = 0
    t_ys = []
    while i < curve_step:
        sin_pos = np.sin(i * np.pi / (curve_step - 1))
        if not is_curve_append:
            sin_pos = 0 - sin_pos
        t_ys.append(cur_y + sin_pos * curve_strength)
        cur_y = cur_y + sin_pos * curve_strength
        i += 1
    return curve_step, t_ys, cur_y

def gen_one(idx):
    
    xs = [0.0, 15.0, 30.0, 45.0, 60.0, 99.1086, 137.254, 173.498, 206.946, 236.777, 266.607, 300.056, 336.299, 374.445, 413.553, 428.553, 443.553, 458.553, 473.553, 488.553, 503.553, 518.553, 533.553, 548.553, 587.641, 625.857, 662.1, 695.549, 728.998, 758.829, 792.277, 828.521, 866.666, 881.666, 896.666, 911.553, 926.553, 986.1086, 1024.254, 1060.498, 1093.946, 1123.777, 1153.607, 1187.056, 1223.299, 1261.445, 1300.553, 1315.553, 1330.553, 1345.553, 1384.641, 1422.857, 1459.1, 1492.549, 1525.998, 1555.829, 1589.277, 1625.521, 1663.666, 1693.666, 1723.666]

    ys = []
    cur_y = 0

    while len(ys) < len(xs):
        
        start_curve = random_pick([True, False], [0.3, 0.7])
        if start_curve:
            curve_step, t_ys, cur_y = curve(cur_y)
            ys.extend(t_ys)
            
        else:
            ys.append(cur_y)
            
    if len(ys) > len(xs):
        ys = ys[:len(xs)]
    

    plt.figure('Draw')
    
    plt.plot(ys,xs)  # plot绘制折线图
    
    plt.show()  # 显示绘图

    arr = []
    for i in list(range(len(xs))):
        arr.append([xs[i], ys[i], 0])
    scipy.io.savemat('./scenario/GenscenarioData/roadCenters' + str(idx) + '.mat', 
                    {'roadCenters': np.array(arr)})
        

    i = 0
    ys_shake = []
    while i < len(ys):
        is_shake = random_pick([True, False], [0.4, 0.6])
        if is_shake:
            shake = random.uniform(-5, 5)
            ys_shake.append(ys[i] + shake)
        else:
            ys_shake.append(ys[i])
        i += 1
    
    arr1 = []
    for i in list(range(len(xs))):
        arr1.append([xs[i], ys_shake[i], 0])
    scipy.io.savemat('./scenario/GenscenarioData/waypoints' + str(idx) + '.mat', 
                    {'waypoints': np.array(arr)})
        
if __name__ == '__main__':
    for i in list(range(10)):
        gen_one(i)
from simso.generator.task_generator import *
import pandas as pd
import numpy as np
import time

taskset_count = 1000
time_step = 25
max_mul = 40
mul_range = range(1, max_mul + 1)
utilization_sum = 0.7
task_count = 20
is_HI_possibility = 0.5
criticality_factor = 2

time_selections = [time_step * x for x in list(mul_range)]

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break 
    return item 

if __name__ == "__main__":
    
    df = pd.DataFrame(columns=['taskset', 'name', 'wcet', 'wcet_high', 'period', 'criticality'])
    
    utilizations_all = gen_uunifastdiscard(taskset_count, utilization_sum, task_count)
    
    taskset = 0
    for utilizations_set in utilizations_all:
        
        name_idx = 0
        
        for utilization in utilizations_set:
            criticality = random_pick(["LO", "HI"], [1 - is_HI_possibility, is_HI_possibility])
            period = random_pick(time_selections, [1 / max_mul] * max_mul)
            wcet = period * utilization
            wcet_high = wcet * criticality_factor if criticality == "HI" else wcet
            df.loc[len(df)] = [taskset, "T" + str(name_idx), wcet, wcet_high, period, criticality]
            
            name_idx += 1
            
        
        taskset += 1
        
    df.to_csv("./randomdata/randomdata-" + str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())) + ".csv")
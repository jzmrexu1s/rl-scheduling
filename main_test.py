from test import main
import pandas as pd
from simso.generator.task_generator import gen_uunifastdiscard
import json

# print(gen_uunifastdiscard(1, 0.5, 5))
result = {}
df = pd.read_csv('result.csv',sep=',')
print(df.head())
scheduler_classes = [
    "simso.schedulers.EDF_VD_mono_new", "simso.schedulers.EDF_VD_mono_LA_RL", "simso.schedulers.EDF_VD_mono_LA_maxQoS", "simso.schedulers.EDF_VD_mono_LA", 
    "simso.schedulers.EDF_VD_mono_LA_between_random", "simso.schedulers.EDF_VD_mono_LA_between_uni"
]
# scheduler_classes = [
#     "simso.schedulers.EDF_VD_mono_LA_RL"
# ]

for cls in scheduler_classes:
    
    jobs_count, aborted_jobs_count, terminated_jobs_count, power, efficiency, overhead, time_series, power_series, terminate_series = main([], cls, 1)
    
    result[cls.split('.')[2]] = [time_series, power_series, terminate_series]
    
    new_data = pd.DataFrame({
        'name': [cls.split('.')[2]],
        'jobs_count': [jobs_count],
        'aborted_jobs_count': [aborted_jobs_count],
        'terminated_jobs_count': [terminated_jobs_count],
        'power': [power],
        'efficiency': [efficiency],
        'overhead': [overhead]
    })
    
    df = pd.concat([df, new_data], ignore_index=True)
    
    # for i in [j * 10 for j in range(1, 50)]:
    #     o = 0
    #     for k in range(0, 4):
    
    #         jobs_count, aborted_jobs_count, terminated_jobs_count, power, efficiency, overhead = main([], cls, i)
    #         o += overhead
            
    #         if k == 3:
    #             new_data = pd.DataFrame({
    #                 'name': [cls.split('.')[2]],
    #                 'jobs_count': [jobs_count],
    #                 'aborted_jobs_count': [aborted_jobs_count],
    #                 'terminated_jobs_count': [terminated_jobs_count],
    #                 'power': [power],
    #                 'efficiency': [efficiency],
    #                 'overhead': [o / 3]
    #             })
            
    #             df = pd.concat([df, new_data], ignore_index=True)
        
        
df = pd.concat([df, pd.DataFrame({
        'name': [''],
        'jobs_count': [0],
        'aborted_jobs_count': [0],
        'terminated_jobs_count': [0],
        'power': [0],
        'efficiency': [0],
        'overhead': [0]
    })], ignore_index=True)
df.to_csv('result.csv', index=False)
with open('result.json', 'w') as f:
    json.dump(result, f)
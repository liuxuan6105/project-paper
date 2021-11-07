from google.colab import drive
drive.mount('drive/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import OrderedDict
import os
from pathlib import Path
import met

"""## Path n Code prep"""
# fuzzy code
fuzzy_code = ''
# split code
split_code = ''
# Path for results
path=''
Path(path).mkdir(parents=True, exist_ok=True)
# path source data (train)
directory = ''

"""# Data Prep"""
data_trans_medis = []
data_trans_medis_kardio = []
data_trans_medis_nonkardio = []
kode = []
# looping files in directory
i=0
for filename in os.listdir(directory):
    if filename.startswith('data_fuzzy'+fuzzy_code) and filename.endswith('.xlsx') and 'train' in filename and 'trans' in filename and split_code in filename:
        folder = os.path.join(directory, filename)
        # k = folder.split('Transaksi Medis_(')
        # k = k[-1].split(').xlsx')[0]
        kode.append(fuzzy_code)
        # print(filename)
        temp_df = pd.read_excel(folder)        
        data_trans_medis_kardio.append(temp_df[temp_df['Diagnosis']=='Cardio'])
        data_trans_medis_nonkardio.append(temp_df[temp_df['Diagnosis']=='Normal'])        
        
iu=[
    [
      'Age: Young','Age: Adult', 'Age: Midle-Aged', 'Age: Mature', 'Age: Old', 
      'BMI: Severely Skinny', 'BMI: Skinny', 'BMI: Normal', 
      'BMI: Overweight', 'BMI: Low Risk Obese', 
      'BMI: Moderate Risk Obese', 'BMI: High Risk Obese', 
      'BP: Normal', 'BP: Elevated', 
      'BP: Low Risk Hypertension', 'BP: Moderate Risk Hypertension', 
      'BP: High Risk Hypertension',       
    ]
  ]

eu = {
      'Age: Young':1, 'Age: Adult':1,'Age: Midle-Aged':1, 'Age: Mature':3.27, 'Age: Old':3.14, 
      'BMI: Severely Skinny':2.42, 'BMI: Skinny':2.42, 'BMI: Normal':1, 
      'BMI: Overweight':.64, 'BMI: Low Risk Obese':.72, 
      'BMI: Moderate Risk Obese':.72, 'BMI: High Risk Obese':.72, 
      'BP: Normal':1, 'BP: Elevated':1.29,
      'BP: Low Risk Hypertension':2.59, 'BP: Moderate Risk Hypertension':2.59, 
      'BP: High Risk Hypertension':2.59,       
    }

data_trans_medis_kardio_tuple = []
data_trans_medis_nonkardio_tuple = []

for i in range(len(data_trans_medis_kardio)):
    temp_data = data_trans_medis_kardio[i].copy().drop(columns=['Id', 'Diagnosis'])
    temp_data['Symptomps'] = temp_data['Symptomps'].apply(lambda u: met.konvert(u))
    data_trans_medis_kardio_tuple.append(temp_data)
    
    temp_data_2 = data_trans_medis_nonkardio[i].copy().drop(columns=['Id', 'Diagnosis'])
    temp_data_2['Symptomps'] = temp_data_2['Symptomps'].apply(lambda u: met.konvert(u))
    data_trans_medis_nonkardio_tuple.append(temp_data_2)

data_trans_medis_kardio_tuple[0]['Symptomps']=data_trans_medis_kardio_tuple[0]['Symptomps'].apply(lambda x:met.case_1(x))
data_trans_medis_nonkardio_tuple[0]['Symptomps']=data_trans_medis_nonkardio_tuple[0]['Symptomps'].apply(lambda x:met.case_1(x))


"""# HURIM"""
T_running = []
list_running_time = []
util_sup = []
list_hasil = []
start_loop=time.time()
for minUtil in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for maxSup in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]:
        Profit_Table = eu
        for y in ['Cardio', 'Normal']:
            if y=='Cardio':
                mulai=time.time()
                df = data_trans_medis_kardio_tuple[0].copy()
                database_file = df
                problem = met.UPTree(database_file, Profit_Table,min_util=minUtil, max_sup=int(maxSup*len(database_file)))
                hasilnya = problem.solve_df()
                selesai=time.time()
                single_running = met.running_time(mulai, selesai)
                list_running_time.append(single_running)
                util_sup.append([[minUtil, maxSup, len(hasilnya)], hasilnya, y])
                list_hasil.append(hasilnya)
            else:
                mulai=time.time()
                df = data_trans_medis_nonkardio_tuple[0].copy()
                database_file = df
                problem = met.UPTree(database_file, Profit_Table,min_util=minUtil, max_sup=int(maxSup*len(database_file)))
                hasilnya = problem.solve_df()
                selesai=time.time()
                single_running = met.running_time(mulai, selesai)
                list_running_time.append(single_running)
                util_sup.append([[minUtil, maxSup, len(hasilnya)], hasilnya, y])
                list_hasil.append(hasilnya)

t_r = met.running_time(start_loop, time.time(), '\nTotal time: ')
T_running.append([t_r, list_running_time])

a, b, c, d = met.mySummary(util_sup, list_hasil)
d = np.transpose(d)

res_summary = pd.DataFrame({'Min Util':a,
                           'Max Sup':b,
                           'Number of HURI (itemset)':c,
                           'Running Time': list_running_time,
                           'Number of 1-itemset':d[0],
                           'Number of 2-itemset':d[1],
                           'Number of 3-itemset':d[2],
                           'Number of 4-itemset':d[3],
                           'Number of 5-itemset':d[4],
                           'Number of 6-itemset':d[5],
                           'Number of 7-itemset':d[6],
                           'Number of 8-itemset':d[7],
                           'Number of 9-itemset':d[8],
                           'Number of 10-itemset':d[9],
                          })

met.save_to_df_V1(util_sup, path, kat=kode[0])
met.save_to_df_V2(util_sup, path, kat=kode[0])
met.plot_total_rule(util_sup, path, x='Max Sup', hue='Min Util', kat=kode[0])
met.plot_total_rule(util_sup, path, kat=kode[0])


"""# Prediction"""
tes_huri = pd.read_excel('')
tes_data = pd.read_excel('')
tes_data.Symptomps = tes_data.Symptomps.apply(lambda x:met.input_to_case1(x))
tes_data_ = tes_data.copy()
tes_data_['HURI'], tes_data_['enc_huri'], tes_data_['Recognize Symptomps'], tes_data_['Unrecognize Symptomps'], tes_data_['Matched Symptomps'], tes_data_['Missmatched Symptomps'], tes_data_['Prediction'], tes_data_['Utility'], tes_data_['DTW Score'] =zip(*tes_data_['Symptomps'].apply(lambda x:met.my_pred(x, tes_huri)))

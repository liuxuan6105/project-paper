def cat_bp(ap_hi, ap_lo):
  '''
  this function to get blood pressure category from ap_hi and ap_lo values
  '''
  if ap_hi==1 and ap_lo==1:
    return 1
  elif ap_hi==2 and ap_lo==2:
    return 2
  elif (ap_hi==2 and ap_lo!=2) or (ap_hi!=2 and ap_lo==2) or (ap_hi==1 and ap_lo!=1) or (ap_hi!=1 and ap_lo==1) :
    return max(ap_hi, ap_lo)
  else:
    return min(ap_hi, ap_lo)

def trap_bound(feature_list, n):
  '''
  this function generates point/boundary as input into function 
  for fuzzification with trapezoidal membership function
  '''
  mid = []
  points = np.linspace(feature_list[0], feature_list[-1]+1,n*2)
  points = [round(point,2) for point in points]
  for k in range(1, len(points)-1, 2):
      mid.append(round((points[k]+points[k+1])/2,2))
  return points[1]-points[0], points, mid

def tri_bound(feature_list, n):
  '''
  this function generates point/boundary as input into function 
  for fuzzification with triangular membership function
  '''
  mid = []    
  points = np.linspace(feature_list[0], feature_list[-1]+1, n+2)
  points = [round(point,2) for point in points]
  for k in range(1, len(points)-2):
    mid.append(round((points[k]+points[k+1])/2,2))
  return points[1]-points[0], points, mid

def ga_bound(feature_list, n, feature_vis):
  '''
  this function generates point/boundary as input into function 
  for fuzzification with gaussian membership function
  '''
  mid = []      
  points = np.linspace(feature_list[0], feature_list[-1]+1,n)
  points = [round(point,2) for point in points]
  for k in range(len(points)-1):
    q = (points[k]+points[k+1])/2
    mid.append(round(q,2))
    pos = int((q-min(feature_list))*100)
    P = [p/1000 for p in range(0,10000)]
    if k==1:
      for p in P:
        cari_sigma = my.GaussMf(feature_vis, [p, points[k]])                    
        if cari_sigma[pos] <= 0.5:
          sigma=p
          break    
  return points[1]-points[0], points, mid, sigma

  
def generate_trans_data(data, percentage):
  '''
  This function is to convert the dataset to the form of medical transaction data

  Input:
    > data: Dataframe from preprocessing
    > percentage: percentage for tes data
  Output:
    medical transaction data (full, train, test)
  '''
  from sklearn.model_selection import train_test_split
  import pandas as pd
  import numpy as np
  
  data_trans_medis = data['Gender']+', '+data['Cholesterol']+', '+data['Gluc']+', '+data['Smoke']+', '+data['Active']+', '+data['Alco']+', '+data['Age']+', '+data['BMI']+', '+data['BP']
  data_trans_medis = data_trans_medis.to_list()
  diagnosis = data['Cardio'].to_list()
  id = ['TID_'+str(x+1) for x in range(len(data_trans_medis))]
  data_trans = pd.DataFrame({'Id': id, 'Symptomps': data_trans_medis, 'Diagnosis': diagnosis})
  train_data, tes_data = train_test_split(data_trans, 
                                          random_state=42,
                                          test_size=percentage)

  return data_trans, train_data, tes_data
  
  
def plot_correlation(title, df, col, kode='', path=path_fig):
  '''
  Custom function for plot correlation 
  '''
  plt.rcParams['figure.figsize'] = (8,16)
  g = sns.heatmap(df, annot=True, linewidths=.5, cmap=col, center=0, vmin=-1, vmax=1, annot_kws={"size": 16})
  g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 16) #, rotation=45, ha='right')
  g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 16) #, rotation=45)
  cbar = g.collections[0].colorbar    
  cbar.ax.tick_params(labelsize=18)
  plt.title(title, fontsize = 30)
  plt.tight_layout()
  plt.savefig(path_fig +title +kode +'.jpg', dpi=300)
  plt.show()
  plt.close()
  

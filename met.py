import queue
from functools import cmp_to_key
from collections import OrderedDict
import numpy as np	
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time


def konvert (list_item):
    temp_item = list_item.split(', ')
    temp_list = []
    for item in temp_item:
        for i in range(len(iu)):
            if item in iu[i]:
                temp_list.append((item, i+1))        
    return temp_list

def case_1(row):
  return row[-3:]
  
def running_time(start, end, argumen='Total running time:'):
    total_time = end - start
    total_time_s = total_time
    unit = 's'
    if total_time > 60 and total_time < 3600:
        total_time = total_time/60
        unit = 'minutes'
    if total_time > 3600:
        total_time = total_time/3600
        unit = 'hour'
    print('{} {} {}'.format(argumen, total_time, unit))
    return total_time_s

def myCountN(u):
    count = 1
    for i in u:
        if i == ',':
            count += 1
    return count

def SumOfCountN(n, huri):    
    res = 0
    if myCountN(huri) == n:
        res+=1
    return res

def mySummary(util_sup, list_hasil):
    _min_util = []
    _max_sup = []
    _total_huri = []
    _total_itemset = []   
    for i in range(len(util_sup)):    
        _min_util.append(util_sup[i][0][0])
        _max_sup.append(util_sup[i][0][1])
        _total_huri.append(util_sup[i][0][2])        
        _1 = _2 = _3 = _4 = _5 = _6 = _7 = _8 = _9 = _10 = _11 = _12 = 0        
        for j in range(len(util_sup[i][1])):
            _1 += SumOfCountN(1, util_sup[i][1][j][0])
            _2 += SumOfCountN(2, util_sup[i][1][j][0])
            _3 += SumOfCountN(3, util_sup[i][1][j][0])
            _4 += SumOfCountN(4, util_sup[i][1][j][0])
            _5 += SumOfCountN(5, util_sup[i][1][j][0])
            _6 += SumOfCountN(6, util_sup[i][1][j][0])
            _7 += SumOfCountN(7, util_sup[i][1][j][0])
            _8 += SumOfCountN(8, util_sup[i][1][j][0])
            _9 += SumOfCountN(9, util_sup[i][1][j][0])
            _10 += SumOfCountN(10, util_sup[i][1][j][0])
            _11 += SumOfCountN(11, util_sup[i][1][j][0])
            _12 += SumOfCountN(12, util_sup[i][1][j][0])        
        _total_itemset.append([_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12])        
    return _min_util, _max_sup, _total_huri, _total_itemset

def save_to_df_V1(util_sup, path, n, fuzzy_cat=''):
    u=1
    for i in [g for g in range(n) if g%2==0]:
        _util = []
        _sup = []
        _huri = []
        _total_huri = 0
        _diagnosis = []       
        for j in [i, i+1]:
            _total_huri += len(util_sup[j][1])        
            for k in range(len(util_sup[j][1])):
                _huri.append(util_sup[j][1][k][0])
                _diagnosis.append(util_sup[j][2])
                _util.append(util_sup[j][1][k][1])
                _sup.append(util_sup[j][1][k][2])
        result_df = pd.DataFrame({'HURI': _huri,
                                   'Diagnosis': _diagnosis,
                                   'Utility': _util,
                                   'Support': _sup})
        result_df['HURI Length'] = result_df['HURI'].apply(lambda u: myCountN(u))
        result_df = result_df[['HURI Length', 'HURI', 'Diagnosis', 'Utility', 'Support']]
        result_df = result_df.sort_values(['Support', 'Utility'], ascending=[False, False])
        result_df.to_excel(path+fuzzy_cat+'_'+str(u)+'_result_(min_util='+str(util_sup[i][0][0])+')_(max_sup='+str(util_sup[i][0][1])+')_(Total HURI='+str(_total_huri)+').xlsx', index=False)
        u+=1

def save_to_df_V2(util_sup, path, n, fuzzy_cat=''):    
    u=1
    for i in [g for g in range(n) if g%2==0]:
        _util = []
        _sup = []
        _huri = []
        _total_huri = 0
        _diagnosis = []        
        for j in [i, i+1]:
            _total_huri += len(util_sup[j][1])        
            for k in range(len(util_sup[j][1])):
                _huri.append(util_sup[j][1][k][0]+' => '+util_sup[j][2])
                _util.append(util_sup[j][1][k][1])
                _sup.append(util_sup[j][1][k][2])
        result_df = pd.DataFrame({'Rule': _huri,
                                   'Utility': _util,
                                   'Support': _sup})
        result_df['Rule Length'] = result_df['Rule'].apply(lambda u: myCountN(u))
        result_df = result_df[['Rule Length', 'Rule', 'Utility', 'Support']]
        result_df = result_df.sort_values(['Support', 'Utility'], ascending=[False, False])
        result_df.to_excel(path+fuzzy_cat+'_R_'+str(u)+'_result_(min_util='+str(util_sup[i][0][0])+')_(max_sup='+str(util_sup[i][0][1])+')_(Total HURI='+str(_total_huri)+').xlsx', index=False)
        u+=1

def plot_total_rule(util_sup, path, x='Min Util', hue='Max Sup', fuzzy_cat=''):
    plt.close()
    _huri = []
    _util = []
    _sup = []
    _total_rule = []
    for i in [g for g in range(200) if g%2==0]:
        jml_rule = 0        
        _sup.append(util_sup[i][0][1])
        _util.append(util_sup[i][0][0])        
        for j in [i, i+1]:
            jml_rule += len(util_sup[j][1])                        
        _total_rule.append(jml_rule)            
    result_df = pd.DataFrame({'Min Util': _util,
                               'Max Sup': _sup,
                               'Total HURI': _total_rule})    
    plt.rcParams['figure.figsize'] = (20,8)    
    sns.set_style("whitegrid")
    g_vis = sns.catplot(data=result_df, x=x, y='Total HURI', hue=hue, kind='bar')
    g_vis.set_ylabels('Total HURI')
    if hue=='Max Sup':      
      g_vis._legend.set_title(r'$\alpha$')
      g_vis.set_xlabels(r'$\beta$', fontsize=14)
    else:
      g_vis._legend.set_title(r'$\beta$')
      g_vis.set_xlabels(r'$\alpha$', fontsize=14)      
    g_vis.fig.suptitle('Comparison of the Number of HURI')
    # plt.tight_layout()
    plt.savefig(path+fuzzy_cat+'_Comparison Total Generated Rules  (Hue='+hue+').jpg', dpi=300)
	
class Node:
  name = ""	
  count = None	
  nu = None	
  parent = None
  hlink = None	
  children = None
  level = None
  mnu = None

  def __init__(self,name,parent = None,nu=0,mnu=999999999):
    self.name = name
    self.count = 1
    self.nu = nu
    self.parent = parent
    self.children = {}
    self.mnu = mnu
    if self.parent!=None:
      self.level = self.parent.level + 1
    else:
      self.level = 0

  def show(self, level=0, show=False):
    if (self.parent != None and show==True):
      print('.', end=' ')
      #print('([^{},{}], -> [{}({},{}), {}]). level: {}'.format(str(self.parent.name),
                                   #str(self.parent.nu),
                                   #self.name, self.nu, self.count, self.mnu,
                                   #level))
    if (self.parent != None and show==False):
      print('.', end=' ')
      #print('([^{},{}], -> [{}({},{}), {}]).'.format(str(self.parent.name),
                               #str(self.parent.nu),
                               #self.name, self.nu, self.count, self.mnu,))

  def insert_child_node(self,i,val,mnu):
    if i in list(self.children.keys()):
      node = self.children[i]
      node.count += 1
      node.nu += val
      node.mnu_awal = node.mnu
      node.mnu = min(node.mnu,mnu)
    else:
      self.children[i] = Node(i,self,val,mnu)
    return self.children[i]


class HeaderTable:
  table = None

  def __init__(self,items):
    self.table = {item : {"utility":0,"link":None,"last":None} for item in items}

  def show(self):
    print('\n  {:^54s}'.format('Header Table'))
    print('='*58)
    print('| {:^30s} | {:^14s} | {:^5s}|'.format("name","utility","link"))
    print('='*58)
    for item in self.table.keys():
      print('| {:30s} | {:14.2f} | {} |'.format(item, round(self.table[item]["utility"], 2), self.table[item]["link"]))
    print('='*58, end='\n\n')

  def increment_utility(self,item_name,increment):
    if item_name in list(self.table.keys()): 
      self.table[item_name]["utility"] += increment
      return True
    else:
      return False

  def dgu(self,min_util):
    self.table = {k: v for k, v in self.table.items() if v["utility"] >= min_util}

  def dlu(self,min_util):
    self.table = {k: v for k, v in self.table.items() if v["utility"] >= min_util}

  def dpred(self, tes_item):
    # print(tes_item)
    # print({k: v for k, v in self.table.items() if k in tes_item min_util})
    self.table = {k: v for k, v in self.table.items() if k in tes_item}
    # {key: self.table[key] for key in ky_lit}

        
class UPTree:
  item_set 				= None
  header_table 			= None
  tree_root				= None
  profit_hash 			= None
  min_util				= None
  max_sup 				= None #int(1*len(database_file))
  current_pattern_base	= ""
  infinity 				= 9999999
  database_file			= None
  profit_table			= None
  test_item = None

  def __init__(self,db=None,profit_hash=None,min_util=None, max_sup=None, tesItem=None):
    self.profit_hash 			= profit_hash
    if profit_hash != None:
      self.item_set 				= list(profit_hash.keys())
      self.header_table 			= HeaderTable(self.item_set)
    if min_util == None:
      self.min_util = 0
    self.min_util				= min_util
    self.max_sup				= max_sup#int(max_sup*len(database_file))
    self.tree_root				= Node("Root")
    self.database_file			= db
    self.profit_table			= profit_hash
    self.test_item = tesItem


  def from_patterns(self,pattern_base,min_util,x):
    self.current_pattern_base = x
    item_set = []
    for patterns in pattern_base:
      [pattern,support,cost] = patterns
      for [item,mnu] in pattern:
        if item not in item_set:
          item_set.append(item)
    self.item_set = item_set
    self.header_table = HeaderTable(self.item_set)
    self.min_util = min_util
    for patterns in pattern_base:
      [pattern,support,cost] = patterns
      for [item,mnu] in pattern:
        self.header_table.increment_utility(item,cost*support)
    self.header_table.table = dict(OrderedDict(sorted(self.header_table.table.items(), key=lambda x: x[1]['utility'], reverse=True)))
    self.header_table.dlu(self.min_util)
    for i in range(len(pattern_base)):
      [pattern,support,cost] = pattern_base[i]
      new_pattern = []
      for [item,mnu] in pattern:
        present = bool(item in list(self.header_table.table.keys()))
        if present:
          new_pattern.append([item,mnu])
        if not present:
          pattern_base[i][2] -= (mnu*support) 
      pattern_base[i][0] = new_pattern
    for i in range(len(pattern_base)):
      pattern_base[i][0] = sorted(pattern_base[i][0], key=cmp_to_key(lambda x,y: self.get_head_val(y) - self.get_head_val(x)))
    for patterns in pattern_base:
      [pattern,support,cost] = patterns
      if len(pattern)==0:
        continue
      current_node = self.tree_root
      sum_mnu_coming_after = 0 
      for [i,mnu] in pattern[1:]:
        sum_mnu_coming_after += mnu*support 
      current_val = cost - sum_mnu_coming_after 
      current_node = current_node.insert_child_node(pattern[0][0],current_val,pattern[0][1]) 
      for [item,mnu] in pattern[1:]:
        current_val += mnu*support 
        current_node = current_node.insert_child_node(item,current_val,mnu) 

  def get_head_val(self,item_mnu):
    [item,mnu] = item_mnu
    return self.header_table.table[item]["utility"]

  def calculate_tu(self,row): 
    Transaction_Utility = 0
    for item in row:
      item_name = item[0]
      quantity  = item[1]
      item_value = self.profit_table[item_name]*quantity
      Transaction_Utility += item_value
    return Transaction_Utility

  def insert_reorganized_transaction(self,transaction):
    current_node = self.tree_root
    current_val = 0
    for i in transaction:
      item = i[0]
      quantity = i[1]
      nu = self.profit_hash[item]*quantity
      current_val += nu
      current_node = current_node.insert_child_node(item,current_val,nu)

  def show_header_table(self):
    self.header_table.show()

  def dbscan_df(self):
    for u in range(self.database_file.shape[0]):
      tu = self.calculate_tu(self.database_file.iloc[u]['Symptomps'])
      for item in self.database_file.iloc[u]['Symptomps']:
        self.header_table.increment_utility(item[0],tu)
    self.header_table.table = {k: v for k, v in self.header_table.table.items() if v["utility"] > 0}    
    self.header_table.table = dict(OrderedDict(sorted(self.header_table.table.items(), key=lambda x: x[1]['utility'], reverse=True)))
    if len(list(self.header_table.table.keys())) > 0:
      self.min_util = self.min_util * self.header_table.table[list(self.header_table.table.keys())[0]]['utility']

  def reorganized_dbscan_dgn_df(self,show=False):    
    for u in range(self.database_file.shape[0]):
      filtered_row = []
      for item in self.database_file.iloc[u]['Symptomps']:
        if item[0] in list(self.header_table.table.keys()):
          filtered_row.append(item)        
      self.database_file.iloc[u]['Symptomps'] = sorted(filtered_row, 
                                                     key=cmp_to_key(lambda x,y: self.header_table.table[y[0]]["utility"] - self.header_table.table[x[0]]["utility"]))
      tu = self.calculate_tu(self.database_file.iloc[u]['Symptomps'])
      self.insert_reorganized_transaction(sorted(filtered_row,
                                                       key=cmp_to_key(lambda x,y: self.header_table.table[y[0]]["utility"] - self.header_table.table[x[0]]["utility"])))

      if(show):
        if self.database_file.shape[0] < 10:
          print('{}. {:180s} | (TU: {})'.format(u+1, str(self.database_file.iloc[u]['Symptomps']), tu))
        else:
          if u<3:
            print('{}. {:180s} | (TU: {})'.format(u+1, str(self.database_file.iloc[u]['Symptomps']), tu))
          elif u in range(self.database_file.shape[0]-3,self.database_file.shape[0]):
            print('{}. {:180s} | (TU: {})'.format(u+1, str(self.database_file.iloc[u]['Symptomps']), tu))
          elif u in np.linspace(3, self.database_file.shape[0]-2, 10, dtype=int):
            print('......')

  def dgu(self):
    self.header_table.dgu(self.min_util)

  def d_pred(self):
    # print(self.test_item)
    self.header_table.dpred(self.test_item)

  def show_tree(self):
    q = queue.Queue()
    current_level = 0
    q.put(self.tree_root)
    member_lvl_count={}
    while not q.empty():
      n = q.get()
      if(n.level!=current_level):
        current_level=n.level
      if current_level not in member_lvl_count:
        member_lvl_count[current_level]=1
      else:
        member_lvl_count[current_level]+=1
      if(n.name!="Root"):
        if(n.name not in self.header_table.table.keys()):
          continue
        elif (self.header_table.table[n.name]["link"]==None):
          self.header_table.table[n.name]["link"] = n
          self.header_table.table[n.name]["last"] = n
        else:
          self.header_table.table[n.name]["last"].hlink = n
          self.header_table.table[n.name]["last"] = n
      for child_node_name in list(n.children.keys()):
        q.put(n.children[child_node_name])

  def hurim_upraregrowth(self):
    phui = []
    urutan = list(self.header_table.table.keys())
    urutan.reverse()
    for item in urutan: 
      if(self.header_table.table[item]["utility"]>self.min_util):
        item_potential_value = 0
        huri = []
        sup_list = []
        current = self.header_table.table[item]["link"]
        if(current != None):
          sup=0
          while(True):
            item_potential_value += current.nu
            pb =[[],0,0] 
            pb[1] = current.count
            pb[2] = current.nu
            sup_list.append(current.count)
            up = current.parent
            while(up.parent!=None):
              pb[0].append([up.name,up.mnu])
              up = up.parent                
            sup += pb[1]
            if len(pb[0])!=0:
              huri.append(pb)
            if(current.hlink == None):
              break
            current = current.hlink
          if(item_potential_value>self.min_util and (0 < sup < self.max_sup)):
            phui.append([item,item_potential_value, sup]) 
        tree = UPTree(min_util=self.min_util, max_sup=self.max_sup)
        tree.from_patterns(huri,self.min_util,self.current_pattern_base+item)
        tree.show_tree()
        if all([t > self.max_sup for (t) in sup_list]):
          continue
        else:
          retreived = tree.hurim_upraregrowth()
          for i in retreived:
            phui.append([item+', '+i[0],i[1],i[2]])
    return phui

  def solve_df(self):
#     print('. ', end=' ')		
    self.dbscan_df()
    # self.show_header_table()
#     print('. ', end=' ')		
    self.dgu()
    #self.show_header_table()
    print('. ', end=' ')		
    self.reorganized_dbscan_dgn_df()
    self.show_tree()
#     print('. ', end=' ')		
    return self.hurim_upraregrowth()
	
	
def encode_diag(diag):
  if diag=='Cardio':
    return 1
  return 0
  
  
def input_to_case1(row):
  row = row.split(', ')
  return ', '.join(row[-3:])  

  
def my_pred(symp_seqs, data_huri):
  x = symp_seqs.split(', ')
  list_enc_huri, list_dtw_score = [], []
  list_recognize_symps, list_matched_symps = [], []
  list_missmatched_symps, list_unrecognize_symps= [], []  
  for y_idx in range(data_huri.shape[0]):  
    y = data_huri.iloc[y_idx]['HURI'].split(', ')
    y1 = data_huri.iloc[y_idx]['Utility']
    y2 = data_huri.iloc[y_idx]['Diagnosis']	
    # list encoding huri
    enc_huri = []
    # intialize list matched and missmatched symps
    matched_symps = []
    missmatched_symps = []
    # Iterate over symp seq.
    for symp_seq in x:
      # get single symptomps and its category
      symp, cat = symp_seq.split(': ')  
      # Iterate over huri seq.
      for huri_seq in y:
        # get single huri and its category
        huri, cat_ = huri_seq.split(': ')
        # matching to the symptmps 
        if huri == symp and cat_ == cat:      
          enc_huri.append(1)
          matched_symps.append(huri+': '+cat_)          
          break
        if huri == symp and cat_ != cat:      
          enc_huri.append(0)
          missmatched_symps.append(huri+': '+cat_)          
          break   
    # get rec and unrec symps
    x_ = [symp.split(': ')[0] for symp in x]
    y_ = [symp.split(': ')[0] for symp in y]
    recognize_symps = ', '.join([symp for symp in x_ if symp in y_])
    unrecognize_symps = ', '.join([symp for symp in x_ if symp not in y_])
    # encode symp
    enc_data = [1]*len(x)
    # calculate dtw score
    dtw_score = fastdtw(enc_data, enc_huri, dist=euclidean)[0]
    if len(enc_huri)==0:
      dtw_score=9999999999
    list_dtw_score.append(dtw_score)
    # change matched & missmatched symps to str
    matched_symps = ', '.join(matched_symps)
    missmatched_symps = ', '.join(missmatched_symps)
    # add item to lists
    list_enc_huri.append(enc_huri)
    list_recognize_symps.append(recognize_symps)
    list_unrecognize_symps.append(unrecognize_symps)
    list_matched_symps.append(matched_symps)
    list_missmatched_symps.append(missmatched_symps)  
  # get min. index
  idx_min = min(range(len(list_dtw_score)), key=list_dtw_score.__getitem__)  
  # get best prediction info
  y = data_huri.iloc[idx_min]['HURI']
  y1 = data_huri.iloc[idx_min]['Utility']
  y2 = data_huri.iloc[idx_min]['Diagnosis']
  enc_huri = list_enc_huri[idx_min]
  recognize_symps = list_recognize_symps[idx_min]
  unrecognize_symps = list_unrecognize_symps[idx_min]
  matched_symps = list_matched_symps[idx_min]
  missmatched_symps = list_missmatched_symps[idx_min]
  dtw_score = list_dtw_score[idx_min]
  return y, enc_huri, recognize_symps, unrecognize_symps, matched_symps, missmatched_symps, y2, y1, dtw_score
  

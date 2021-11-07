'''
This library is used to plot the fuzzification results and the correlation between attributes after the fuzzification process.
'''

import matplotlib.pyplot as plt
import seaborn as sns

path_1 = ''
path_2 = ''


def garis(x, y):
  '''
  This function is to plot a line as a marker on
  X coordinates from Y=0 to Y=y.

  Parameter:
    x = coordinate X
    y = maximum line height desired

  Outputs:
    Plot the dotted line as the X-coordinate marker.
  '''
  plt.plot([x, x], [0, y], 'k--')
	
def garis_atas(x, y):
  '''
  This function is to plot a line as a marker at coordinate X
  from Y=y to Y=1.

  Parameter:
    x = coordinate X
    y = minimum desired line height

  Outputs:
    Plot the dotted line as the X-coordinate marker.
  '''
  plt.plot([x, x], [y, 1], 'k--')  
	
def mf (x, y, kategori, t_tengah, judul, kode, xlbl, titik,
    y_garis_bawah=[1], y_garis_atas=[0.56], y_text=[0.5],
    pisah_garis=[False], rot=[0], rot_xtick=0,
    simpan=True, path=path_1, lang='id'):
  '''
  Fungsi ini untuk plot himpunan fuzzy dengan fungsi keanggotaan dan
  range nilai yang diberikan.

  Parameter:
    x: range nilai pada sb. X
    y: list fungsi keanggotaan
    kategori: kategori untuk setiap fungsi keanggotaan
    t_tengah: koordinat X untuk setiap kategori
    judul: judul plot dan sekaligus sebagai tambahan pada nama file jika disimpan
    kode: tambahan kode untuk nama file jika ingin menyimpan hasil plot
    xlbl: keterangan sumbu X
    titik: kumpulan titik sebagai untuk membantu memperjelas fungsi keanggotaan
    y_garis_bawah: untuk garis penanda titik, sebagai parameter Y untuk 
             fungsi `garis` (default)=[1])
    y_garis_atas: untuk garis penanda titik, sebagai parameter Y untuk 
            fungsi `garis_atas` (default=[0.56])
    y_text: koordinat Y pada setiap kategori dari fungsi keanggotaan yang diberikan
        (default=[0.5])
    pisah_garis: parameter boolean untuk memisahkan garis vertikal penanda koordinat X
           pada setiap titik (default=[False])
    rot: untuk merotasi kategori (default=[0])
    rot_xtick: untuk merotasi titik pada sb. X (default=0)
    simpan: untuk menyimpan langsung hasil plot (default=False)
    path: directory untuk menyimpan	hasil plot

  Output:
      Line plot himpunan fuzzy.
  '''

  plt.close()	
  sns.set_style('white')
  warna = ['r', 'g', 'b','deeppink', 'k', 'm', 'c', 'darkorange', 'gray', 'chocolate']*10
  plt.figure(figsize=(10,6))	
  if y_garis_bawah == [1]:
    y_garis_bawah=[1]*(len(titik)-2)
  if y_garis_atas == [0.56]:
    y_garis_atas=[0.56]*(len(titik)-2)
  if pisah_garis == [False]:
    pisah_garis = [False]*(len(titik)-2)
  if y_text == [0.5]:
    y_text = [0.5]*(len(kategori))
  if rot == [0]:
    rot = [0]*(len(kategori))	
  for i in range(len(kategori)):
    plt.plot(x, y[i], warna[i], linewidth=2)
    plt.text(t_tengah[i], y_text[i], kategori[i], fontsize=18.5, rotation=rot[i]) #-len(kategori[i])/20
  plt.title(judul, fontsize=18)
  plt.ylim(0,1.05)
  plt.xlim(x[0],x[-1])
  for u in range(1, len(titik)-1):        
    if pisah_garis[u-1] is False:
      garis(titik[u], y_garis_bawah[u-1])
    else:
      garis_atas(titik[u], y_garis_atas[u-1])
      garis(titik[u], y_garis_bawah[u-1])			
  plt.tick_params(which='major', axis='both', labelsize=14)
  plt.tick_params(axis='x', rotation=rot_xtick) #baru
  plt.xticks(titik)
  if lang=='id':
    plt.ylabel('Derajat Keanggotaan', fontsize=16)
  else:
    plt.ylabel('Degree of Membership', fontsize=16)
  plt.xlabel(xlbl, fontsize=16)	
  plt.savefig(path+judul+kode+'.jpg', dpi=300)
  #     plt.savefig(path_2+judul+kode+'.jpg', dpi=300)

def korelasi(judul, datanya, warna, kode='', simpan=True, path=path_1):
  '''
  Fungsi ini untuk membuat plot heatmap korelasi dari data yang diberikan.

  Parameter:
    judul: judul untuk heatmap dan juga sebagai kode untuk nama file jika ingin disimpan langsung
    datanya: sumber data yang ingin dibuat heatmapnya
    warna: palette warna yang diinginkan untuk heatmap
    kode: kode sebagai tambahan nama file jika ingin disimpan langsung
    simpan: untuk menyimpan file (Default=False)
    path: path/directory untuk menyimpan hasil heatmpat (Default='')

  Output:
      Plot heatmap dari data yang diberikan.
  '''

  plt.close()
  plt.rcParams['figure.figsize'] = (16,10)	
  g = sns.heatmap(datanya, annot=True, linewidths=.5, cmap=warna, center=0, vmin=-1, vmax=1, annot_kws={"size": 16})
  g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 16, rotation=45, ha='right')
  g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 16, rotation=45)
  cbar = g.collections[0].colorbar    
  cbar.ax.tick_params(labelsize=18)
  plt.title(judul, fontsize = 30)
  plt.tight_layout()
  plt.savefig(path_1 +judul +kode +'.jpg', dpi=300)
    # plt.savefig(path_2 +'\\' +judul +'.png', dpi=300)
  plt.show()
  plt.close()

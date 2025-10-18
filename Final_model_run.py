import scipy.io
import torch
from GCN_model import DeepChebNet
from scipy.stats import pearsonr
import numpy as np
import math
import torch.nn as nn
import pandas as pd
from torch_geometric.data import Data



#Pearson mat
def pearson_mat(data): #data: HO atlas of the subject
    n = data.shape[1]  # Number of features
    corr_matrix = np.zeros((n, n))  # Initialize empty matrix

    for i in range(n):
        for j in range(n):
            corr_matrix[i, j],_ = pearsonr(data[:, i], data[:, j])
  
    #flattened upper triangular vector
    flattened_upper_triangular=np.array(corr_matrix[np.triu_indices(N, k=1)])
    feature_mask=scipy.io.loadmat('feature_mask_pearson.mat')['feature_mask'][0]
    #feature masking through rfe
    X_features=flattened_upper_triangular[np.bool_(feature_mask)]


    return X_features

#Euclidean similarity
def euclidean_similarity(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)  # Compute Euclidean distance
    similarity = 1 / (1 + distance)  # Convert to similarity (0 to 1)
    return abs(similarity)



#Constructing matrix based on similarity between two subjects

def get_value(X_selected):
  sig=0.07
  sim1_mat=np.zeros((X_selected.shape[0],X_selected.shape[0]))
  for i in range(len(X_selected)):
    for j in range(len(X_selected)):
      euc_dist=euclidean_similarity(X_selected[i],X_selected[j])
      sim1_mat[i][j]=math.exp(-(math.pow(euc_dist,2))/(2*math.pow(sig,2)))
  return sim1_mat

#For categorical types
def kronecker_cat(mat):
  out_mat=np.zeros((len(mat), len(mat)))
  for i in range (len(mat)):
    for j in range (len(mat)):
      if mat[i]==mat[j] and i!=j:
        out_mat[i][j]=1
      else:
        out_mat[i][j]=0
  return out_mat

#For continuous values
def kronecker_num(mat):
  thresh=7.6
  out_mat=np.zeros((len(mat), len(mat)))
  for i in range (len(mat)):
    for j in range (len(mat)):
      if (abs(mat[i]-mat[j]))<thresh and i!=j:
        out_mat[i][j]=1
      else:
        out_mat[i][j]=0
  return out_mat

#get phenotype data
def get_vals(file_id,df):
  # Step 1: Find the index label(s) where 'Name' is 'Bob'
  index_label = df[df['FILE_ID'] == file_id].index[0]

  # Step 2: Convert the index label to integer position
  position = df.index.get_loc(index_label)

  # Step 3: Use iloc to access the row
  row = df.iloc[position]

  # Access specific columns
  age = row['AGE_AT_SCAN']
  gen = row['SEX']
  site = row['SITE_ID']
  dx = row['DX_GROUP']

  return float(age), int(gen), site, int(dx)

def process_data(file_id,path=""):
  #get files
  df=pd.read_csv('asd_files.csv')
  train_data=scipy.io.loadmat('train_pearson.mat')['train_pearson']
  test_data=scipy.io.loadmat('test_pearson.mat')['test_pearson']
  val_data=scipy.io.loadmat('val_pearson.mat')['val_pearson']
  mat_data=np.concatenate((train_data,test_data,val_data), axis=0)

  pheno_gen=df['SEX'].tolist()
  pheno_age=df['AGE_AT_SCAN'].tolist()
  pheno_site=df['SITE_ID'].tolist()
  y=df['DX_GROUP'].tolist()

  #get subject's data
  age,sex,site, dx_class=get_vals(file_id, df)
  #sex=1
  #dx_class=1
  #age=14.53
  #file_id='NYU_0050999'
  sub_path=path+file_id+'_rois_ho.1D'
  data=np.loadtxt(sub_path)
  print('Patient Atlas data loaded')

  pheno_age.append(age)
  pheno_gen.append(sex)
  pheno_site.append(site)
  y.append(dx_class)
  y=[a-b for a,b in zip(y,[1]*len(y))]
  N=111
  #pearson matrix and selected feature vectors
  X=scipy.io.loadmat('all_data.mat')['X'].tolist()
  X.append(pearson_mat(data))
  #print( X)
  X=np.array(X)
  X_selected=X[:,:2052]
  print('Constructing Graph')

  #sim1
  sim1_mat=get_value(X_selected)
  sim1=np.zeros((X_selected.shape[0],X_selected.shape[0]))
  for i in range(len(sim1_mat)):
    for j in range(len(sim1_mat)):
      if i==j:
        sim1[i][j]=0
      else:
        '''if sim1_mat[i][j]<0.5:
          sim1[i][j]=0
        else:'''
        sim1[i][j]=sim1_mat[i][j]

  #sim2
  #for Gender
  pheno_gen_mat=kronecker_cat(pheno_gen)
  #for site
  pheno_site_mat=kronecker_cat(pheno_site)
  #for age
  pheno_age_mat=kronecker_num(pheno_age)

  sim2=pheno_mat=pheno_gen_mat+pheno_age_mat

  #c
  c_dash=sim1*sim2

  c=np.zeros((872,872))
  for i in range(len(c_dash)):
    for j in range(len(c_dash)):
      if i==j:
        c[i][j]=0
      else:
        if c_dash[i][j]<0.5:
          c[i][j]=0
        else:
          c[i][j]=c_dash[i][j]
  A=c

  #get weights and indices of edges
  edge_indices = np.array(np.nonzero(A))  # Shape: (2, num_edges)
  # Get edge weights corresponding to those edges
  edge_weights = A[edge_indices[0], edge_indices[1]]

  #converting numpy to tensors
  edge_indices=torch.tensor(edge_indices, dtype=torch.int64).to(device)
  edge_weights=torch.tensor(edge_weights, dtype=torch.float).to(device)
  X=torch.tensor(X_selected, dtype=torch.float).to(device)
  y=torch.tensor(y, dtype=torch.float).to(device)

  print('Graph Object Creation')
  data = Data(x=X, edge_index=edge_indices, edge_attr=edge_weights, y=y)
  print('Graph Initialization')
  #get_model and load checkpoint
  model = DeepChebNet(
    input_dim=2052,
    hidden_dims=[128, 128, 128, 128, 128, 128, 128],
    output_dim=128,
    K=3,
    dropout=0.2,
    dropedge_prob=0.3)
  print('Loading pretrained model')
  checkpoint = torch.load('checkpoints/checkpoint_epoch_Final.pth', map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  model.eval()

  with torch.no_grad():
    output = model(data.x, data.edge_index, data.edge_attr).squeeze()  # Full graph pass
    prediction = (output > 0.5).float()  # Binary prediction for all nodes
    return ('ASD' if prediction[-1]==0 else 'Typical Control'),output[-1]

relu=nn.ReLU()
n1=7
dout=0.2
N=111
device='cuda' if torch.cuda.is_available() else 'cpu'

"""
path="data-20250421T122540Z-001/data/cpac/nofilt_noglobal/"
file_id='NYU_0051070'
process_data(file_id,path=path)
"""
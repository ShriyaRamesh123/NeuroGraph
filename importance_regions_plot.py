import nibabel as nib
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from skimage import measure
from skimage.draw import ellipsoid
import scipy.io
import random
import colorsys


def generate_random_color():
    # Generate a random hue (0-360 degrees)
    hue = random.randint(0, 360)
    
    # Keep saturation and lightness in a moderate range (e.g., 40-80%)
    saturation = random.uniform(0.4, 0.8)
    lightness = random.uniform(0.4, 0.8)
    
    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    
    # Return the color in RGBA format with 0.7 opacity
    return f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.7)'



#Function to generate random color
'''
def generate_random_color():
    return f'rgba({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)}, 0.7)'
'''

def plot_regions(li , seg, roi_l):
    roi_meshes = []
    for temp in li:
        color=generate_random_color()
        for roi_label in temp:
            if roi_label == -2:
                continue
            roi_mask = (seg == roi_label).astype(np.uint8)  # Binary mask for the current ROI
            # Generate mesh from this binary mask
            verts, faces, normals, values = measure.marching_cubes(roi_mask, level=0.5)
            x, y, z = verts.T
            i, j, k = faces.T

            hover_text = roi_l.get(roi_label)

            mesh2 = go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.5, i=i, j=j, k=k, name=hover_text,
            hoverinfo='text',  # Enable hover text
            text=[hover_text] * len(x), showlegend=True  )
            roi_meshes.append(mesh2)
    return roi_meshes
    



def highlight_regions(file_id=None, important_regions=None):
    file_path1='data-20250421T122540Z-001/ABIDE_pcp/cpac/nofilt_noglobal/'
    #Get Atlas data for segmentation and 4D fMRI
    file_path=file_path1+file_id+'_func_preproc.nii.gz'
    im = nib.load(file_path).get_fdata()[:,:,:,0] #for first timeframe
    seg = nib.load('ho_roi_atlas.nii/ho_mask_pad.nii').get_fdata()

    
    #Get Important regions
    important_regions=scipy.io.loadmat('important_regions.mat')['important_regions'][0]
    
    labels, roi=pd.read_csv('ho_labels.csv')['Unnamed: 1'].tolist()[1:], pd.read_csv('ho_labels.csv')['# Generated from atlas label .xml files distributed with FSL'].tolist()[1:]
    
    roi_l={}
    for i in important_regions:
        for j in i:
            if j==-2:
                roi_l[-2]='Background'
                continue
            ind=roi.index(str(j))
            roi_l[j]=labels[ind]



    #plotting the brain 
    verts, faces, normals, values = measure.marching_cubes(im, 0)
    x, y, z = verts.T
    i, j, k = faces.T

    mesh1 = go.Mesh3d(x=x, y=y, z=z,color='gray', opacity=0.5, i=i, j=j, k=k)

    all_meshes=[mesh1]+plot_regions(important_regions,seg, roi_l)
    bfig = go.Figure(data=all_meshes)
    #bfig.show()
    bfig.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white'),
)
    return bfig

    
def get_static_figure(file_id):
 if file_id!=None: 
    return highlight_regions(file_id) 
    

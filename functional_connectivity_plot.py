import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from skimage import measure
import scipy.io
from plotly.colors import sample_colorscale

# Load .1d file (e.g., AFNI timeseries file)
def load_time_series(path, roi_ids, regions):
    series= np.loadtxt(path)[:,1:]
    new_series=[]
    s=set()
    for li in regions:
        for i in li:
            if i==-2:
                continue
            s.add(roi_ids.index(str(i)))
    
    for i in s:
    
        new_series.append(series[:,i])
    return np.array(new_series)


def value_to_color(val, vmin, vmax, colormap="Blues"):
    norm = (val - vmin) / (vmax - vmin + 1e-6)
    return sample_colorscale(colormap, norm)[0]

def create_roi_mesh(seg, roi_label, color, roi_name):
    roi_mask = (seg == roi_label).astype(np.uint8)
    if roi_mask.sum() == 0:
        return None
    verts, faces, normals, values = measure.marching_cubes(roi_mask, level=0.5)
    x, y, z = verts.T
    i, j, k = faces.T
    return go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.6, i=i, j=j, k=k,
                     name=roi_name, hoverinfo='text', text=[roi_name] * len(x), showlegend=True)

def highlight_regions_with_animation(seg_path, func_path, labels_csv, regions_mat, timeseries_path):

    # Load labels and region metadata
    regions = scipy.io.loadmat(regions_mat)['important_regions'][0]
    labels = pd.read_csv(labels_csv)['Unnamed: 1'].tolist()[1:]
    roi_ids = pd.read_csv(labels_csv)['# Generated from atlas label .xml files distributed with FSL'].tolist()[1:]

    # Load files
    seg = nib.load(seg_path).get_fdata()
    im = nib.load(func_path).get_fdata()[:, :, :, 0]
    time_series = load_time_series(timeseries_path, roi_ids, regions)  # shape: [T, num_rois]
    

    

    roi_l = {}
    unique_rois = sorted({r for group in regions for r in group if r != -2})
    #print("Time series shape:", time_series.shape)
    #print("Number of ROIs:", len(unique_rois))
    for j in unique_rois:
        roi_l[j] = labels[roi_ids.index(str(j))]

    # Base brain mesh
    verts, faces, normals, values = measure.marching_cubes(im, 0)
    x, y, z = verts.T
    i, j, k = faces.T
    base_mesh = go.Mesh3d(x=x, y=y, z=z, color='lightgray', opacity=0.2, i=i, j=j, k=k, name="Brain")

    # Get min/max for colormap scaling
    vmin, vmax = np.min(time_series), np.max(time_series)

    # Make frames for each time point
    frames = []
    for t in range(time_series.shape[1]):
        if t%5!=0:
            continue
        roi_meshes = [base_mesh]
        for idx, roi_label in enumerate(unique_rois):
            val = time_series[idx, t]
            color = value_to_color(val, vmin, vmax)
            mesh = create_roi_mesh(seg, roi_label, color, roi_l[roi_label])
            if mesh:
                roi_meshes.append(mesh)
    
        frames.append(go.Frame(data=roi_meshes, name=str(t//5)))

    # Initial figure
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            updatemenus=[dict(
                type='buttons',
                buttons=[dict(label='Play', method='animate', args=[None])],
                showactive=False
            )],
            sliders=[{
                'steps': [{'method': 'animate', 'args': [[str(k)], {'mode': 'immediate'}], 'label': str(k)} for k in range(len(frames))],
                'transition': {'duration': 0},
                'x': 0, 'y': 0, 'len': 1.0
            }]
        )
    )
    fig.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white'),
)
    #fig.show()
    return fig

# Call function
def get_animated_figure(file_id):
    file_path1='data-20250421T122540Z-001/ABIDE_pcp/cpac/nofilt_noglobal/'
    if file_id!=None:
        return highlight_regions_with_animation(
        seg_path='ho_roi_atlas.nii/ho_mask_pad.nii',
        func_path=file_path1+file_id+'_func_preproc.nii.gz',
        labels_csv='ho_labels.csv',
        regions_mat='important_regions.mat',
        timeseries_path='data-20250421T122540Z-001/data/cpac/nofilt_noglobal/'+file_id+'_rois_ho.1D'
        )

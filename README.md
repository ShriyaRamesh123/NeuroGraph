
# NeuroGraph

NeuroGraph is an ML project that is used for the objective diagnosis of Autism Spectrum Disorder Detection using the subject's resting state functional MRI data.


## Tools and Libraries Used

Language: ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Libraries: 

* ![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-FF9800?style=for-the-badge&logo=pytorch&logoColor=white)
* ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
* ![Plotly](https://img.shields.io/badge/Plotly-17181A?style=for-the-badge&logo=plotly&logoColor=white)
* ![Nilearn](https://img.shields.io/badge/Nilearn-3499CD?style=for-the-badge&logo=python&logoColor=white)
* ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
* ![Dash](https://img.shields.io/badge/Dash-008DE4?style=for-the-badge&logo=plotly&logoColor=white)
* ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)




## Table of Contents

1) About
2) Usage


## Usage

### Running the Project
1) Open the ASD_project_folder in VS Code
2) Run Final_app_run.py script. Wait for this message in the terminal:

Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'Final_app_run'
 * Debug mode: on

3) Open index1.html file in the html_css folder from vscode File Explorer on the left side of the screen. Right click on any blank area of the html file open in VS Code and click on open with live server from the drop down.
This should open a web page with NeuroGraph as the title and blue animated background.

4) Enter file name you want to check for ASD in the Enter File ID section. eg: Pitt_0050003
The list of File names you can try this out is in the ----- folder

5) Click on run prediction and results will display. This will take some time to run and display the results. Then the graph/brain plot will also load. You can interact with the plot by selecting or unselecting some regions given on the right hand side. You can use left click and drag or right click and drag to orient the plot in different ways.




## Screenshots

![App Landing Page](https://github.com/ShriyaRamesh123/NeuroGraph/blob/main/Landing%20apage.png)

![ASD prediction](https://github.com/ShriyaRamesh123/NeuroGraph/blob/main/Prediction%20result.png)

![Top regions in brain contributing to ASD](https://github.com/ShriyaRamesh123/NeuroGraph/blob/main/Graph%20plot.png)




[pytorch_geo]: https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo.png
[pytorch_geo-url]: https://pytorch-geometric.readthedocs.io/en/latest/

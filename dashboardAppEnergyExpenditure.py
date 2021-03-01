import dash, glob, plotly, os, sys, math, json
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from collections import deque
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import kurtosis, skew
import statsmodels.api as sm
from jointAnglesAndMotion import *
import plotly.express as px
from sklearn.datasets import make_classification

STAND_START_COL=26
SW_START_COL=28
MW_START_COL=41
JOG_START_COL=54
SPRINT_START_COL=67
SPRINT_TIME=68

ACTIVITY=['Jog', 'Light Walk']
PATH=['Json Files/jog100.json/', 'Json Files/LW100/']
START_JSON=[0, 89]
NUM_ROWS=[40, 52]
END_JSON=[1206,1650]



ELBOW1_PTS=[2, 3, 4]
ELBOW2_PTS=[5, 6, 7]
KNEE1_PTS=[9, 10, 11]
KNEE2_PTS=[12, 13, 14]

participant_data=pd.read_csv('participantData.csv', skiprows=[51, 52, 53, 54, 55])
actigraph_data_list=[pd.read_csv(filename, skiprows=10) for filename in glob.glob("Actigraph Data/*.csv")]
image_list=[Image.open(filename, mode='r') for filename in glob.glob('Json Files/Jog_images/*.PNG')]

time=actigraph_data_list[0]['Time'].values[:].tolist()
axis1=actigraph_data_list[0]['Axis1'].values[:].tolist()
axis2=actigraph_data_list[0]['Axis2'].values[:].tolist()
axis3=actigraph_data_list[0]['Axis3'].values[:].tolist()


app_colors = {
    'background': '#FFFFFF',
    'text': 'slategray',
    'plot':'#41EAD4',
    'bar':'#FBFC74',
    'someothercolor':'#FF206E',
}

app=dash.Dash(__name__)
app.layout=html.Div(
    [html.Div(className='container-fluid', children=[html.H3('hi', style={'textAlign': 'center','color':'#FFFFFF'}),
                                                     html.H1('ENERGY EXPENDITURE', style={'textAlign': 'center','color':"darkslategray"}),
                                                     ],
              style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000}),
     html.Div(className='row', children=[dcc.Graph(id='actigraph-plot', animate=False, style={'width': '50%', 'display': 'inline-block'}),
                                        html.Div(children=[html.Img(id='image', n_clicks=0, style={'width': '100%','height':'130%', 'display': 'inline-block'}),
                                                 dcc.Slider(id='image-slider', min=0, max=len(image_list)-1, step=1, value=1)], style={'width': '50%', 'display': 'inline-block'})]),
      html.Div(className='row', children=[html.H2('ACTIVITY LEVEL', style={'color':app_colors['text']}),
                                                     dcc.Dropdown(id="dropdown", options=[{'label': x, 'value': x}
                                                                                          for x in ACTIVITY], value=ACTIVITY[0]),
                                                     html.H2('ELBOW AND KNEE ANGLE PLOTS', style={'textAlign': 'center','color':app_colors['text']}),
                                          dcc.Graph(id='elbowangles-graph', animate=False, style={'width': '50%', 'display': 'inline-block'}),
                                          dcc.Graph(id='kneeangles-graph', animate=False, style={'width': '50%', 'display': 'inline-block'})]),
     html.Div(className='row', children=[html.H3('Regression with Joint Angle Values', style={'textAlign': 'center','color':app_colors['text']}),
                                         dcc.Graph(id='regression-plot1', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'}),
                                         dcc.Markdown(id='m1',style={'width': '50%', 'display': 'inline-block'})]),
     html.Div(className='row', children=[html.H3('Regression with Change in Joint Angle Values', style={'textAlign': 'center','color':app_colors['text']}),
                                         dcc.Markdown(id="m2",style={'width': '50%', 'display': 'inline-block'}),
                                         dcc.Graph(id='regression-plot2', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'})]),
     html.Div(className='row', children=[html.H3('Regression with Joint Angle Values and Change in Joint Angle Values', style={'textAlign': 'center','color':app_colors['text']}),
                                         dcc.Graph(id='regression-plot3', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'}),
                                         dcc.Markdown(id="m3",style={'width': '50%', 'display': 'inline-block'})]),
     html.Div(className='row', children=[html.H3('FEATURE IMPORTANCE RANKING', style={'textAlign': 'center','color':app_colors['text']}),
                                         dcc.Graph(id='feature-importance', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'}),
                                         dcc.Graph(id='significant-features', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'})]),
     html.Div(className='row', children=[html.H3('COMPARING DIFFERENT MODELS', style={'textAlign': 'center','color':app_colors['text']}),
                                         dcc.Graph(id='errors-subplot1', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'}),
                                         dcc.Graph(id='errors-subplot2', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'}),
                                         dcc.Graph(id='errors-subplot3', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'}),
                                         dcc.Markdown(id="model-comparison",style={'width': '50%', 'display': 'inline-block'})]),
    html.Div(className='row', children=[html.H3('MODEL EVALUATION WITH DIFFERENT DATA', style={'textAlign': 'center','color':app_colors['text']}),
                                         dcc.Graph(id='JogModelLwData', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'}),
                                         dcc.Graph(id='LwModelJogData', animate=False, className='col s12 m6 l6', style={'width': '50%', 'display': 'inline-block'})]),

     dcc.Interval(id='actigraph-update', interval=1*1000),
     dcc.Interval(id='angle-update', interval=1*1000),
     dcc.Interval(id='regression-update', interval=1*1000),
     dcc.Interval(id='image-update', interval=1*1000)],
     style={'backgroundColor': app_colors['background'], 'margin-top':'-30px', 'height':'2000px'})

@app.callback(Output("actigraph-plot", "figure"), [Input('actigraph-update', 'interval'),  Input('image-slider','value')])
def update_actigraph_subplots(n, val):
    fig = plotly.subplots.make_subplots(rows=3, cols=1, shared_xaxes=True,vertical_spacing=0.009,horizontal_spacing=0.009, y_title='Acceleration', subplot_titles=['ACTIGRAPH DATA', None, None])
    fig.append_trace({'x':list(time),'y':list(axis1),'type':'scatter','name':'x-acceleration'},1,1)
    fig.append_trace({'x':list(time),'y':list(axis2),'type':'scatter','name':'y-acceleration'},2,1)
    fig.append_trace({'x':list(time),'y':list(axis3),'type':'scatter','name':'z-acceleration'},3,1)
    fig.add_shape(go.layout.Shape(type="line",yref="paper",xref="x",x0="13:31:00",y0=0,x1="13:31:00",y1=1000,opacity=0.5,line=dict(color='black', width=3),),row=1,col=1)
    fig.add_shape(go.layout.Shape(type="line",yref="paper",xref="x",x0="13:31:00",y0=0,x1="13:31:00",y1=1000,opacity=0.5,line=dict(color='black', width=3),),row=2,col=1)
    fig.add_shape(go.layout.Shape(type="line",yref="paper",xref="x",x0="13:31:00",y0=0,x1="13:31:00",y1=1000,opacity=0.5,line=dict(color='black', width=3),),row=3,col=1)
    fig.add_shape(go.layout.Shape(type="line",yref="paper",xref="x",x0="13:35:00",y0=0,x1="13:35:00",y1=1000,opacity=0.5,line=dict(color='black', width=3),),row=1,col=1)
    fig.add_shape(go.layout.Shape(type="line",yref="paper",xref="x",x0="13:35:00",y0=0,x1="13:35:00",y1=1000,opacity=0.5,line=dict(color='black', width=3),),row=2,col=1)
    fig.add_shape(go.layout.Shape(type="line",yref="paper",xref="x",x0="13:35:00",y0=0,x1="13:35:00",y1=1000,opacity=0.5,line=dict(color='black', width=3),),row=3,col=1)
    return fig

@app.callback(Output("image", 'src'),  [Input('actigraph-update', 'interval'), Input('image-slider','value')])
def update_image(n, slider_val):
    return image_list[slider_val]

@app.callback([Output("elbowangles-graph","figure"), Output("kneeangles-graph","figure")], [Input('angle-update', 'interval'), Input("dropdown", "value")])
def update_elbow_angles_plot(n, val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    ae1, ae2, ak1, ak2=joint_angles(json_files, PATH[i])
    se1, se2, sk1, sk2, t=change_in_angles(START_JSON[i], NUM_ROWS[i], json_files, PATH[i])
    fig1=plotly.subplots.make_subplots(rows=2, cols=2, x_title="Time (Seconds)", y_title="Angle (Degrees)")
    fig1.append_trace(go.Scatter(name="E.A.1",x=t, y=ae1), 1,1)
    fig1.append_trace(go.Scatter(name="Speed E.A.1",x=t, y=se1), 2,1)
    fig1.append_trace(go.Scatter(name="E.A.2",x=t, y=ae2), 1,2)
    fig1.append_trace(go.Scatter(name="Speed E.A.2",x=t, y=se2), 2,2)
    fig2=plotly.subplots.make_subplots(rows=2, cols=2, x_title="Time (Seconds)", y_title="Angle (Degrees)")
    fig2.append_trace(go.Scatter(name="K.A.1",x=t, y=ak1),1,1)
    fig2.append_trace(go.Scatter(name="Speed K.A.1",x=t, y=sk1), 2,1)
    fig2.append_trace(go.Scatter(name="K.A.2",x=t, y=ak2), 1,2)
    fig2.append_trace(go.Scatter(name="Speed K.A.2",x=t, y=sk2), 2,2)
    return fig1, fig2

@app.callback([Output("regression-plot1","figure"),Output("regression-plot2","figure"), Output("regression-plot3","figure")], [Input('dropdown', 'value')])
def update_predicted_vs_actual_regression_plots(val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    df=make_dataFrame(START_JSON[i], END_JSON[i], json_files, PATH[i])
    m1, m2, m3, y1, y2, y3, py11, py12, py13=compare_twosided_wilcoxon_actual_predicted(df, NUM_ROWS[i], FEATURES1, CLASSIFICATION)
    fig1=plotly.subplots.make_subplots(rows=1, cols=1, x_title="Actigraph Accelerations", y_title="Predicted Accelerations")
    fig1.append_trace(go.Scatter(name="Axis1",x=y1, y=py11,  mode='markers'), 1,1)
    fig1.append_trace(go.Scatter(name="Axis2",x=y2, y=py12, mode='markers'), 1,1)
    fig1.append_trace(go.Scatter(name="Axis3",x=y3, y=py13, mode='markers'), 1,1)
    m1, m2, m3, y1, y2, y3, py21, py22, py23=compare_twosided_wilcoxon_actual_predicted(df, NUM_ROWS[i], FEATURES2, CLASSIFICATION)
    fig2=plotly.subplots.make_subplots(rows=1, cols=1, x_title="Actigraph Accelerations", y_title="Predicted Accelerations")
    fig2.append_trace(go.Scatter(name="Axis1",x=y1, y=py21,  mode='markers'), 1,1)
    fig2.append_trace(go.Scatter(name="Axis2",x=y2, y=py22, mode='markers'), 1,1)
    fig2.append_trace(go.Scatter(name="Axis3",x=y3, y=py23, mode='markers'), 1,1)
    m1, m2, m3, y1, y2, y3, py31, py32, py33=compare_twosided_wilcoxon_actual_predicted(df, NUM_ROWS[i], FEATURES3, CLASSIFICATION)
    fig3=plotly.subplots.make_subplots(rows=1, cols=1, x_title="Actigraph Accelerations", y_title="Predicted Accelerations")
    fig3.append_trace(go.Scatter(name="Axis1",x=y1, y=py31,  mode='markers'), 1,1)
    fig3.append_trace(go.Scatter(name="Axis2",x=y2, y=py32, mode='markers'), 1,1)
    fig3.append_trace(go.Scatter(name="Axis3",x=y3, y=py33, mode='markers'), 1,1)
    return fig1, fig2, fig3


@app.callback(Output("m1", "children"), [Input('dropdown', 'value')])
def update_analysis1(val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    df=make_dataFrame(START_JSON[i], END_JSON[i], json_files, PATH[i])
    m1, m2, m3, y1, y2, y3, py11, py12, py13=compare_twosided_wilcoxon_actual_predicted(df, NUM_ROWS[i], FEATURES1, CLASSIFICATION)
    title="**Statistical Analysis**"
    hn='''*Null Hypothesis:* The predicted acceleration values are equal to the actigraph acceleration values.'''
    ha='''*Alternate Hypothesis:* The predicted acceleration values are not equal to the actigraph acceleration values.'''
    w1,p1=m1
    w2,p2=m2
    w3,p3=m3
    axis1="Probability Value for Axis1 acceleration: "+str(round(p1,4))
    axis2="Probability Value for Axis2 acceleration: "+str(round(p2,4))
    axis3="Probability Value for Axis3 acceleration: "+str(round(p3,4))
    if (p1<SIGNIFICANCE_LEVEL or p2<SIGNIFICANCE_LEVEL or p3<SIGNIFICANCE_LEVEL):
        conclusion="At a significance level of "+str(SIGNIFICANCE_LEVEL)+", we reject the null hypthosis because one or more probabilities is less than the alpha value. The predicted acceleration values are not equal to the actigraph acceleration values."
    else:
        conclusion="At a significance level of "+str(SIGNIFICANCE_LEVEL)+", we fail to reject the null hypthosis. The predicted acceleration values are equal to the actigraph acceleration values."

    return title+'\n\n\n'+hn+'\n'+ha+'\n\n'+axis1+'\n\n'+axis2+'\n\n'+axis3+'\n\n'+conclusion+' \n \n'+'''*(Wilcoxon Test)*'''


@app.callback(Output("m2", "children"), [Input('dropdown', 'value')])
def update_analysis1(val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    df=make_dataFrame(START_JSON[i], END_JSON[i], json_files, PATH[i])
    m1, m2, m3, y1, y2, y3, py11, py12, py13=compare_twosided_wilcoxon_actual_predicted(df, NUM_ROWS[i], FEATURES2, CLASSIFICATION)
    title="**Statistical Analysis**"
    hn='''*Null Hypothesis:* The predicted acceleration values are equal to the actigraph acceleration values.'''
    ha='''*Alternate Hypothesis:* The predicted acceleration values are not equal to the actigraph acceleration values.'''
    w1,p1=m1
    w2,p2=m2
    w3,p3=m3
    axis1="Probability Value for Axis1 acceleration: "+str(round(p1,4))
    axis2="Probability Value for Axis2 acceleration: "+str(round(p2,4))
    axis3="Probability Value for Axis3 acceleration: "+str(round(p3,4))
    if (p1<SIGNIFICANCE_LEVEL or p2<SIGNIFICANCE_LEVEL or p3<SIGNIFICANCE_LEVEL):
       conclusion="At a significance level of "+str(SIGNIFICANCE_LEVEL)+", we reject the null hypthosis because one or more probabilities is less than the alpha value. The predicted acceleration values are not equal to the actigraph acceleration values."
    else:
        conclusion="At a significance level of "+str(SIGNIFICANCE_LEVEL)+", we fail to reject the null hypthosis. The predicted acceleration values are equal to the actigraph acceleration values."
    return title+'\n\n\n'+hn+'\n'+ha+'\n\n'+axis1+'\n\n'+axis2+'\n\n'+axis3+'\n\n'+conclusion+' \n \n'+'''*(Wilcoxon Test)*'''


@app.callback(Output("m3", "children"), [Input('dropdown', 'value')])
def update_analysis1(val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    df=make_dataFrame(START_JSON[i], END_JSON[i], json_files, PATH[i])
    m1, m2, m3, y1, y2, y3, py11, py12, py13=compare_twosided_wilcoxon_actual_predicted(df, NUM_ROWS[i], FEATURES3, CLASSIFICATION)
    title="**Statistical Analysis**"
    hn='''*Null Hypothesis:* The predicted acceleration values are equal to the actigraph acceleration values.'''
    ha='''*Alternate Hypothesis:* The predicted acceleration values are not equal to the actigraph acceleration values.'''
    w1,p1=m1
    w2,p2=m2
    w3,p3=m3
    axis1="Probability Value for Axis1 acceleration: "+str(round(p1,4))
    axis2="Probability Value for Axis2 acceleration: "+str(round(p2,4))
    axis3="Probability Value for Axis3 acceleration: "+str(round(p3,4))
    if (p1<SIGNIFICANCE_LEVEL or p2<SIGNIFICANCE_LEVEL or p3<SIGNIFICANCE_LEVEL):
        conclusion="At a significance level of "+str(SIGNIFICANCE_LEVEL)+", we reject the null hypthosis because one or more probabilities is less than the alpha value. The predicted acceleration values are not equal to the actigraph acceleration values."
    else:
        conclusion="At a significance level of "+str(SIGNIFICANCE_LEVEL)+", we fail to reject the null hypthosis. The predicted acceleration values are equal to the actigraph acceleration values."

    return title+'\n\n\n'+hn+'\n'+ha+'\n\n'+axis1+'\n\n'+axis2+'\n\n'+axis3+'\n\n'+conclusion+' \n \n'+'''*(Wilcoxon Test)*'''
@app.callback(Output("feature-importance", "figure"), [Input('dropdown', 'value')])
def feature_importance_plot(val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    df=make_dataFrame(START_JSON[i], END_JSON[i], json_files, PATH[i])
    importance=get_feature_importance(df, FEATURES3, CLASSIFICATION, NUM_ROWS[i])
    fig= go.Figure([go.Bar(name="Importance Ranking",x= FEATURES3, y=importance, marker_color='indianred')])
    return fig
@app.callback(Output("significant-features", "figure"), [Input('dropdown', 'value')])
def significant_features_fit(val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    df=make_dataFrame(START_JSON[i], END_JSON[i], json_files, PATH[i])
    importance=get_feature_importance(df, FEATURES3, CLASSIFICATION, NUM_ROWS[i])
    significant_features=[]
    for k in importance:
        if k!=0.:
            significant_features.append(FEATURES3[importance.tolist().index(k)])
    m1, m2, m3, y1, y2, y3, py11, py12, py13=compare_twosided_wilcoxon_actual_predicted(df, NUM_ROWS[i], significant_features, CLASSIFICATION)
    fig1=plotly.subplots.make_subplots(rows=1, cols=1, x_title="Actigraph Accelerations", y_title="Predicted Accelerations")
    fig1.append_trace(go.Scatter(name="Axis1",x=y1, y=py11,  mode='markers'), 1,1)
    fig1.append_trace(go.Scatter(name="Axis2",x=y2, y=py12, mode='markers'), 1,1)
    fig1.append_trace(go.Scatter(name="Axis3",x=y3, y=py13, mode='markers'), 1,1)
    return fig1

@app.callback([Output("errors-subplot1", "figure"),Output("errors-subplot2", "figure"),Output("errors-subplot3", "figure")], [Input('dropdown', 'value')])
def errors_subplot(val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    df=make_dataFrame(START_JSON[i], END_JSON[i], json_files, PATH[i])
    clf=develop_regression_model(df, FEATURES1,CLASSIFICATION)
    y_pred1, y1=evaluate_regression_model(clf, df, FEATURES1, CLASSIFICATION, labels=None)
    errors1=y1-y_pred1
    clf=develop_regression_model(df, FEATURES2,CLASSIFICATION)
    y_pred2, y2=evaluate_regression_model(clf, df, FEATURES2, CLASSIFICATION, labels=None)
    errors2=y2-y_pred2
    clf=develop_regression_model(df, FEATURES3,CLASSIFICATION)
    y_pred3, y3=evaluate_regression_model(clf, df, FEATURES3, CLASSIFICATION, labels=None)
    errors3=y3-y_pred3
    fig1=px.histogram(errors1,x=errors1.index.name, title="ANGLES (actual-predicted)")
    fig2=px.histogram(errors2,x=errors2.index.name, title="CHANGE IN ANGLES (actual-predicted)")
    fig3=px.histogram(errors3,x=errors3.index.name, title="ANGLES AND CHANGE IN ANGLES (actual-predicted)")
    return fig1, fig2, fig3
@app.callback(Output("model-comparison", "children"), [Input('dropdown', 'value')])
def compare_models_analysis(val):
    i=ACTIVITY.index(val)
    json_files=get_json_files(PATH[i])
    df=make_dataFrame(START_JSON[i], END_JSON[i], json_files, PATH[i])
    clf=develop_regression_model(df, FEATURES1,CLASSIFICATION)
    y_pred1, y1=evaluate_regression_model(clf, df, FEATURES1, CLASSIFICATION, labels=None)
    errors1=y1-y_pred1
    clf=develop_regression_model(df, FEATURES2,CLASSIFICATION)
    y_pred2, y2=evaluate_regression_model(clf, df, FEATURES2, CLASSIFICATION, labels=None)
    errors2=y2-y_pred2
    clf=develop_regression_model(df, FEATURES3,CLASSIFICATION)
    y_pred3, y3=evaluate_regression_model(clf, df, FEATURES3, CLASSIFICATION, labels=None)
    errors3=y3-y_pred3
    h0="*Null Hypothesis:* The difference between the actual and predicted acceleration values is equal for all feature combinations"
    ha1="*Alternate Hypothesis:* The difference between the actual and predicted acceleration values is greater for angles than change in angles feature combination."
    ha2="*Alternate Hypothesis:* The difference between the actual and predicted acceleration values is greater for angles than angles and change in angles feature combination."
    ha3="*Alternate Hypothesis:* The difference between the actual and predicted acceleration values is greater for change in angles than angles and change in angles feature combination."
    wcAxis11, wcAxis12, wcAxis13=wilcoxon_comparison_different_fits(errors1, errors2, NUM_ROWS[i])
    wcAxis21, wcAxis22, wcAxis23=wilcoxon_comparison_different_fits(errors1, errors3, NUM_ROWS[i])
    wcAxis31, wcAxis32, wcAxis33=wilcoxon_comparison_different_fits(errors2, errors3, NUM_ROWS[i])
    p1="Probability Values:    **Axis1**= "+str(round(wcAxis11[1],4))+", **Axis2**="+str(round(wcAxis12[1],4))+", **Axis3**="+str(round(wcAxis13[1],4))
    p2="Probability Values:    **Axis1**= "+str(round(wcAxis21[1],4))+", **Axis2**="+str(round(wcAxis22[1],4))+", **Axis3**="+str(round(wcAxis23[1],4))
    p3="Probability Values:    **Axis1**= "+str(round(wcAxis31[1],4))+", **Axis2**="+str(round(wcAxis32[1],4))+", **Axis3**="+str(round(wcAxis33[1],4))
    conclusion="The feature combination of both angles and change in angles produces the most accurate model followed by the change in angles."
    return h0+"\n\n"+ha1+"\n\n"+p1+"\n\n"+ha2+"\n\n"+p2+"\n\n"+ha3+"\n\n"+p3+"\n\n"+conclusion
@app.callback([Output('JogModelLwData',"figure"),Output('LwModelJogData',"figure")], [Input('dropdown', 'value')])
def evaluate_model(val):
    json_filesJog=get_json_files(PATH[0])
    json_filesLW=get_json_files(PATH[1])
    dfJog=make_dataFrame(START_JSON[0], END_JSON[0], json_filesJog, PATH[0])
    dfLW=make_dataFrame(START_JSON[1], END_JSON[1], json_filesLW, PATH[1])
    clf1=develop_regression_model(dfLW, FEATURES3, CLASSIFICATION)
    y_pred1, y1=evaluate_regression_model(clf1, dfJog, FEATURES3, CLASSIFICATION, labels=None)
    py1, py2, py3=[],[],[]
    for i in range(0, NUM_ROWS[0]+1):
        py1.append(y_pred1[i][0])
        py2.append(y_pred1[i][1])
        py3.append(y_pred1[i][2])
    fig1=plotly.subplots.make_subplots(rows=1, cols=1, x_title="Actigraph Accelerations", y_title="Predicted Accelerations", subplot_titles=["Light Walk model fit with Jogging Data"])
    fig1.append_trace(go.Scatter(name="Axis1",x=y1, y=py1,  mode='markers'), 1,1)
    fig1.append_trace(go.Scatter(name="Axis2",x=y1, y=py2, mode='markers'), 1,1)
    fig1.append_trace(go.Scatter(name="Axis3",x=y1, y=py3, mode='markers'), 1,1)

    clf2=develop_regression_model(dfJog, FEATURES3, CLASSIFICATION)
    y_pred2, y2=evaluate_regression_model(clf2, dfLW, FEATURES3, CLASSIFICATION, labels=None)
    py1, py2, py3=[],[],[]
    for i in range(0, NUM_ROWS[0]+1):
        py1.append(y_pred2[i][0])
        py2.append(y_pred2[i][1])
        py3.append(y_pred2[i][2])
    fig2=plotly.subplots.make_subplots(rows=1, cols=1, x_title="Actigraph Accelerations", y_title="Predicted Accelerations", subplot_titles=["Jog model fit with Light Walk Data"])
    fig2.append_trace(go.Scatter(name="Axis1",x=y1, y=py1,  mode='markers'), 1,1)
    fig2.append_trace(go.Scatter(name="Axis2",x=y1, y=py2, mode='markers'), 1,1)
    fig2.append_trace(go.Scatter(name="Axis3",x=y1, y=py3, mode='markers'), 1,1)
    return fig1, fig2


app.run_server(debug=True, threaded=True)
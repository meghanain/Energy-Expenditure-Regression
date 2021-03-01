
import json, os, math, sklearn, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from scipy.stats import wilcoxon
import graphviz

#Keypoint indexes corresponding to the OpenPose Json Data.
ELBOW1_PTS=[2, 3, 4]
ELBOW2_PTS=[5, 6, 7]
KNEE1_PTS=[9, 10, 11]
KNEE2_PTS=[12, 13, 14]

#Reads the participant demographic data and stores it in participant data.
participant_data=pd.read_csv('participantData.csv', skiprows=[51, 52, 53, 54, 55])

#Stores every participant's actigraph data in the actigraph_data_list.
actigraph_data_list=[pd.read_csv(filename, skiprows=10) for filename in glob.glob("Actigraph Data/*.csv")]

#Stores the time, and actigraph data for participant 0. Can be made into a function.
time=actigraph_data_list[0]['Time'].values[:].tolist()
axis1=actigraph_data_list[0]['Axis1'].values[:].tolist()
axis2=actigraph_data_list[0]['Axis2'].values[:].tolist()
axis3=actigraph_data_list[0]['Axis3'].values[:].tolist()

FPS=30# Video frames per second

#Classification and features for Decision Tree Regression Model Development.
CLASSIFICATION=["Axis1", "Axis2", "Axis3"]
FEATURES1=["Elbow1", "Elbow2", "Knee1", "Knee2"]
FEATURES2=["Change_E1","Change_E2","Change_K1","Change_K2"]
FEATURES3=FEATURES1+FEATURES2
FEATURES_LIST=[FEATURES1, FEATURES2, FEATURES3]

SIGNIFICANCE_LEVEL=0.10

def main():
    json_path_LW='Json Files/LW100/'
    json_files_LW=get_json_files(json_path_LW)
    dfLW=make_dataFrame(89, 1650, json_files_LW, json_path_LW)
    get_feature_importance(dfLW, FEATURES3, CLASSIFICATION, 52)
    json_path_jog='Json Files/jog100.json/'
    json_files_jog=get_json_files(json_path_jog)
    dfJOG=make_dataFrame(0, 1206, json_files_jog, json_path_jog)
    get_feature_importance(dfJOG, FEATURES3, CLASSIFICATION, 40)
    #compare_errors(dfLW, 52)
    '''json_path_LW='Json Files/LW100/'
    json_files_LW=get_json_files(json_path_LW)
    dfLW=make_dataFrame(89, 1650, json_files_LW, json_path_LW)
    print(compare_models(dfLW,52))
    json_path_jog='Json Files/jog100.json/'
    json_files_jog=get_json_files(json_path_jog)
    dfJOG=make_dataFrame(0, 1206, json_files_jog, json_path_jog)
    print(compare_models(dfJOG, 40))'''
    pass


def get_json_files(path):#Consumes a path and returns a list of .json files in the folder
    return [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]

def getIndiPATimes(id):#Consumes person's id and returns (stand, SW, MW, Jog, sprint, sprint time length) for each participant
    return (participant_data['Stand_TimeStart'].values[id]+":00", participant_data['SW_TimeStart'].values[id]+":00", participant_data['MW_TimeStart'].values[id]+":00", participant_data['Jog_TimeStart'].values[id]+":00", participant_data['Sprint_TimeStart'].values[id]+":00", participant_data['Sprint_Time_seconds'].values[id])

def get_LW_DATA(id):#Consumes person's id and returns the corresponding light walk time and actigraph data.
    i1=time.index(getIndiPATimes(id)[1])
    i2=time.index(getIndiPATimes(id)[2])
    return time[i1:i2], axis1[i1:i2], axis2[i1:i2], axis3[i1:i2]

def get_JOG_DATA(id):#Consumes person's id and returns the corresponding jog time and actigraph data.
    i1=time.index(getIndiPATimes(id)[3])
    i2=time.index(getIndiPATimes(id)[4])
    return time[i1:i2], axis1[i1:i2], axis2[i1:i2], axis3[i1:i2]

def get_keypoints(json_files, path):
    '''Consumes the json_files list and path, and reads them.
    Returns the number of people annotated followed by a list keypoint x locations, y locations, and confidence level of the keypoint.'''
    keypoints=[]
    for file in json_files:
        with open(path+file,'r') as myfile:
            data=myfile.read()
            if (len(json.loads(data)['people'])==0):
                keypoints.append(-1)
            for i in range(0,len(json.loads(data)['people'])):
                keypoints_person=json.loads(data)['people'][i]['pose_keypoints_2d']
                x=keypoints_person[0::3]
                y=keypoints_person[1::3]
                c=x=keypoints_person[2::3]
                keypoints.append((i+1, x, y, c))
    return keypoints

def calculate_angles(a0, a1, b0, b1, c0, c1):
    #Calculates the angles given the elbow and knee keypoints (requires 3 points x,y locations).
    ang = math.degrees(
        math.atan2(c1-b1, c0-b0) - math.atan2(a1-b1, a0-b0))
    return ang + 360 if ang < 0 else ang

def joint_angles(json_files, path):
    #Consumes json_files list and path. Returns lists of elbow and knee angles in all the json files.
    elbow_angle1, elbow_angle2, knee_angle1, knee_angle2=[],[],[],[]
    all_kp=get_keypoints(json_files, path)
    for i in range(0, len(json_files)):
        kp=all_kp[i]
        if (kp==-1):
            elbow_angle1.append(-1)
            elbow_angle2.append(-1)
            knee_angle1.append(-1)
            knee_angle2.append(-1)
            continue
        elbow_angle1.append(calculate_angles(kp[1][ELBOW1_PTS[0]], kp[2][ELBOW1_PTS[0]], kp[1][ELBOW1_PTS[1]], kp[2][ELBOW1_PTS[1]], kp[1][ELBOW1_PTS[2]], kp[2][ELBOW1_PTS[2]]))
        elbow_angle2.append(calculate_angles(kp[1][ELBOW2_PTS[0]], kp[2][ELBOW2_PTS[0]], kp[1][ELBOW2_PTS[1]], kp[2][ELBOW2_PTS[1]], kp[1][ELBOW2_PTS[2]], kp[2][ELBOW2_PTS[2]]))
        knee_angle1.append(calculate_angles(kp[1][KNEE1_PTS[0]], kp[2][KNEE1_PTS[0]], kp[1][KNEE1_PTS[1]], kp[2][KNEE1_PTS[1]], kp[1][KNEE1_PTS[2]], kp[2][KNEE1_PTS[2]]))
        knee_angle2.append(calculate_angles(kp[1][KNEE2_PTS[0]], kp[2][KNEE2_PTS[0]], kp[1][KNEE2_PTS[1]], kp[2][KNEE2_PTS[1]], kp[1][KNEE2_PTS[2]], kp[2][KNEE2_PTS[2]]))
    return (elbow_angle1, elbow_angle2, knee_angle1, knee_angle2)

def change_in_angles(start_json, num_of_rows, json_files, path):
    '''Consumes the start_json (starting file from when the data should be recorded), num_of_rows ((end json-start_json)/30), json_files list, and json path.
    Returns change in elbow and angle per second.
    '''
    e1, e2, k1, k2=joint_angles(json_files, path)
    se1, se2, sk1, sk2, t=[],[],[],[],[]
    for i in range(0, num_of_rows):
        t.append(i)
        se1.append(e1[start_json+(FPS*(i+1))]-e1[start_json+(FPS*i)])
        se2.append(e2[start_json+(FPS*(i+1))]-e2[start_json+(FPS*i)])
        sk1.append(k1[start_json+(FPS*(i+1))]-k1[start_json+(FPS*i)])
        sk2.append(k2[start_json+(FPS*(i+1))]-k2[start_json+(FPS*i)])
        if i==num_of_rows-1:
            t.append(i+1)
            se1.append(se1[i])
            se2.append(se2[i])
            sk1.append(sk1[i])
            sk2.append(sk2[i])
    return se1, se2, sk1, sk2, t
def make_dataFrame(start_json, end_json, json_files, path):
    '''Consumes the start_json (starting file from when the data should be recorded), end_json (last file whose data
    can be recorded), json_files list, and json path.
    Returns change in elbow and angle per second.
    '''
    num_rows=(end_json-start_json)//FPS
    e1, e2, k1, k2=joint_angles(json_files, path)
    se1, se2, sk1, sk2, t=change_in_angles(start_json, num_rows, json_files, path)
    df=pd.DataFrame({'Time':[i for i in range(0,num_rows+1)],
                     'Axis1':[axis1[i] for i in range(0,num_rows+1)],
                     'Axis2':[axis2[i] for i in range(0,num_rows+1)],
                     'Axis3':[axis3[i] for i in range(0,num_rows+1)],
                     'Elbow1':e1[start_json:end_json:FPS],
                     'Elbow2':e2[start_json:end_json:FPS],
                     'Knee1':k1[start_json:end_json:FPS],
                     'Knee2':k1[start_json:end_json:FPS],
                     'Change_E1':se1,
                     'Change_E2':se2,
                     'Change_K1':sk1,
                     'Change_K2':sk2
                     })
    #df.to_excel('./LW_data.xlsx')
    return df
def develop_regression_model(df, features, classification):
    #Consumes a data frame, list of features and a list of classifications to return a regression fit.
    X = df[features]
    y = df[classification]
    clf = DecisionTreeRegressor(max_depth=3, random_state=4)
    clf = clf.fit(X, y)
    
    return clf

def evaluate_regression_model(clf, df, features, classification, labels=None):
    #Consumes a fit, data frame, list of features, classifications and returns a y_predictated value based on the fit.
    X = df[features]
    y = df[classification]
    if len(X) == 0:
        return {"accuracy":-1, "tnr":-1, "fpr":-1, "fnr":-1, "tpr":-1}
    y_pred = clf.predict(X)
    return y_pred, y
def compare_twosided_wilcoxon_actual_predicted(df, num_rows, features, classification):
    #Consumes a dataframe, number of rows in the dataframe, features and classification
    y_predAxis1, y_predAxis2, y_predAxis3=[],[],[]
    clf=develop_regression_model(df, features,classification)
    y_pred, y=evaluate_regression_model(clf, df, features, classification, labels=None)
    for i in range(0, num_rows+1):
        y_predAxis1.append(y_pred[i][0])
        y_predAxis2.append(y_pred[i][1])
        y_predAxis3.append(y_pred[i][2])
    wcAxis1=wilcoxon(list(y["Axis1"].values), y_predAxis1, alternative="two-sided")
    wcAxis2=wilcoxon(list(y["Axis2"].values), y_predAxis2, alternative="two-sided")
    wcAxis3=wilcoxon(list(y["Axis3"].values), y_predAxis3, alternative="two-sided")
    return wcAxis1, wcAxis2, wcAxis3, list(y["Axis1"].values), list(y["Axis2"].values),list(y["Axis3"].values), y_predAxis1, y_predAxis2, y_predAxis3

def get_feature_importance(df, features, classification, num_rows):
    model=develop_regression_model(df, features, classification)
    importance=model.feature_importances_
    #for i,v in enumerate(importance):
    #    print('Feature: %0d, Score: %.5f' % (i,v))
    #plt.bar([x for x in range(len(importance))], importance)
    #plt.show()

    return importance

def wilcoxon_comparison_different_fits(error1, error2, num_rows):
    wcAxis1=wilcoxon(list(map(abs, list(error1["Axis1"].values))), list(map(abs,list(error2["Axis1"].values))), alternative="greater")
    wcAxis2=wilcoxon(list(map(abs, list(error1["Axis2"].values))), list(map(abs, list(error2["Axis2"].values))), alternative="greater")
    wcAxis3=wilcoxon(list(map(abs, list(error1["Axis3"].values))), list(map(abs, list(error2["Axis3"].values))), alternative="greater")
    return wcAxis1, wcAxis2, wcAxis3

main()
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from userapp.models import *
from adminapp.models import *
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

from sklearn.ensemble  import AdaBoostClassifier
from sklearn.svm  import SVC
from django.core.paginator import Paginator
from xgboost import XGBClassifier
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#ML models
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Create your views here.
def admin_index(request):
    messages.success(request,'login successfull')
    all_users_count =  UserDetails.objects.all().count()
    pending_users_count = UserDetails.objects.filter(user_status = 'pending').count()
    rejected_users_count = UserDetails.objects.filter(user_status = 'Rejected').count()
    accepted_users_count = UserDetails.objects.filter(user_status = 'Accepted').count()
    datasets_count = Upload_dataset_model.objects.all().count()
    no_of_predicts = Predict_details.objects.all().count()
    return render(request, 'admin/index.html',{'a' : pending_users_count, 'b' : all_users_count, 'c' : rejected_users_count, 'd' : accepted_users_count, 'e' : datasets_count, 'f' : no_of_predicts})


def admin_pending(request):
    users = UserDetails.objects.filter(user_status ='pending')
    context = {'u':users}
    return render(request, "admin/pending.html", context)

def Admin_Reject_Btn(request, x):
        user = UserDetails.objects.get(user_id = x)
        user.user_status = 'Rejected'
        messages.success(request,'Status Changed successfull')

        user.save()
        messages.warning(request, 'Rejected') 
      
        return redirect('pending')

def Admin_accept_Btn(req, x):
        user = UserDetails.objects.get(user_id = x) 
        user.user_status = 'Accepted' 
        messages.success(req,'Status Changed successfull')
 
        user.save()
        messages.success(req, 'Accepted') 
        return redirect('pending')

def admin_manage(request):
    a = UserDetails.objects.all()
    paginator = Paginator(a, 5) 
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request, "admin/manage.html", {'all':post})

def admin_upload(request):
    if request.method == 'POST':
        file = request.FILES['data_file']
        file_size = str((file.size)/1024) +' kb'
        Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(request, 'Your dataset was uploaded..')
    return render(request, "admin/anxiety-upload-data.html")

def stress_admin_upload(request):
    if request.method == 'POST':
        file = request.FILES['data_file']
        file_size = str((file.size)/1024) +' kb'
        stress_Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(request, 'Your dataset was uploaded..')
    return render(request, "admin/stress-upload-data.html")

def depression_admin_upload(request):
    if request.method == 'POST':
        file = request.FILES['data_file']
        file_size = str((file.size)/1024) +' kb'
        depression_Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(request, 'Your dataset was uploaded..')
    return render(request, "admin/depression-upload-data.html")



def delete_dataset(request, id):
    try:
        dataset = Upload_dataset_model.objects.get(user_id=id).delete()
        messages.warning(request, 'Dataset was deleted..!')
    except Upload_dataset_model.DoesNotExist:
        # Handle the case where Upload_dataset_model does not exist
        pass

    try:
        dataset = depression_Upload_dataset_model.objects.get(user_id=id).delete()
    except depression_Upload_dataset_model.DoesNotExist:
        # Handle the case where depression_Upload_dataset_model does not exist
        pass

    try:
        dataset = stress_Upload_dataset_model.objects.get(user_id=id).delete()
    except stress_Upload_dataset_model.DoesNotExist:
        # Handle the case where stress_Upload_dataset_model does not exist
        pass
    
    return redirect('view')



from itertools import chain

def admin_view(request):
    anxiety_dataset = Upload_dataset_model.objects.all()
    stress_dataset = stress_Upload_dataset_model.objects.all()
    dataset = depression_Upload_dataset_model.objects.all()

    # Combine all datasets into a single list
    all_datasets = list(chain(anxiety_dataset, stress_dataset, dataset))

    paginator = Paginator(all_datasets, 5)
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)

    return render(request, "admin/view-data.html", {'data': all_datasets, 'user': post})


def view_view(request):
    # df=pd.read_csv('heart.csv')
    data_anxiety = Upload_dataset_model.objects.last()
    data_stress = stress_Upload_dataset_model.objects.last()
    data = depression_Upload_dataset_model.objects.last()
    print(data_anxiety,type(data_anxiety),'aaaaa')
    filea = str(data_anxiety.Dataset)
    dfa = pd.read_csv(f'./media/{filea}')
    tablea = dfa.to_html(table_id='data_table')

    print(data_stress,type(data_stress),'sssss')
    files = str(data_stress.Dataset)
    dfs = pd.read_csv(f'./media/{files}')
    tables = dfs.to_html(table_id='data_table')

    print(data,type(data),'sssss')
    filed = str(data.Dataset)
    dfd = pd.read_csv(f'./media/{filed}')
    tabled = dfd.to_html(table_id='data_table')

    return render(request,'admin/view-view.html', {'a':tablea,'d':tables,'s':tabled})

def admin_xgboost_algo(request):
    return render(request, "admin/xgboost-algo.html")


# ADABoost_btn
def xgboost_btn(req):
    dataset = stress_Upload_dataset_model.objects.last()
    df_ax=pd.read_csv(dataset.Dataset.path)
   

    # independent features
    X=df_ax.drop("Stress1",axis=1)

    # dependent feature
    y=df_ax["Stress1"]


    
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

    from xgboost import XGBRegressor

    ADB = XGBRegressor()
    ADB.fit(X_train, y_train)

    # prediction
    train_prediction= ADB.predict(X_train)
    test_prediction= ADB.predict(X_test)
    print('*'*20)
    # evaluation
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    accuracy = round(r2_score(y_test,test_prediction)*100, 2)
    precession = (np.sqrt(mean_squared_error(y_test,test_prediction)))
    recall = (mean_squared_error(y_test,test_prediction))
    f1 = (mean_absolute_error(y_test,test_prediction))
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "XGBoost Algorithm"
    ADA_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = ADA_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/xgboost-algo.html',{'i': data})

def Linear_Regression_Anxiety(request):
    return render(request, "admin/Linear-Regression-A.html")

def Linear_Regression_Anxiety_btn(req):
    dataset = Upload_dataset_model.objects.last()
    df_ax=pd.read_csv(dataset.Dataset.path)
   

    # independent features
    X=df_ax.drop("Anxiety1",axis=1)

    # dependent feature
    y=df_ax["Anxiety1"]


    
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # prediction
    train_prediction= lr.predict(X_train)
    test_prediction= lr.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    accuracy = round(r2_score(y_test,test_prediction)*100, 2)
    precession = (np.sqrt(mean_squared_error(y_test,test_prediction)))
    recall = (mean_squared_error(y_test,test_prediction))
    f1 = (mean_absolute_error(y_test,test_prediction))
    name = "Linear Regression Algorithm"
    Logistic.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = Logistic.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    req.session['accuracy']=accuracy
    return render(req, 'admin/Linear-Regression-A.html',{'i': data})

def admin_decission_algo(request):
    return render(request, "admin/decission-algo.html")

def Decisiontree_btn(req):
    dataset = depression_Upload_dataset_model.objects.last()
    df_ax=pd.read_csv(dataset.Dataset.path)
   

    X=df_ax.drop("Depression1",axis=1)

    # dependent feature
    y=df_ax["Depression1"]


    
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
    from sklearn.linear_model import LinearRegression


    
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
    
   #  XGBoost
    from sklearn.tree import DecisionTreeClassifier
    DEC = DecisionTreeClassifier()
    DEC.fit(X_train, y_train)

    # prediction
    train_prediction= DEC.predict(X_train)
    test_prediction= DEC.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    accuracy = round(r2_score(y_test,test_prediction)*100, 2)
    precession = (np.sqrt(mean_squared_error(y_test,test_prediction)))
    recall = (mean_squared_error(y_test,test_prediction))
    f1 = (mean_absolute_error(y_test,test_prediction))
    name = "Decision Tree Algorithm"
    DECISSION_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = DECISSION_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    req.session['des_accuracy']=accuracy

    return render(req, 'admin/decission-algo.html',{'i':data})

def admin_knn_algo(request):
    return render(request, "admin/knn-algo.html")

def KNN_btn(req):
    dataset = Upload_dataset_model.objects.last()
    df_ax=pd.read_csv(dataset.Dataset.path)
   

    # independent features
    X=df_ax.drop("Severity_Level",axis=1)

    # dependent feature
    y=df_ax["Severity_Level"]


    
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)

    # prediction
    train_prediction= KNN.predict(X_train)
    test_prediction= KNN.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "KNN Algorithm"
    KNN_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = KNN_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/knn-algo.html',{'i': data})

def admin_RandomForestRegressor_algo(request):
    return render(request, "admin/RandomForestRegressor.html")

def RandomForestRegressor_btn(req):
    dataset = Upload_dataset_model.objects.last()
    df_ax=pd.read_csv(dataset.Dataset.path)
   

    X=df_ax.drop("Anxiety1",axis=1)

    # dependent feature
    y=df_ax["Anxiety1"]


    
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

    from sklearn.ensemble import RandomForestRegressor

    rfr = RandomForestRegressor(random_state=42, max_depth=7, n_estimators=128)
    rfr.fit(X_train, y_train)


    print('*'*10)

    # prediction
    train_prediction= rfr.predict(X_train)
    test_prediction= rfr.predict(X_test)
    print('*'*10)
    
    # evaluation
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    accuracy = round(r2_score(y_test,test_prediction)*100, 2)
    precession = (np.sqrt(mean_squared_error(y_test,test_prediction)))
    recall = (mean_squared_error(y_test,test_prediction))
    f1 = (mean_absolute_error(y_test,test_prediction))
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Random Forest Algorithm"
    SXM_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = SXM_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/RandomForestRegressor.html',{'i':data})

def Linear_Regression_Depression(request):
    return render(request, "admin/Linear-Regression-D.html")

def Linear_Regression_Depression_btn(req):
    dataset = depression_Upload_dataset_model.objects.last()
    df_ax=pd.read_csv(dataset.Dataset.path)
   

    # independent features
    X=df_ax.drop("Depression1",axis=1)

    # dependent feature
    y=df_ax["Depression1"]


    
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # prediction
    train_prediction= lr.predict(X_train)
    test_prediction= lr.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    accuracy = round(r2_score(y_test,test_prediction)*100, 2)
    precession = (np.sqrt(mean_squared_error(y_test,test_prediction)))
    recall = (mean_squared_error(y_test,test_prediction))
    f1 = (mean_absolute_error(y_test,test_prediction))
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Linear Regression Algorithm"
    RandomForest.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = RandomForest.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    req.session['ran_accuracy']=accuracy

    return render(req, 'admin/Linear-Regression-D.html',{'i': data})

def Linear_Regression_stress(request):
    return render (request, 'admin/Linear-Regression-s.html')

def Linear_Regression_stress_bnt(req):
    dataset = stress_Upload_dataset_model.objects.last()
    df_ax=pd.read_csv(dataset.Dataset.path)
   

    # independent features
    X=df_ax.drop("Stress1",axis=1)

    # dependent feature
    y=df_ax["Stress1"]


    
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # prediction
    train_prediction= lr.predict(X_train)
    test_prediction= lr.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    accuracy = round(r2_score(y_test,test_prediction)*100, 2)
    precision = (np.sqrt(mean_squared_error(y_test,test_prediction)))
    recall = (mean_squared_error(y_test,test_prediction))
    f1 = (mean_absolute_error(y_test,test_prediction))

    name = "Linear Regression Algorithm"
    GradientBoosting.objects.create(Accuracy=accuracy,Precession=precision,F1_Score=f1,Recall=recall,Name=name)
    data = GradientBoosting.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/Linear-Regression-s.html',{'i': data})


def admin_comparison_graph_d(request):
    ran_accuracy = request.session.get('ran_accuracy')
    des_accuracy = request.session.get('des_accuracy')

    details4 = des_accuracy



    # g = details6.Accuracy
    details7 = ran_accuracy
    # h = details7.Accuracy

    return render(request, 'admin/comparison-graph-d.html', {'dt':details4,'ran':details7})

def admin_comparison_graph_a(request):
    accuracy = request.session.get('accuracy')
    ran_accuracy = request.session.get('ran_accuracy')
    des_accuracy = request.session.get('des_accuracy')



    deatails1 = ADA_ALGO.objects.last()
    b = deatails1.Accuracy

    deatails3 = SXM_ALGO.objects.last()
    d = deatails3.Accuracy
    details4 = des_accuracy
    # e = details4.Accuracy
    details6 = accuracy
    # g = details6.Accuracy
    details7 = ran_accuracy
    # h = details7.Accuracy
    print( details4, details6, details7,"kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
    details9 = GradientBoosting.objects.last()
    z = details9.Accuracy
    return render(request, 'admin/comparison-graph-a.html', {'ada':b,'sxm':d,'dt':details4,'log':details6, 'ran':details7, 'gst': z})


def admin_comparison_graph_s(request):

    deatails1 = ADA_ALGO.objects.last()

    b = deatails1.Accuracy



    details9 = GradientBoosting.objects.last()
    z = details9.Accuracy
    return render(request, 'admin/comparison-graph.s.html', {'gst': z,'ada':b})

def Change_Status(req, id):
    # user_id = req.session['User_Id']
    user = UserDetails.objects.get(user_id = id)
    if user.user_status == 'Accepted':
        user.user_status = 'Rejected'   
        user.save()
        messages.success(req, 'Status Succefully Changed ') 
        return redirect('manage')
    else:
        user.user_status = 'Accepted'
        user.save()
        messages.success(req, 'Status Succefully Changed  ')
        return redirect('manage')
    
def Delete_User(req, id):
    UserDetails.objects.get(user_id = id).delete()
    messages.info(req, 'Deleted  ') 
    return redirect('manage')
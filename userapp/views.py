from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from userapp.models import *
import urllib.request
import pandas as pd
import time
from adminapp.models import *
import urllib.parse
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np


# Create your views here.
def sendSMS(user, otp, mobile):
    data = urllib.parse.urlencode({
        'username': 'Codebook',
        'apikey': '56dbbdc9cea86b276f6c',
        'mobile': mobile,
        'message': f'Hello {user}, your OTP for account activation is {otp}. This message is generated from https://www.codebook.in server. Thank you',
        'senderid': 'CODEBK'
    })
    data = data.encode('utf-8')
    # Disable SSL certificate verification
    # context = ssl._create_unverified_context()
    request = urllib.request.Request("https://smslogin.co/v3/api.php?")
    f = urllib.request.urlopen(request, data)
    return f.read()

def user_services(request):
    return render(request, 'user/services.html')

def user_register(request):
    if request.method == 'POST':
        username = request.POST.get('user_name')
        email = request.POST.get('email_address')
        password = request.POST.get('email_password')
        number = request.POST.get('contact_number')
        file = request.FILES['user_file']
        print(request)
        print(username, email, password, number, file, 'data')
        otp = str(random.randint(1000, 9999))
        print(otp, 'generated otp')
        try:
            UserDetails.objects.get(user_email = email)
            messages.info(request, 'mail already registered')
            return redirect('register')
        except:
            mail_message = f'Registration Successfully\n Your 4 digit Pin is below\n {otp}'
            print(mail_message)
            send_mail("Student Password", mail_message , settings.EMAIL_HOST_USER, [email])
            # text message
            sendSMS(username, otp, number)
        
            UserDetails.objects.create(otp=otp, user_contact = number, user_username = username, user_email = email, user_password = password,user_file = file)
            request.session['user_email'] = email
            return redirect('otp')
    return render(request, 'user/register.html')

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email_address')
        password = request.POST.get('email_password')
        print(email, password)
        try:
            user = UserDetails.objects.get(user_email = email, user_password = password)
            print(user)
            request.session['user_id'] = user.user_id
            a = request.session['user_id']
            print(a)

            if user.user_password ==  password :
                if user.user_status == 'Accepted':
                    if user.otp_status == 'verified':

                        messages.success(request,'login successfull')
                        request.session['user_id'] = user.user_id
                        print('login sucessfull')
                        user.No_Of_Times_Login += 1
                        user.save()
                        return redirect('dashboard')
                    else:
                         return redirect('otp')
                elif user.user_password ==  password and user.user_status == 'Rejected':
                    messages.warning(request,"you account is rejected")
                else:
                    messages.info(request,"your account is in pending")
            else:
                 messages.error(request,'Login credentials was incorrect...')
        except:
            print(';invalid credentials')
            print('exce')
            return redirect('login')
    return render(request, "user/user.html")

def user_admin(request):
    admin_name = 'admin@gmail.com'
    admin_password = 'admin'
    if request.method == 'POST':
        adminemail = request.POST.get('emailaddress')
        adminpassword = request.POST.get('emailpassword')
        if admin_name == adminemail and admin_password == adminpassword:
            messages.success(request,'login successfull')

            return redirect('admin_dashboard')
        
        else:
            messages.error(request,"login credentials was incorrect....")

            return redirect('admin')
    return render(request, "user/admin.html")

def user_otp(request):
    user_id = request.session['user_email']
    user =UserDetails.objects.get(user_email = user_id)
    messages.success(request, 'OTP  Sent successfully')
    print(user_id)
    print(user, 'user avilable')
    print(type(user.otp))
    print(user. otp, 'creaetd otp')   
    if request.method == 'POST':
        u_otp = request.POST.get('otp')
        u_otp = int(u_otp)
        print(u_otp, 'enter otp')
        if u_otp == user.otp:
            print('if')
            user.otp_status  = 'verified'
            user.save()
            messages.success(request, 'OTP  verified successfully')
            return redirect('login')
        else:
            print('else')
            messages.error(request, 'Invalid OTP  ') 
            return redirect('otp')
    return render(request, 'user/otp.html')

def user_index(request):
    return render(request, 'user/index.html')
 
def user_about(request):
    return render(request, "user/about.html")

def user_contact(request):
    return render(request, "user/contact.html")

def user_dashboard(request):
    prediction_count =  UserDetails.objects.all().count()
    user_id = request.session["user_id"]
    user = UserDetails.objects.get(user_id = user_id)
    return render(request, "user/dashboard.html", {'predictions' : prediction_count, 'la' : user})

def user_myprofile(request):
    views_id = request.session['user_id']
    user = UserDetails.objects.get(user_id = views_id)
    print(user, 'user_id')
    if request.method =='POST':
        username = request.POST.get('user_name')
        email = request.POST.get('email_address')
        number = request.POST.get('contact_number')
        password = request.POST.get('email_password')
        age = request.POST.get('Age_int')
        date = request.POST.get('date')
        print(username, email, number, password, date, age, 'data') 

        user.user_username = username
        user.user_email = email
        user.user_contact = number
        user.user_password = password
        user.user_dates = date 

        if len(request.FILES)!=0:
            file = request.FILES['user_file']
            user.user_file = file
            user.user_username = username
            user.user_email = email
            user.user_contact = number
            user.user_password = password
            user.save()
            messages.success(request, 'Updated Successfully...!')

        else:
            user.user_username = username
            user.user_email = email
            user.user_contact = number
            user.user_password = password
            user.save()
            messages.success(request, 'Updated Successfully...!')


    return render(request, "user/myprofile.html", {'i':user})

def user_anxiety(req):
    if req.method == 'POST':
        age = req.POST.get('age')
        RelationShip = req.POST.get('RelationShip')
        Family = req.POST.get('Family')
        Current = req.POST.get('Current')
        Education = req.POST.get('Education')
        Gender = req.POST.get('Gender')
        q1_nutrition = req.POST.get('q1_nutrition')
        q2_screentime = req.POST.get('q2_screentime')
        q3_screentime = req.POST.get('q3_screentime')
        q4_screentime = req.POST.get('q4_screentime')
        q5_screentime = req.POST.get('q5_screentime')
        q6_screentime = req.POST.get('q6_screentime')
        q7_frequency = req.POST.get('q7_frequency')


        print(age,RelationShip,Gender,Education,Current,Family,q1_nutrition,q2_screentime,q3_screentime,q4_screentime,q5_screentime,q6_screentime,q7_frequency, 'dataaaaaaaaaaaa')
        age = int(age)
        RelationShip = int(RelationShip)
        Family = int(Family)
        Current = int(Current)
        Education = int(Education)
        Gender = int(Gender)
        q1_nutrition = int(q1_nutrition)
        q2_screentime = int(q2_screentime)
        q3_screentime = int(q3_screentime)
        q4_screentime = int(q4_screentime)
        q5_screentime = int(q5_screentime)
        q6_screentime = int(q6_screentime)
        q7_frequency = int(q7_frequency)

            
        # print(type(age),x)
        # DATASET.objects.create(Age = age, Glucose = sex, BloodPressure = plasma_CA19_9, SkinThickness = creatinine, Insulin = lyve1, BMI = regb1, DiabetesPedigreeFunction = tff1)
        import pickle
        file_path = 'DASS21/lr_anxity.pkl'  # Path to the saved model file

        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            res =loaded_model.predict([[age,RelationShip,Gender,Education,Current,Family,q1_nutrition,q2_screentime,q3_screentime,q4_screentime,q5_screentime,q6_screentime,q7_frequency]])

        
            print(res,"resssssssssssssssssssresssssssssssssss")

            result_int = int(res[0])

            print(result_int,'sadfffsadfasdfhsdfjhsioufgweifygascnxufurygfxoqnfufyw8ofer' )

            dataset = Upload_dataset_model.objects.last()
            df=pd.read_csv(dataset.Dataset.path)
            X = df.drop('Anxiety1', axis = 1)
            y = df['Anxiety1']


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
            req.session['acra'] = accuracy
            req.session['prea'] = precession
            req.session['reca'] = recall
            req.session['fa'] = f1
            print(precession, accuracy,recall, f1,'uuuuuuuuuuuuuuuuuuuuuuuuuuu')

            if result_int >= 0 and result_int <= 6:
                messages.success(req, "Normal")
            elif result_int >= 8 and result_int <= 10:
                messages.warning(req, "Mild")
            elif result_int >= 12 and result_int <= 14:
                messages.warning(req, "Moderate")
            elif result_int >= 16 and result_int <= 18:
                messages.warning(req, "Severe")
            else:
                messages.warning(req, "Extremely Severe")


            req.session['resa'] = result_int
            context2 = {'acra': accuracy,'prea': precession,'f':f1,'reca':recall,'resa':result_int}

            
        return redirect("anxiety_result")
           
    return render(req, "user/anxiety-detection.html")

def user_stress(req):
    if req.method == 'POST':
        age = req.POST.get('age')
        RelationShip = req.POST.get('RelationShip')
        Family = req.POST.get('Family')
        Current = req.POST.get('Current')
        Education = req.POST.get('Education')
        Gender = req.POST.get('Gender')
        q1_nutrition = req.POST.get('q1_nutrition')
        q2_screentime = req.POST.get('q2_screentime')
        q3_screentime = req.POST.get('q3_screentime')
        q4_screentime = req.POST.get('q4_screentime')
        q5_screentime = req.POST.get('q5_screentime')
        q6_screentime = req.POST.get('q6_screentime')
        q7_frequency = req.POST.get('q7_frequency')


        print(age,RelationShip,Gender,Education,Current,Family,q1_nutrition,q2_screentime,q3_screentime,q4_screentime,q5_screentime,q6_screentime,q7_frequency, 'dataaaaaaaaaaaa')
        age = int(age)
        RelationShip = int(RelationShip)
        Family = int(Family)
        Current = int(Current)
        Education = int(Education)
        Gender = int(Gender)
        q1_nutrition = int(q1_nutrition)
        q2_screentime = int(q2_screentime)
        q3_screentime = int(q3_screentime)
        q4_screentime = int(q4_screentime)
        q5_screentime = int(q5_screentime)
        q6_screentime = int(q6_screentime)
        q7_frequency = int(q7_frequency)

            
        # print(type(age),x)
        # DATASET.objects.create(Age = age, Glucose = sex, BloodPressure = plasma_CA19_9, SkinThickness = creatinine, Insulin = lyve1, BMI = regb1, DiabetesPedigreeFunction = tff1)
        import pickle
        file_path = 'DASS21/lr_stress.pkl'  # Path to the saved model file

        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            res =loaded_model.predict([[age,RelationShip,Gender,Education,Current,Family,q1_nutrition,q2_screentime,q3_screentime,q4_screentime,q5_screentime,q6_screentime,q7_frequency]])
            print(type(res),"resssssssssssssssssssresssssssssssssss")


            result_int = int(res)
            result_in = result_int + 1
            print(res, 'sadfffsadfasdfhsdfjhsioufgweifygascnxufurygfxoqnfufyw8ofer' )

            dataset = stress_Upload_dataset_model.objects.last()
            # print(dataset.Dataset)
            df=pd.read_csv(dataset.Dataset.path)
            X = df.drop('Stress1', axis = 1)
            y = df['Stress1']


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
            req.session['acrs'] = accuracy
            req.session['pres'] = precession
            req.session['recs'] = recall
            req.session['fs'] = f1
            print(precession, accuracy,recall, f1,'uuuuuuuuuuuuuuuuuuuuuuuuuuu')
            if result_int >= 0 and result_int <= 14:
                messages.success(req, "Normal")
            elif result_int >= 16 and result_int <= 18:
                messages.warning(req, "Mild")
            elif result_int >= 20 and result_int <= 24:
                messages.warning(req, "Moderate")
            elif result_int >= 26 and result_int <= 32:
                messages.warning(req, "Severe")
            else:
                messages.warning(req, "Extremely Severe")


            req.session['ress'] = result_int
            context2 = {'acra': accuracy,'prea': precession,'f':f1,'reca':recall,'resa':result_int}

            

        return redirect("stress_result")
    return render(req, 'user/stress.html')

def user_depression(req):
    if req.method == 'POST':
        age = req.POST.get('age')
        RelationShip = req.POST.get('RelationShip')
        Family = req.POST.get('Family')
        Current = req.POST.get('Current')
        Education = req.POST.get('Education')
        Gender = req.POST.get('Gender')
        q1_nutrition = req.POST.get('q1_nutrition')
        q2_screentime = req.POST.get('q2_screentime')
        q3_screentime = req.POST.get('q3_screentime')
        q4_screentime = req.POST.get('q4_screentime')
        q5_screentime = req.POST.get('q5_screentime')
        q6_screentime = req.POST.get('q6_screentime')
        q7_frequency = req.POST.get('q7_frequency')


        print(age,RelationShip,Gender,Education,Current,Family,q1_nutrition,q2_screentime,q3_screentime,q4_screentime,q5_screentime,q6_screentime,q7_frequency, 'dataaaaaaaaaaaa')
        age = int(age)
        RelationShip = int(RelationShip)
        Family = int(Family)
        Current = int(Current)
        Education = int(Education)
        Gender = int(Gender)
        q1_nutrition = int(q1_nutrition)
        q2_screentime = int(q2_screentime)
        q3_screentime = int(q3_screentime)
        q4_screentime = int(q4_screentime)
        q5_screentime = int(q5_screentime)
        q6_screentime = int(q6_screentime)
        q7_frequency = int(q7_frequency)

            
        # print(type(age),x)
        # DATASET.objects.create(Age = age, Glucose = sex, BloodPressure = plasma_CA19_9, SkinThickness = creatinine, Insulin = lyve1, BMI = regb1, DiabetesPedigreeFunction = tff1)
        import pickle
        file_path = 'DASS21/lr_depression.pkl'  # Path to the saved model file

        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            res =loaded_model.predict([[age,RelationShip,Gender,Education,Current,Family,q1_nutrition,q2_screentime,q3_screentime,q4_screentime,q5_screentime,q6_screentime,q7_frequency]])
            print(type(res),"resssssssssssssssssssresssssssssssssss")


            result_int = int(res)
            result_int = result_int + 1
            print(res,'sadfffsadfasdfhsdfjhsioufgweifygascnxufurygfxoqnfufyw8ofer' )

            dataset = depression_Upload_dataset_model.objects.last()
            # print(dataset.Dataset)
            df=pd.read_csv(dataset.Dataset.path)
            X = df.drop('Depression1', axis = 1)
            y = df['Depression1']

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
            req.session['acrd'] = accuracy
            req.session['pred'] = precession
            req.session['recd'] = recall
            req.session['fd'] = f1
            print(precession, accuracy,recall, f1,'uuuuuuuuuuuuuuuuuuuuuuuuuuu')
            if result_int >= 0 and result_int <= 8:
                messages.success(req, "Normal")
            elif result_int >= 10 and result_int <= 12:
                messages.warning(req, "Mild")
            elif result_int >= 14 and result_int <= 20:
                messages.warning(req, "Moderate")
            elif result_int >= 22 and result_int <= 26:
                messages.warning(req, "Severe")
            else:
                messages.warning(req, "Extremely Severe")




            context2 = {'acr': accuracy,'pre': precession,'f':f1,'rec':recall,'res':result_int}
            req.session['resd'] = result_int

            # print(type(res), 'ttttttttttttttttttttttttt', context)
            print(res)
        return redirect("depression_result")
    return render(req, 'user/depression.html')



def userlogout(request):
    view_id = request.session["user_id"]
    user = UserDetails.objects.get(user_id = view_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(request, 'You are logged out..')
    # print(user.Last_Login_Time)
    # print(user.Last_Login_Date)
    return redirect('login')


def user_feedback(request):
    return render(request, 'user/feedback.html')

def user_result_anxiety(request):
    accuracy = request.session.get('acra')
    precession = request.session.get('prea')
    recall = request.session.get('reca')
    f1 = request.session.get('fa')
    x = request.session.get('resa')

    return render(request, "user/anxiety-result.html",{'accuracy':accuracy, 'precession':precession, 'recall': recall, 'f1':f1, 'res':x})


def user_result_Depression(request):
    accuracy = request.session.get('acrd')
    precession = request.session.get('pred')
    recall = request.session.get('recd')
    f1 = request.session.get('fd')
    x = request.session.get('resd')

    return render(request, "user/Depression-result.html",{'accuracy':accuracy, 'precession':precession, 'recall': recall, 'f1':f1, 'res':x})


def user_result_Stress(request):
    accuracy = request.session.get('acrs')
    precession = request.session.get('pres')
    recall = request.session.get('recs')
    f1 = request.session.get('fs')
    x = request.session.get('ress')
    print(accuracy,precession,recall,f1,x)

    return render(request, "user/stress-result.html",{'accuracy':accuracy, 'precession':precession, 'recall': recall, 'f1':f1, 'res':x})

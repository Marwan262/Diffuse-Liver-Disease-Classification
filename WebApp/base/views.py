from tkinter import Y
from django.forms import ImageField
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import mixins
from rest_framework.exceptions import APIException
from django.db import transaction
from operator import truediv
import SimpleITK as sitk    
from queue import Empty
import random
from rest_framework import status 
import numpy as np
from radiomics import glrlm, glcm, firstorder
from .forms import ImageForm
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from django.core.files.storage import FileSystemStorage
from skimage.measure import shannon_entropy
import string
from xmlrpc.client import boolean
import math
from django.shortcuts import redirect, render
from django.http import HttpResponse
from .models import Patient
from .models import Doctor
from .models import Admin
from .models import Report
from django.contrib.auth.models import User
from django.contrib import messages
from joblib import load
import sklearn
import pandas as pd

def loadmodel():
    models = {}
    classifiers = ['fatty_cirrhosis', 'normal_cirrhosis', 'normal_fatty']
    for name in classifiers:
        models[name] = [
            load(open(f"I:/_Marwan Documents/Grad/Django 2/fork/WebApp/models/{name}_mlp.joblib", 'rb')),
            load(open(f"I:/_Marwan Documents/Grad/Django 2/fork/WebApp/models/{name}_std.joblib", 'rb')),
            load(open(f"I:/_Marwan Documents/Grad/Django 2/fork/WebApp/models/{name}_cols.joblib", 'rb'))
        ]
    return models

def classify_model(data, model, std, cols):
    X = pd.DataFrame(std.transform(data[cols]), columns = cols, index = data.index)
    y_pred = model.predict(X) 
    pred=images_pred(y_pred)
    return pred
    
def classify_img(data, models):
    pred = {
        'normal': 0,
        'fatty': 0,
        'cirrhosis': 0
    }
    for key in models.keys():
        model, std, cols = models[key]
        pred[classify_model(data, model, std, cols)] += 1
    result = max(pred, key=pred.get)
    if pred[result] == 1:
        return "abstain"
    return result

def status(path):
    # try:
    name = 'I:\\_Marwan Documents\\Grad\\Django 2\\fork\\WebApp\\media\\scans\\' + str(path)
    img = sitk.ReadImage(name, sitk.sitkUInt8)
    models = loadmodel()
    data = pd.DataFrame(feature_extraction(img, [(155,94), (94,94)]))
    
    result = classify_img(data, models)
    return result
    # except ValueError as e: 
    #     return Response(e.args[0], 400) 

def images_pred(y_pred):
    count = 0
    prediction = {}

    for i in y_pred:
        if i not in prediction.keys():
            prediction[i]=1
        else: prediction[i]+=1

    return max(prediction, key=prediction.get)

def testform(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        
        result = status('s')
        if form.is_valid():
            img_obj = form.cleaned_data['scan']
            # form.save()
            # img_obj = form.instance
            return render(request, 'testform.html', {'form' : form, 'img_obj': img_obj, 'result' : result})
    form = ImageForm
    return render(request, 'testform.html', {'form': form})

def index(request):
    return render(request, 'index.html')
    
def index(request):
    return render(request, 'index.html')

def get_prediction(request, id):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            # return redirect('patients')
            path = form.cleaned_data['scan']
            form.save()
            result = status(path)
            patient = Patient.objects.get(id=id)
            request.session['result'] = result
            request.session['path'] = str(path)
            return redirect('addreport', id)
    form = ImageForm
    context = {'form' : form}
    return render(request, 'classify.html', {'form' : form})

def addreport(request, id):
    if request.method == 'POST':
        notes = request.POST.get('notes')
        classification = request.POST.get('classification')
        doctor = request.session['id']
        patient = id
        diagnosis = request.POST.get('diagnosis')
        Report.objects.create(notes=notes, diagnosis=diagnosis, classification=classification, image=request.session['path'], doctor=doctor, patient=patient)
        return redirect('patient', id)
    form = ImageForm
    patient = Patient.objects.get(id=id)
    context = {'patient' : patient, 'form' : form}
    return render(request, 'addReport.html', context) 
 
def report(request, id):
    if request.method == 'POST':
        # return render(request, 'index.html')
        notes = request.POST.get('notesE')
        classification = request.POST.get('classificationE')
        Report.objects.filter(id = id).update(notes = notes, classification = classification)
        return render(request, 'viewReport.html', {'report' : report}) 
    report = Report.objects.get(id = id)
    patient = Patient.objects.get(id=report.patient)
    doctor = Doctor.objects.get(id=report.doctor)
    context = {'patient' : patient, 'doctor' : doctor, 'report' : report}
    return render(request, 'viewReport.html', context) 
           
def about(request):
    return render(request, 'about.html')

def profile(request):
    if 'role' in request.session:
        if request.session['role'] == 'patient':
            patient = Patient.objects.get(id = request.session['id'])
            reports = Report.objects.filter(patient = request.session['id'])
            return render(request, 'profile.html', {'patient' : patient, 'reports' : reports}) 
        else :
            return redirect('index')
    else:
        return render(request, 'login.html')
      

def patients(request):
    if 'role' in request.session:
        if request.session['role'] == 'doctor':
            patients = Patient.objects.filter(assigned_doctor=request.session['id'])
            context = {'patients': patients}
            return render(request, 'patients.html', context) 

        elif request.session['role'] == 'admin':
            patients = Patient.objects.all()
            context = {'patients': patients}
            return render(request, 'patients.html', context)

        if request.session['role'] != 'doctor' and request.session['role'] != 'admin':
            return render(request, 'index.html')
    else:
        return render(request, 'index.html')

def doctors(request):
    doctors = Doctor.objects.all()
    context = {'doctors': doctors}
    return render(request, 'doctors.html', context)       
       
def patient(request, id):
    if request.method == 'POST':
        if 'newpass' in request.POST:
            newpass = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
            request.session['newpass'] = newpass
            return redirect('patient', id)
        elif 'confirmpass' in request.POST:
            if 'newpass' in request.session:
                Patient.objects.filter(id=id).update(password=request.session['newpass'])
                del request.session['newpass']
            else:
                return redirect('patient', id)
    patient = Patient.objects.filter(id=id)
    reports = Report.objects.filter(patient=id)
    # doctors = Doctor.objects.filter(id=reports[0].doctor)
    context1 = {'patients': patients}
    context2 = {'reports': reports}
    return render(request, 'viewPatient.html', {'patient':patient[0], 'reports':reports}) 
       
def addpatient(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        birthdate = request.POST.get('birthdate')
        phoneno = request.POST.get('phoneno')
        error = False
        if not name or not birthdate or not phoneno:
            messages.error(request, 'Please fill in all fields.')
            error = True
        if any(char.isdigit() for char in name):
            messages.error(request, 'Name cannot contain numbers.')
            error = True
        if len(phoneno)< 10 and phoneno.isnumeric():
            messages.error(request, 'Please enter a valid phone number.')   
            error = True
        if not error:
            doctor = Doctor.objects.get(id=request.session['id'])
            N = 8
            password = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
            x = name.replace(" ", "")
            username = x.lower()

            query = Patient(patient_name=name, birth_date=birthdate, phone_num=phoneno, password=password, username=username, assigned_doctor=doctor)
            query.save()
            return redirect('patients')
        
    return render(request, 'addPatient.html')

def adddoctor(request):
    if request.method == 'POST':
        name = request.POST.get('fullname')
        username = name.replace(" ", "")
        N = 6
        password = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

        query = Doctor(name=name, username=username, password=password)
        query.save()
        return redirect('doctors')
    return render(request, 'addDoctor.html')    

def deletepatient(request, id):
    Patient.objects.filter(id=id).delete()

    return redirect('patients')

def deletedoctor(request, id):
    Doctor.objects.filter(id=id).delete()

    return redirect('doctors')    

def deletereport(request, id):
    Report.objects.filter(id=id).delete()

    return redirect('patients')    

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            patient = Patient.objects.get(username=username)
            if patient.password == password:
                request.session['role'] = 'patient'
                request.session['id'] = patient.id
                request.session['name'] = patient.patient_name
                request.session['loggedin'] = True
                return redirect('profile')
        except Patient.DoesNotExist:
            patient = None

        if patient == None:
            try:
                doctor = Doctor.objects.get(username=username)
                if doctor.password == password:
                    request.session['is_doctor'] = True
                    request.session['role'] = 'doctor'
                    request.session['id'] = doctor.id
                    request.session['name'] = doctor.name
                    request.session['loggedin'] = True
                    return redirect('index')
            except Doctor.DoesNotExist:
                doctor = None

        if doctor == None:
            try:
                admin = Admin.objects.get(username=username)
                if admin.password == password:
                    request.session['role'] = 'admin'
                    request.session['name'] = admin.name
                    request.session['id'] = admin.id
                    request.session['loggedin'] = True
                    return redirect('index')
            except Admin.DoesNotExist:
                admin = None 

        if admin == None:
            # return redirect('login')
            messages.error(request, 'Incorrect credentials.')
    if 'loggedin' in request.session:
        return render(request, 'index.html')
    else :
        return render(request, 'login.html')    

def logout(request):
    try:
        del request.session['role']
        del request.session['loggedin']
        del request.session['id']
        del request.session['report']
        del request.session['newpass']
    except KeyError:
        pass
    return redirect('login')

def dist(p, q):
    return np.linalg.norm(np.array(p)-np.array(q)) 

def get_length(img, mask):
    # top right, bottom left
    tr_distance = []
    bl_distance = []

    # top left, bottom right
    tl_distance = []
    br_distance = []

    for x, y in mask:
        
        tr_distance.append(dist([0, img.shape[1]], [x + 32, y]))
        bl_distance.append(dist([img.shape[0], 0], [x, y + 32]))

        tl_distance.append(dist([0, 0], [x, y]))
        br_distance.append(dist(img.shape, [x + 32, y + 32]))

    top_right = mask[tr_distance.index(min(tr_distance))]
    bottom_left = mask[bl_distance.index(min(bl_distance))]

    top_left = mask[tl_distance.index(min(tl_distance))]
    bottom_right = mask[br_distance.index(min(br_distance))]

    return max(dist(top_right, bottom_left), dist(top_left, bottom_right))

def extract_roi(img, start , size = (32,32)):
    img = sitk.GetArrayFromImage(img)
    roi = img[start[0]:start[0]+size[0],start[1]:start[1]+size[1]]
    mask = np.zeros(img.shape)
    mask[start[0]:start[0]+size[0],start[1]:start[1]+size[1]] = 1
    return roi, mask


def feature_extraction(img, roi_pos):
    roi_mask_arr = []
    for pos in roi_pos:
        roi_mask_arr.append(extract_roi(img, pos))

    # 0 45 90 135 degrees
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    da_dict = {
        0: "d1_0",
        1: "d1_45",
        2: "d1_90",
        3: "d1_135",

        4: "d2_0",
        5: "d2_45",
        6: "d2_90",
        7: "d2_135",

        8: "d3_0",
        9: "d3_45",
        10: "d3_90",
        11: "d3_135",

    }

    length = get_length(sitk.GetArrayFromImage(img), roi_pos)

    feat_arr = []
    for roi, mask in roi_mask_arr:
        features = {}

        glcm_mtx = greycomatrix(roi, distances = [1,2,3], angles = angles, levels = 256)
        con = greycoprops(glcm_mtx, 'contrast').flatten()
        hom = greycoprops(glcm_mtx, 'homogeneity').flatten()
        en = greycoprops(glcm_mtx, 'energy').flatten()
        corr = greycoprops(glcm_mtx, 'correlation').flatten()

        for j in range(len(da_dict)):
            features[f'contrast_{da_dict[j]}'] = con[j]
            features[f'homogeneity_{da_dict[j]}'] = hom[j]
            features[f'energy_{da_dict[j]}'] = en[j]
            features[f'correlation_{da_dict[j]}'] = corr[j]

        features[f'entropy'] = shannon_entropy(roi)

        features['length'] = length

        # features[f'mean'] = np.mean(roi)
        # features[f'variance'] = np.var(roi)

        # pyradiomics
        mask = sitk.GetImageFromArray(mask)
        # First Order features
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(img, mask)
        # firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableAllFeatures()
        results = firstOrderFeatures.execute()
        for col in results.keys():
            features[col] = results[col].item()

        # GLCM features
        glcmFeatures = glcm.RadiomicsGLCM(img, mask)
        glcmFeatures.enableAllFeatures()
        results = glcmFeatures.execute()
        for col in results.keys():
            features[col] = results[col].item()
        #
        # GLRLM features
        glrlmFeatures = glrlm.RadiomicsGLRLM(img, mask)
        glrlmFeatures.enableAllFeatures()
        results = glrlmFeatures.execute()
        features['LongRunEmphasis'] = results['LongRunEmphasis'].item()
        features['RunPercentage'] = results['RunPercentage'].item()
        for col in results.keys():
            features[col] = results[col].item()

        feat_arr.append(features)

    return feat_arr

def build_dataframe(images):
    # dataframe consists of features of 1 ROI per image
    # column name roiNum_feature
    data = pd.DataFrame()

    for name, img, cls, mask in images:
        feat_arr = feature_extraction(img, mask)
        count = 1
        for row in feat_arr:
            row['name'] = name
            count += 1
            row['target'] = cls
            data = data.append(row,ignore_index=True)
    return data
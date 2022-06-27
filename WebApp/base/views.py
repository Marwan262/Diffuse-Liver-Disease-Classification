from datetime import datetime
import re
import time
from tkinter import Y
from django.forms import ImageField
from rest_framework.response import Response
from rest_framework import viewsets
from pathlib import Path
from os import path
from rest_framework import mixins
from rest_framework.exceptions import APIException
from django.db import transaction
from operator import truediv
import SimpleITK as sitk    
from queue import Empty
import random
from rest_framework import status 
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from radiomics import glrlm, glcm, firstorder

from WebApp.settings import BASE_DIR
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
import torch
from .models import Report
from django.contrib.auth.models import User
from django.contrib import messages
from joblib import load
import sklearn
import logging
import pandas as pd

# returns index page
def index(request):
    return render(request, 'index.html')

# compiles data to be written in pdf page, returns pdf page
def pdf(request, id):
    report = Report.objects.get(id = id)
    doctor = Doctor.objects.get(id = report.doctor)
    patient = Patient.objects.get(id = report.patient)

    today = datetime.today()
    age = today.year - patient.birth_date.year - ((today.month, today.day) < (patient.birth_date.month, patient.birth_date.day))

    return render(request, "pdf.html", {'report': report, 'patient' : patient ,'doctor' : doctor, 'age' : age})

# returns index/home page
def index(request):
    return render(request, 'index.html')

# returns addReport page, submits add report form into database
def addreport(request, id):
    if 'role' in request.session:
        if request.session['role'] == 'doctor':
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

# returns viewReport page, submits edited report to database
def report(request, id):
    if 'role' in request.session:
        if request.session['role'] in ('patient', 'doctor', 'admin'):
            report = Report.objects.get(id = id)
            patient = Patient.objects.get(id=report.patient)
            doctor = Doctor.objects.get(id=report.doctor)
            if request.method == 'POST':
                logging.debug("post")
                notes = request.POST.get('notesE')
                classification = request.POST.get('classificationE')
                Report.objects.filter(id = id).update(notes = notes, classification = classification)
                return render(request, 'viewReport.html', {'report' : report, 'patient' : patient, 'doctor': doctor}) 
        
            context = {'patient' : patient, 'doctor' : doctor, 'report' : report}
            return render(request, 'viewReport.html', context) 

def checkEmail(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if(re.fullmatch(regex, email)):
        return True

# returns profile page
def profile(request):
    if 'role' in request.session:
        if request.session['role'] == 'patient':
            if request.method == 'POST':
                name = request.POST.get('editName')
                email = request.POST.get('editEmail')
                phone = request.POST.get('editPhone')
                if not any(str.isdigit(c) for c in name) and phone.isnumeric() and  checkEmail(email):
                    Patient.objects.filter(id = request.session['id']).update(email = email, patient_name = name, phone_num = phone)
                else:
                    patient = Patient.objects.get(id = request.session['id'])
                    reports = Report.objects.filter(patient = request.session['id'])
                    if len(reports) > 0:
                        id = reports[0].doctor
                        doctor = Doctor.objects.get(id=id).name
                    return render(request, 'profile.html', {'patient' : patient, 'doctor' : doctor, 'reports' : reports}) 
            patient = Patient.objects.get(id = request.session['id'])
            reports = Report.objects.filter(patient = request.session['id'])
            conditions = patient.medical_conditions.split(',')
            if len(reports) > 0:
                id = reports[0].doctor
                doctor = Doctor.objects.get(id=id).name  
            return render(request, 'profile.html', {'patient' : patient, 'doctor' : doctor, 'conditions' : conditions, 'reports' : reports}) 
        else :
            return redirect('index')
    else:
        return render(request, 'login.html')

# returns patients page
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
# returns doctors page
def doctors(request):
    if 'role' in request.session:
        if request.session['role'] == 'admin':
            doctors = Doctor.objects.all()
            context = {'doctors': doctors}
            return render(request, 'doctors.html', context)       

# returns viewDoctor page
def doctor(request, id):
    if 'role' in request.session:
        if request.session['role'] == 'admin':
            doctor = Doctor.objects.get(id=id)
            patients = Patient.objects.filter(assigned_doctor=id)   
            return render(request, 'viewDoctor.html', {'patients':patients, 'doctor' : doctor}) 

# returns viewPatient page, submits new password to database
def patient(request, id):
    if 'role' in request.session:
        if request.session['role'] in ('doctor', 'admin'):
            if request.method == 'POST':
                if 'newpass' in request.POST:
                    newpass = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
                    Patient.objects.filter(id=id).update(password=newpass)
                    return redirect('patient', id)
                elif 'addCondition' in request.POST:
                    newCondition = request.POST.get('newCondition')
                    patient = Patient.objects.get(id=id)
                    if patient.medical_conditions:
                        conditions = patient.medical_conditions.split(',')
                        conditions.append(newCondition)
                        conditions = ','.join(conditions)
                        Patient.objects.filter(id = id).update(medical_conditions = conditions)
                    else:
                        Patient.objects.filter(id = id).update(medical_conditions = newCondition)
                    return redirect('patient', id)
            patient = Patient.objects.get(id=id)
            doc = patient.assigned_doctor
            reports = Report.objects.filter(patient=id)
            doctor = Doctor.objects.get(name=doc)
            if patient.medical_conditions:
                conditions = patient.medical_conditions.split(',')
            # if len(reports) > 0:
            #     doctor_id = reports[0].doctor
            #     doctor = Doctor.objects.get(id=doctor_id).name  
            # else:
            #     doctor = ''
            if patient.medical_conditions:
                return render(request, 'viewPatient.html', {'patient':patient, 'conditions' : conditions, 'reports':reports, 'doctor' : doctor}) 
            else:
                return render(request, 'viewPatient.html', {'patient':patient, 'reports':reports, 'doctor' : doctor}) 

# returns addPatient page, submits new patient to database
def addpatient(request):
    if 'role' in request.session:
        if request.session['role'] == 'doctor':
            if request.method == 'POST':
                name = request.POST.get('name')
                email = request.POST.get('email')
                birthdate = request.POST.get('birthdate')
                phoneno = request.POST.get('phoneno')
                error = False
                if not name or not birthdate or not phoneno:
                    messages.error(request, 'Please fill in all fields.')
                    error = True
                if any(char.isdigit() for char in name):
                    messages.error(request, 'Name cannot contain numbers.')
                    error = True
                if len(phoneno) < 11 or not phoneno.isnumeric():
                    messages.error(request, 'Please enter a valid phone number.')   
                    error = True
                if not error:
                    doctor = Doctor.objects.get(id=request.session['id'])
                    N = 8
                    password = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
                    x = name.replace(" ", "")
                    username = x.lower()

                    query = Patient(patient_name=name, email=email, birth_date=birthdate, phone_num=phoneno, password=password, username=username, assigned_doctor=doctor)
                    query.save()
                    return redirect('patients')
            return render(request, 'addPatient.html')

# returns addDoctor page, submits new doctor to database
def adddoctor(request):
    if 'role' in request.session:
        if request.session['role'] == 'admin':
            if request.method == 'POST':
                name = request.POST.get('name')
                username = name.replace(" ", "")
                N = 6
                password = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

                query = Doctor(name=name, username=username, password=password)
                query.save()
                return render(request, 'doctors.html')
            return render(request, 'addDoctor.html')    

# deletes patient from database
def deletepatient(request, id):
    if 'role' in request.session:
        if request.session['role'] == 'doctor':
            Patient.objects.filter(id=id).delete()
            return redirect('patients')

# deletes doctor from database
def deletedoctor(request, id):
    if 'role' in request.session:
        if request.session['role'] == 'admin':
            Doctor.objects.filter(id=id).delete()
            return redirect('doctors')    

def archive(request, id):
    if 'role' in request.session:
        if request.session['role'] == 'doctor':
            Patient.objects.filter(id=id).update(is_archived=True)
            return redirect('patient', id)

def restore(request, id):
    if 'role' in request.session:
        if request.session['role'] == 'doctor':
            Patient.objects.filter(id=id).update(is_archived=False)
            return redirect('patient', id)

def deletecondition(request, condition, id):
    if 'role' in request.session:
        if request.session['role'] == 'doctor':
            patient = Patient.objects.get(id=id)
            conditions = patient.medical_conditions.split(',')
            conditions.remove(condition)
            conditions = ','.join(conditions)
            Patient.objects.filter(id = id).update(medical_conditions = conditions)
            reports = Report.objects.filter(patient=id)
            if patient.medical_conditions:
                conditions = patient.medical_conditions.split(',')
            if len(reports) > 0:
                doc_id = reports[0].doctor
                doctor = Doctor.objects.get(id=doc_id).name  
            else:
                doctor = ''
            return redirect('patient', id)
            return render(request, 'viewPatient.html', {'patient':patient, 'conditions' : conditions, 'reports':reports, 'doctor' : doctor})
            if patient.medical_conditions:
                return render(request, 'viewPatient.html', {'patient':patient, 'conditions' : conditions, 'reports':reports, 'doctor' : doctor}) 
            else:
                return render(request, 'viewPatient.html', {'patient':patient, 'reports':reports, 'doctor' : doctor}) 

# deletes report from database
def deletereport(request, id):
    if 'role' in request.session:
        if request.session['role'] in ('doctor', 'admin'):
            Report.objects.filter(id=id).delete()

            return redirect('patients')    

# returns login page, assigns role, redirects to home/index page
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
                return redirect('index')
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
            messages.error(request, 'Incorrect credentials.')
            return render(request, 'login.html')  
    if 'loggedin' in request.session:
        return render(request, 'index.html')
    else :
        return render(request, 'login.html')    

# removes session variables, redirects to login page
def logout(request):
    try:
        del request.session['role']
        del request.session['loggedin']
        del request.session['id']
        del request.session['report']
        del request.session['newpass']
    except KeyError:
        pass
    return redirect('index')

# returns error page
def error(request):
    return render(request, 'error.html')

# returns classify page, passes image through model, returns classification
def get_prediction(request, id):
    if 'role' in request.session:
        if request.session['role'] == 'doctor':
            if request.method == 'POST':
                form = ImageForm(request.POST, request.FILES)
                if form.is_valid():
                    path = form.cleaned_data['scan']
                    form.save()
                    result = status(path)
                    if result is None:
                        return redirect('error')
                        
                    request.session['result'] = result
                    request.session['path'] = str(path)
                    return redirect('addreport', id)
            form = ImageForm
            return render(request, 'classify.html', {'form' : form})

# passes image through image segmentation and image prediction
def status(path):
    # try:
    name = f'{BASE_DIR}\\media\\scans\\' + str(path)
    img = sitk.ReadImage(name, sitk.sitkUInt8)
    cnn_model = Model()
    np_image = sitk.GetArrayFromImage(img)
    try:
        mask,anno = cnn_model(np_image.reshape(np_image.shape[0], np_image.shape[1], 1))
    except:
        return None

    models = loadmodel()
    roi_pos = split_image(mask)
    data = pd.DataFrame(feature_extraction(img, roi_pos))
    
    result = classify_img(data, models)
    return result
    
    # except ValueError as e: 
    #     return Response(e.args[0], 400) 

# splits mask into 32x32 size ROIs
def split_image(mask, M=32, N=32):
    
    roi_pos = []
    
    for x in range(0,mask.shape[0],M):
        for y in range(0,mask.shape[1],N):
            if 0 not in mask[x:x+M,y:y+N]:
                roi_pos.append((x,y))
    return roi_pos

# loads joblib models
def loadmodel():
    models = {}
    classifiers = ['fatty_cirrhosis', 'normal_cirrhosis', 'normal_fatty']
    for name in classifiers:
        models[name] = [
            load(open(f"{BASE_DIR}/models/{name}_mlp.joblib", 'rb')),
            load(open(f"{BASE_DIR}/models/{name}_std.joblib", 'rb')),
            load(open(f"{BASE_DIR}/models/{name}_cols.joblib", 'rb'))
        ]
    return models

# returns image prediction
def classify_model(data, model, std, cols):
    X = pd.DataFrame(std.transform(data[cols]), columns = cols, index = data.index)
    y_pred = model.predict(X) 
    pred=images_pred(y_pred)
    return pred

# returns ROI prediction
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

# returns image prediction
def images_pred(y_pred):
    prediction = {}

    for i in y_pred:
        if i not in prediction.keys():
            prediction[i]=1
        else: prediction[i]+=1

    return max(prediction, key=prediction.get)

# CNN segmentation model class
class Model:
    # class constructor, initializes CNN model
    def __init__(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE='cpu'
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = f"{BASE_DIR}/models/model_final.pth"

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.predictor = DefaultPredictor(cfg)

    # converts image into required detectron format
    def _convert_to_segments_format(self, image, outputs):
        segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
        annotations = []
        counter = 1
        instances = outputs['instances']
        for i in range(len(instances.pred_classes)):
            category_id = int(instances.pred_classes[i])
            instance_id = counter
            mask = instances.pred_masks[i].cpu()
            segmentation_bitmap[mask] = instance_id
            annotations.append({'id': instance_id, 'category_id': category_id})
            counter += 1
        return segmentation_bitmap, annotations

    # segments image
    def __call__(self, image):
        image = np.array(image)
        outputs = self.predictor(image)
        label, label_data = self._convert_to_segments_format(image, outputs)

        return label, label_data

# returns distance between two points
def dist(p, q):
    return np.linalg.norm(np.array(p)-np.array(q)) 

# returns diagonal liver length
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

# extracts 32x32 area of image, creates mask of area
def extract_roi(img, start , size = (32,32)):
    img = sitk.GetArrayFromImage(img)
    roi = img[start[0]:start[0]+size[0],start[1]:start[1]+size[1]]
    mask = np.zeros(img.shape)
    mask[start[0]:start[0]+size[0],start[1]:start[1]+size[1]] = 1
    return roi, mask

# extracts features from image
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

# builds dataframe from extracted features
def build_dataframe(images):
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
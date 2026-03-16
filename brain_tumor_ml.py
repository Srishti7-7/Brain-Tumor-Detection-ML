import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox

from skimage.feature import hog, graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pickle

# ==============================
# DATASET PATH
# ==============================

train_path = "archive (2)/Training"

# ==============================
# GLOBAL VARIABLES
# ==============================

classes = os.listdir(train_path)

data = []
labels = []

# ==============================
# GLCM FEATURES
# ==============================

def extract_glcm_features(image):

    glcm = graycomatrix(image,
                        distances=[1],
                        angles=[0],
                        levels=256,
                        symmetric=True,
                        normed=True)

    contrast = graycoprops(glcm,'contrast')[0,0]
    energy = graycoprops(glcm,'energy')[0,0]
    homogeneity = graycoprops(glcm,'homogeneity')[0,0]
    correlation = graycoprops(glcm,'correlation')[0,0]

    return [contrast,energy,homogeneity,correlation]


# ==============================
# IMAGE PROCESSING
# ==============================

def process_images(path):

    for category in os.listdir(path):

        category_path = os.path.join(path,category)

        label = classes.index(category)

        for img in os.listdir(category_path):

            img_path = os.path.join(category_path,img)

            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image,(128,128))

            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            hog_features = hog(gray,
                               pixels_per_cell=(16,16),
                               cells_per_block=(2,2),
                               visualize=False)

            glcm_features = extract_glcm_features(gray)

            features = np.hstack((hog_features,glcm_features))

            data.append(features)
            labels.append(label)


# ==============================
# TRAIN MODEL FUNCTION
# ==============================

def train_model():

    global scaler,pca,best_model

    print("Processing images...")

    process_images(train_path)

    X = np.array(data)
    y = np.array(labels)

    print("Dataset Shape:",X.shape)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=200)
    X = pca.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y)

    models = {

    "KNN":KNeighborsClassifier(n_neighbors=7),

    "SVM":SVC(kernel='rbf',probability=True),

    "Random Forest":RandomForestClassifier(n_estimators=200,max_depth=15),

    "Decision Tree":DecisionTreeClassifier(max_depth=10),

    "Logistic Regression":LogisticRegression(max_iter=2000),

    "Naive Bayes":GaussianNB()

    }

    accuracies = {}

    for name,model in models.items():

        print("\nTraining:",name)

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test,pred)

        accuracies[name] = acc

        print("Accuracy:",acc)

        print(classification_report(y_test,pred))

        cm = confusion_matrix(y_test,pred)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
        plt.title("Confusion Matrix - "+name)
        plt.show()

    # BAR GRAPH

    plt.figure(figsize=(8,5))
    plt.bar(accuracies.keys(),accuracies.values())
    plt.title("Accuracy Comparison")
    plt.show()

    # LINE GRAPH

    plt.figure(figsize=(8,5))
    plt.plot(list(accuracies.keys()),list(accuracies.values()),marker='o')
    plt.title("Accuracy Line Graph")
    plt.grid(True)
    plt.show()

    # ROC CURVE

    y_test_bin = label_binarize(y_test,classes=range(len(classes)))

    plt.figure(figsize=(8,6))

    for name,model in models.items():

        probs = model.predict_proba(X_test)

        fpr,tpr,_ = roc_curve(y_test_bin.ravel(),probs.ravel())

        roc_auc = auc(fpr,tpr)

        plt.plot(fpr,tpr,label=name+" AUC="+str(round(roc_auc,2)))

    plt.plot([0,1],[0,1],'k--')

    plt.legend()
    plt.title("ROC Curve Comparison")
    plt.show()

    # BEST MODEL

    best_model_name = max(accuracies,key=accuracies.get)

    best_model = models[best_model_name]

    print("Best Model:",best_model_name)

    # SAVE MODEL

    pickle.dump(best_model,open("brain_tumor_model.pkl","wb"))
    pickle.dump(scaler,open("scaler.pkl","wb"))
    pickle.dump(pca,open("pca.pkl","wb"))

    messagebox.showinfo("Training Completed",
                        "Model trained and saved successfully")


# ==============================
# IMAGE PREPROCESS
# ==============================

def preprocess_image(path):

    img = cv2.imread(path)

    img = cv2.resize(img,(128,128))

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    hog_features = hog(gray,
                       pixels_per_cell=(16,16),
                       cells_per_block=(2,2),
                       visualize=False)

    glcm_features = extract_glcm_features(gray)

    features = np.hstack((hog_features,glcm_features))

    scaler = pickle.load(open("scaler.pkl","rb"))
    pca = pickle.load(open("pca.pkl","rb"))

    features = scaler.transform([features])
    features = pca.transform(features)

    return features


# ==============================
# PREDICT MRI FUNCTION
# ==============================

def predict_mri():

    try:
        model = pickle.load(open("brain_tumor_model.pkl","rb"))
    except:
        messagebox.showerror("Error","Please train the model first")
        return

    file_path = filedialog.askopenfilename(title="Select MRI Image")

    if file_path == "":
        return

    features = preprocess_image(file_path)

    prediction = model.predict(features)

    result = classes[prediction[0]]

    messagebox.showinfo("Prediction Result","Tumor Type: "+result)


# ==============================
# MAIN MENU GUI
# ==============================

window = tk.Tk()

window.title("Brain Tumor Detection System")

window.geometry("400x300")

title = tk.Label(window,
                 text="Brain Tumor Detection",
                 font=("Arial",18))

title.pack(pady=30)

train_btn = tk.Button(window,
                      text="Train Model",
                      command=train_model,
                      width=20,
                      height=2)

train_btn.pack(pady=10)

predict_btn = tk.Button(window,
                        text="Predict MRI Image",
                        command=predict_mri,
                        width=20,
                        height=2)

predict_btn.pack(pady=10)

window.mainloop()
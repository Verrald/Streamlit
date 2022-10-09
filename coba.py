from tkinter import Y
import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, multilabel_confusion_matrix

#--------- HIDE STREAMLIT STYLE ----------#
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html = True)

st.title("Klasifikasi Data Menggunakan Machine Learning")
st.write("""
# Menggunakan Algoritma Machine Learning dan Dataset yang Berbeda
Website ini dibuat untuk pembelajaran, serta latihan saja.
""")

jenis_dataset = st.sidebar.selectbox("Pilih Datasets", 
                            ["Iris",
                             "Breast Cancer", 
                             "Diabetes", 
                             "Wine", 
                             "Digits"])

st.write(f"Datasets yang digunakan adalah {jenis_dataset}")

jenis_algoritma = st.sidebar.selectbox("Pilih Algoritma yang ingin digunakan",
                              ["SVM",
                               "KNN",
                               "Gaussian Naive Bayes",
                               "Multinomial Naive Bayes",
                               "Decision Trees"
                               #"Random Forest",
                               #"MLP"
                               ])

st.write(f"Algoritma yang digunakan adalah {jenis_algoritma}")

@st.cache(suppress_st_warning=True)
def pilih_datasets(jenis):
    data = None
    if jenis == "Iris":
        data = datasets.load_iris()
    elif jenis == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif jenis == "Diabetes":
        data = datasets.load_diabetes()
    elif jenis == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_digits()
    X = data.data
    y = data.target
    return X, y

X, y = pilih_datasets(jenis_dataset)

st.write(f"Jumlah data dan features datasets (jumlah, features):", X.shape)
st.write(f"Jumlah kelas:", len(np.unique(y)))


def tambah_parameter(nama_algoritma):
    params = dict()
    if nama_algoritma == "SVM":
        kernel = st.selectbox('kernel', 
                              ["linear",
                               "poly",
                               "rbf",
                               "sigmoid"])
        params["kernel"] = kernel
        C = st.number_input('Coefficient SVM Classifier', min_value = 1, step = 10, max_value = 1000)
        params["coef"] = C
        gamma = st.selectbox('gamma',
                             ["scale",
                              "auto"])
        params["gamma"] = gamma
        dfs = st.selectbox('Decision Function Shape',
                           ["ovo",
                            "ovr"])
        params["dfs"] = dfs
    elif nama_algoritma == "KNN":
        n_neighbors = st.slider('N Neighbors', 1, 30)
        params["n_neighbors"] = n_neighbors
        alg = st.selectbox('Algorithm used to compute the nearest neighbors',
                            ["auto",
                             "ball_tree",
                             "kd_tree",
                             "brute"])
        params["algo_n"] = alg
    elif nama_algoritma == "Gaussian Naive Bayes":
         var_smoothing = 1e-9
         params['var'] = var_smoothing 
    elif nama_algoritma == "Multinomial Naive Bayes":
        alpha = st.number_input('Alpha (smoothig parameter)', 1, 20)
        params["alpha"] = alpha
    elif nama_algoritma == "Decision Trees":
        criterion = st.selectbox('Criterion',
                                 ["gini",
                                  "entropy",
                                  "log_loss"])
        params["criterion"] = criterion
        splitter = st.selectbox('Splitter',
                                ["best",
                                 "random"])
        params["splitter"] = splitter
        max_features = st.selectbox('Max features', 
                                    ["None",
                                     "auto",
                                     "sqrt",
                                     "log2"])
        params["max_features"] = max_features
    return params
    
params = tambah_parameter(jenis_algoritma)

def pilih_klasifikasi(nama_algoritma, params):
    algo = None
    if nama_algoritma == "SVM":
        algo = SVC(kernel = params['kernel'], C= params['coef'], gamma = params['gamma'], decision_function_shape = params['dfs'])
    elif nama_algoritma == "KNN":
        algo = KNN(n_neighbors = params['n_neighbors'], algorithm = params['algo_n'])
    elif nama_algoritma == "Gaussian Naive Bayes":
        algo = GaussianNB()
    elif nama_algoritma == "Multinomial Naive Bayes":
        algo = MultinomialNB(alpha = params['alpha'])
    elif nama_algoritma == "Decision Trees":
        algo = DecisionTreeClassifier(criterion = params['criterion'], splitter = params['splitter'], max_features = params['max_features'])
    return algo

clf = pilih_klasifikasi(jenis_algoritma, params)

## klasifikasi
if st.button("RUN Classifications"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = clf.fit(X_train, np.ravel(y_train))
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Akurasi {acc*100}%")
    st.write(f"Hasil klasifikasi menunjukkan nilai akurasi dengan nilai {acc*100}%")
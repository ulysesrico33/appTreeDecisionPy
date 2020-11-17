import pandas as pd
import openpyxl
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
from datetime import datetime
import psutil
import time
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

 

def obtenerArbolDec(dataSet,op):
    if op==1:
       print('Elaborando árbol de decisión con entropía...')
    else:
        print('Elaborando árbol de decisión...')
    #Paso 1. Obtener variables dependientes e independientes (X:Independiente, Y:Dependiente)
    #En Python, como nomenclatura, "Feature columns"="Independent (X)", "Target Columns"="Dependent(Y)"
    print('Establenciendo variables independientes (X) y dependientes (Y)...')
    col_indep = ['Desc_curso','carrera']
    X = dataSet[col_indep] 
    col_dep = ['NOTA_CURSO<12']
    Y = dataSet[col_dep]
    #Paso 2: Dividir el dataset original en training set y test set
    print('Obteniendo set de Entrenamiento y pruebas...')
    #Para este ejemplo el los tamaños son: 30% test, 70% training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    imprimirEstadistica()
    print('Iniciando Proceso de árbol de decisión...')
    #Paso 4: Iniciallizar árbol de deicisón
    if op==1:
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    else:
        clf = DecisionTreeClassifier()

    print('Entrenando modelo...')
    imprimirEstadistica()
    #Paso 5: Entrenar modelo
    clf = clf.fit(X_train,Y_train)
    #Predicción para el dataset test
    y_pred = clf.predict(X_test)
    # Midiendo exactitud del modelo
    print("Exactitud:",metrics.accuracy_score(Y_test, y_pred))
    #Gráfica
    chartDecTree1(clf,col_indep)
    print('-----Fin arbol de decisión------')

def obtenerBayes(dataSet):
    print('Elaborando proceso de Bayes...')
    print('Establenciendo variables independientes (X) y dependientes (Y)...')
    col_indep = ['Desc_curso','carrera']
    X = dataSet[col_indep] 
    col_dep = ['NOTA_CURSO<12']
    Y = dataSet[col_dep]
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=109)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Exactitud:",metrics.accuracy_score(y_test, y_pred))
    print('-----Fin proceso de Bayes------')  

def verPrimerosDatos(dataSet):
    print(dataSet.head()) 

#url to replace values in dataframe: https://datatofish.com/replace-values-pandas-dataframe/
#Nota: Poner las columnas que se deseen editar igual que en la hoja.
def procesoEditarDataSet(archivo):
    print('Editando dataset...') 
    #Obtener todas las hojas del archivo
    book = openpyxl.load_workbook(archivo)
    lsAllSheets=[]
    lsAllSheets=book.sheetnames
    lsAllSheets.remove('Sheet1')
    book.close()
    #Obtener el dataFrame principal (Sheet1)
    df_main = pd.read_excel(archivo, sheet_name='Sheet1') 
    #Reemplazar espeacios en blanco por ceros por cada columna
    for colName in df_main.columns:
        df_main[colName] = df_main[colName].apply(lambda x: 0 if x == ' ' else x)
    #Obtener un dataDrame temporal por cada Hoja del archivo excel
    for colName in df_main.columns:
        if colName in lsAllSheets:
            df_temp=pd.read_excel(archivo,sheet_name=colName)
            #Obtener valores current y new 
            lsCurrent=df_temp['current'].values.tolist()
            lsNew=df_temp['new'].values.tolist()
            df_main[colName] = df_main[colName].replace(lsCurrent,lsNew)
            
       
    return df_main 
    

def imprimirHora():
    print('Tiempo ahora:',str(datetime.now()))

def imprimirEstadistica():
    print('Ram usada:',str(psutil.virtual_memory().percent),'%')
    print('CPU usada:',str(psutil.cpu_percent()),'%')

def chartDecTree1(clf,feature_cols):
    print('Generando gráfica')   
    dot_data = StringIO()
    #Procedimiento para convertir los valores a string en class_names
    lsTemp=[]
    for c in clf.classes_:
        lsTemp.append(str(c))
    export_graphviz(clf, out_file=dot_data,  filled=True, rounded=True,special_characters=True, feature_names = feature_cols,class_names=lsTemp)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('clf.png')
    Image(graph.create_png())

def obtenerTiempoSegundos():
    return time.perf_counter()   






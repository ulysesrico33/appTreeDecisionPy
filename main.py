import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import utils as tool


def main():
    #df[nombreColumna][NúmeroFila] da el valor
    #print(df['Desc_curso'][0])
    tiempoInicial=tool.obtenerTiempoSegundos()
    print('Elige una opción:')
    print('1.Árbol de decisión 2.Bayes 3. Ver primeros datos')
    op=input()
    op=int(op)
    #Carga el archivo excel a un DataFrame
    #uevo Comentario
    tool.imprimirHora()
    tool.imprimirEstadistica()
    print('Leyendo excel...')
    df=tool.procesoEditarDataSet('data_v2.xlsx')
    if op==1:
        print('1.Entropía , 2. Sin entropía')
        op=input()
        op=int(op)
        tool.obtenerArbolDec(df,op)      
    if op==2:
        tool.obtenerBayes(df)
    if op==3:
        tool.verPrimerosDatos(df) 

    print('------Estadísticas finales---------')       
    tool.imprimirHora()    
    tiempoFinal=tool.obtenerTiempoSegundos()
    print(f'Tiempo total: {tiempoFinal - tiempoInicial:0.2f} segundos') 
    print('Total de datos procesados:',len(df))  
    print('------Fin de Estadísticas finales---------') 

       
    
if __name__=='__main__':
    main()


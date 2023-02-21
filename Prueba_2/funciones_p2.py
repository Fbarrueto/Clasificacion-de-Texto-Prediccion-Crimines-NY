import numpy as np
import geopandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def col_obj(df):
    '''
    Función permite separar las columnas que son del tipo objetos.
    -----------------------------------------------
    Parámetros:
    - df: DataFrame 
    -----------------------------------------------
    Return:
    - Devuelve DataFrame con objetos
    '''
    columnas_object = df.select_dtypes(['object']).columns
    return df[columnas_object]


def col_int(df):
    '''
    Función permite separar las columnas que son del tipo numérico.
    -----------------------------------------------
    Parámetros:
    - df: DataFrame 
    -----------------------------------------------
    Return:
    - Devuelve DataFrame con floats e integers
    '''
    columnas_int = df.select_dtypes(['int']).columns
    return df[columnas_int]


def col_float(df):
    '''
    Función permite separar las columnas que son del tipo numérico.
    -----------------------------------------------
    Parámetros:
    - df: DataFrame 
    -----------------------------------------------
    Return:
    - Devuelve DataFrame con floats e integers
    '''
    columnas_float = df.select_dtypes(['float']).columns
    return df[columnas_float]


def create_2obj(df):
    '''
    Función permite la creación de la segunda variable objetivo, que determina si un arresto
    fue realizado con violencia o no.
    -----------------------------------------------
    Parámetros:
    - df: DataFrame 
    -----------------------------------------------
    Return:
    - Devuelve un DataFrame con columna extra 'violence' que determina si un arresto fue o no realizado
    con violencia, 1 indica que si se realizó con violencia y 0 que fue de forma no violenta.
    '''
    df['violence'] = np.where((df['pf_hands'] == 'Y') | 
                            (df['pf_wall'] == 'Y') | 
                            (df['pf_grnd'] == 'Y') | 
                            (df['pf_drwep'] == 'Y') | 
                            (df['pf_ptwep'] == 'Y') | 
                            (df['pf_baton'] == 'Y') | 
                            (df['pf_hcuff'] == 'Y') | 
                            (df['pf_pepsp'] == 'Y') | 
                            (df['pf_other'] == 'Y'), 1, 0)
    return df


def bin_vecobj(df):
    df.arstmade = np.where(df.arstmade == 'Y',1,0)
    return df.arstmade


def del_col(df):
    '''
    Función permite eliminar columnas que cierto porcentaje de datos nulos
    -----------------------------------------------
    Parámetros:
    - df: DataFrame 
    -----------------------------------------------
    Return:
    - Devuelve DataFrame sin columnas con datos nulos sobre 60%.
    '''
    for columna in df.columns:
        if df[columna].isnull().sum()*100/df.shape[0] > 60:
            df = df.drop(columns=[columna])
    return df


def coordenates(df):
    df = geopandas.GeoDataFrame(df,geometry = geopandas.points_from_xy(df.xcoord, 
                                                                        df.ycoord))
    return df.drop(columns=['ycoord','xcoord','addrtyp','rescode','premtype','premname','addrnum','stname','stinter','crossst','aptnum','city','addrpct','sector','beat','post'])


def map_arstmade(df, year):
    ''' 
    Función combina las coordenadas de latitud y longitud entregadas en el dataframe, permitiendo
    identificar el lugar donde se realiza el arresto.
    ------------------------------------
    Parámetros:
    - df: DataFrame de los datos de las investigaciones y detenciones realizadas en NY. Debe contener
    'xcoord' y 'ycoord'.
    - year: año al que pertenece el DataFrame
    ------------------------------------
    Return:
    - Entrega mapa con las densidades de los arrestos realizados durante el año entregado.
    '''
    gdf = geopandas.GeoDataFrame(df,geometry = geopandas.points_from_xy(df.xcoord, 
                                                                        df.ycoord))
    gdf.arstmade = np.where(gdf.arstmade == 'N', 0, 1)
    gdf_arst = gdf.query('arstmade == 1')
    world = geopandas.read_file(geopandas.datasets.get_path('nybb'))
    ax = world[world.BoroName == ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']].plot(color='white', edgecolor='black')
    gdf_arst.plot(ax=ax, color='red', alpha=0.1)

    plt.title(f'Arrestos realizados en Nueva York {year}') 
    plt.ylabel('Latitud')
    plt.xlabel('Longitud')
    return plt.show();


def map_violence(df, year):
    ''' 
    Función combina las coordenadas de latitud y longitud entregadas en el dataframe, permitiendo
    identificar el lugar donde se ha realizado un arresto violento.
    ------------------------------------
    Parámetros:
    - df: DataFrame de los datos de las investigaciones y detenciones realizadas en NY. Debe contener
    'xcoord' y 'ycoord'.
    - year: año al que pertenece el DataFrame
    ------------------------------------
    Return:
    - Entrega mapa con las densidades de los arrestos violentos realizados durante el
    año entregado.
    '''
    gdf = geopandas.GeoDataFrame(df,geometry = geopandas.points_from_xy(df.xcoord, 
                                                                        df.ycoord))
    gdf_vio = gdf.query('violence == 1')
    world = geopandas.read_file(geopandas.datasets.get_path('nybb'))
    ax = world[world.BoroName == ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']].plot(color='white', edgecolor='black')
    gdf_vio.plot(ax=ax, color='blue', alpha=0.05)

    plt.title(f'Arrestos violentos realizados en Nueva York {year}') 
    plt.ylabel('Latitud')
    plt.xlabel('Longitud')
    return plt.show();


def objeto_num(df,rows):
    ''' 
    Función covierte variables categóricas a variables numéricas.
    -----------------------------
    Parámetros:
    - df: DataFrame con los datos de investigaciones y detenciones realizadas en NY.
    - rows: listado de columnas a convertir.
    -----------------------------
    Return:
    - Entrega DataFrame con las variables convertidas a números.
    '''
    for row in rows:
        df[row] = df[row].astype('category').cat.codes
    return df


def adj_age(df):
    '''
    La función entrega la edad del individuo acorde a la edad entregada realmente en el dataframe,
    si este valor no se encuentra entre los valores reales definidos por el dataset, 11 y 99 años,
    la nueva edad se determina acorde al año de nacimiento, extraído del 'dob'.
    Si ambos datos no entregan edad dentro del rango definido, se entrega un valor nulo.
    -------------------------------------------------
    Parámetros:
    - df: DataFrame que contiene la variable 'dob' y 'year'.
    --------------------------------------------------
    Return:
    Devuelve Dataframe con columna 'adj_age' eliminando las variables innecesarias.
    '''

    df['year_of_birth'] = df.astype(str)['dob'].map(lambda x: x[-4:]).astype(int)
    df['adj_age'] =df['year'] - df['year_of_birth']
    df['adj_age'] = np.where(df['age']>83, df['adj_age'], df['age'])
    df['adj_age'] = np.where(np.logical_and(df['age'] >= 11, df['age'] < 99), df['adj_age'], 'Nan')
    df['adj_age'] = df['adj_age'].astype('float')
    return df.drop(columns=['year_of_birth','age','dob'])


def height_cm(df):
    '''
    Función permite unir 'ht_feet' y 'ht_inch' para entregar la altura en centímetros.
    ------------------------------------------------
    Parámetros:
    - df: DataFrame con los datos de investigaciones y detenciones realizadas en NY.
    ------------------------------------------------
    Return:
    - Devuelve DataFrame con columna 'ht_cm' eliminando las variables innecesarias.
    '''
    feet_cm = 30.48
    meters = df['ht_feet'].astype(str) + '.' + df['ht_inch'].astype(str)
    df['ht_cm'] = (meters.apply(lambda x: float(x) * feet_cm))
    return df.drop(columns=['ht_feet', 'ht_inch'])


def adj_weight(df):
    '''
    Función permite eliminar pesos irreales del dataset y entregar entre el rango de 90
    libras a 690 (valor estimado por el peso de la persona más gorda en NY).
    ------------------------------------------------
    Parámetros:
    - df: DataFrame con los datos de investigaciones y detenciones realizadas en NY.
    ------------------------------------------------
    Return:
    - Devuelve DataFrame con columna 'adj_weight' eliminando las variables innecesarias.
    '''
    df['adj_weight'] = np.where(np.logical_and(df['weight'] >= 90, df['weight'] < 690), df['weight'], 'Nan')
    df['adj_weight'] = df['adj_weight'].astype('float')
    return df.drop(columns='weight')


def train_function(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred = pipe.predict(X_test)

    print(classification_report(y_train, y_pred_train, digits=4))
    print(classification_report(y_test, y_pred, digits=4))

    return pipe
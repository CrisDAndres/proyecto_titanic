# ---------------------LIBRERÍAS----------------------#
import streamlit as st
import pandas as pd
import numpy as np
# visualización
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# mpdelado y predicción
import json
from joblib import load

st.set_option('deprecation.showPyplotGlobalUse', False)

# ---------------------CONFIGURACIÓN DE LA PÁGINA----------------------#
st.set_page_config(
    page_title="TITANIC",
    page_icon="🚢",
    layout="wide", # Esta opción busca el tam de la pantalla, y de izq a dcha va a intentar que el contenido ocupe el mayor espacio posible
    initial_sidebar_state="expanded", # o collapsed
)

logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Titanic_%281997_film%29_logo.svg/1280px-Titanic_%281997_film%29_logo.svg.png" #También se podría poner una ruta local
image = "https://cdn.pixabay.com/photo/2023/10/06/17/14/ship-8298749_1280.png" 


# ---------------------COSAS QUE VAMOS A USAR EN LA APP----------------------#
df_original = pd.read_csv("Data/titanic.csv")
df = pd.read_csv("Data/titanic_preprocessed.csv")

# ---------------------HEADER----------------------#
st.image(image,width=350)
st.title("Análisis del dataset TITANIC")
# st.write('*Fuente imagen: https://pixabay.com/*') 


# ---------------------SIDEBAR (Barra de menú)----------------------#
# Establece el ancho de la barra lateral y alinea el menú
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 250px;
        margin-left: -50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.image(logo,width=120) 
st.sidebar.title("ÍNDICE")

# ---------------------PAGE 1----------------------#
if st.sidebar.button("Información del dataset"):
    st.markdown("""
                ***Este proyecto tiene como objetivo cargar, preprocesar y analizar el conjunto de datos de los pasajeros del Titanic. El set de datos contiene información sobre pasajeros de este famoso viaje, incluyendo detalles como edad, género, clase, puerto de embarque o estado de supervivencia.***
                
                El dataset tiene 891 filas y 12 columnas, con las siguientes características:

                Columnas numéricas:
                1. ``PassengerId``: Identificador numérico único para cada pasajero.
                2. ``Survived``: 0 si no sobrevivió, 1 si sobrevivió.
                3. ``Pclass``: Clase del pasajero (1: Primera clase, 2: Segunda clase, 3: Tercera clase).
                4. ``Age``: Edad del pasajero.
                5. ``SibSp``: Número de hermanos y/o cónyuges a bordo.
                6. ``Parch``: Número de padres y/o hijos a bordo.
                7. ``Fare``: Tarifa pagada por el pasajero.

                Columnas categóricas:
                1. ``Name``: Nombre del pasajero.
                2. ``Sex``: Género del pasajero (Male: Masculino, Female: Femenino).
                3. ``Ticket``: Número del ticket del pasajero.
                4. ``Cabin``: Número de la cabina del pasajero.
                5. ``Embarked``: Puerto de embarque (C: Cherbourg, Q: Queenstown, S: Southampton).
                
                *Fuente*: https://www.kaggle.com/c/titanic/data
                """
                        
                        )
    
    st.markdown("### Visualización del dataframe original:")
    st.dataframe(df_original.head())
    st.markdown("### Búsqueda de valores nulos y duplicados:")
    st.code("""df.isnull().sum()""", language='python')
    st.write(df_original.isnull().sum()) 
    st.markdown("Hay nulos en las columnas ``Age``, ``Cabin`` y ``Embarked``")
    st.code("""sns.heatmap(df.isnull(), cbar=False)""", language='python')
    st.markdown("""
                Hacemos un mapa de calor para visualizar los valores nulos. Se puede observar como están sobre todo en las columnas ``Age`` y ``Cabin``
                """)
    col1, col2 = st.columns(2) # Para que salga más pequeño el heatmap
    with col1:
        # Crear el mapa de calor con sns.heatmap()
        heatmap = sns.heatmap(df_original.isnull(), cbar=False)
        # Mostrar el mapa de calor en Streamlit
        st.pyplot(heatmap.get_figure())
    
    st.code("""df.duplicated().sum()""", language='python')
    st.write(df_original.duplicated().sum())
    st.markdown("""
                Se puede observar como no hay valores duplicados.
                """)       
# ---------------------PAGE 2----------------------#
               
if st.sidebar.button("Preprocesamiento"):
    st.markdown("""
                - Generamos nueva columna ``Title`` que incluye los títulos de cada pasajero.
                - Pasamos la columna ``PassengerId`` a índice, ya que no aporta información
                - Reparación de valores nulos:
                    - Los nulos de la variable ``Age`` se imputan con la mediana de la columna.
                    - Los nulos de la variable ``Cabin`` se rellenan con la categoría *Unknown*.
                    - Los nulos de la variable ``Embarked`` se imputan con la moda.
                - Reparación de valores atípicos *(outliers)* de la columna ``Fare``.
                
                
                
                Para información del código empleado, visita mi GitHub: https://github.com/CrisDAndres/proyecto_titanic
        """)
    # ---------------------TABS (pestañas)----------------------#
    tab1, tab2 = st.tabs(
        ['Valores nulos','Valores atípicos']) 
    with tab1: 
        col1, col2 = st.columns(2)
    with col1:
        st.write('Primero reparamos la columna ``Cabin``:')
        st.code("""df['Cabin'] = df['Cabin'].fillna('Unknown')""",language='python')
        st.write('Para las variables ``Age`` y ``Embarked`` utilizamos la siguiente función:')
        st.code("""
                def reparar_nulos(df, col):
                    if df[col].dtype == 'int64':
                        df[col] = df[col].fillna(df[col].mean())
                    elif df[col].dtype == 'float64': # La edad en este dataset es de tipo float
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0]) 
                                
                    return df 

                for col in df.columns:
                    reparar_nulos(df, col)""", language='python')
    
        st.write('Observamos que ya no hay valores nulos:')
        st.code('df.isnull().sum()/len(df)*100', language='python')
        st.write(df.isnull().sum()/len(df)*100)

    with col2:
        st.write('')
    with tab2:
        st.code("""df.describe()""", language='python')
        st.write(df.describe())
        st.markdown('Se puede observar que la variable ``Fare`` tiene una desviación estándar muy alta. Vamos a visualizar los datos:')
        st.subheader("Distribución de la tarifa pagada por cada pasajero")
        # Crear el gráfico de caja con Plotly
        fig = go.Figure()
        fig.add_trace(go.Box(x=df_original["Fare"], name='Tarifa', showlegend=False, opacity=None, marker=dict(color='#B098E3', line=dict(color='#665784', width=1))))
        # Personalizar el diseño y los títulos
        fig.update_layout(
            xaxis_title="Tarifa"
        )
        # Mostrar la figura en Streamlit
        st.plotly_chart(fig)
        st.markdown("""
                    Hay valores atípicos superiores.
                    Consideramos valor atípico superior lo que esté fuera del intervalo ``Q3 + Rango Intercuartílico (IQR) x 1.5``.
                    Reemplazamos esos valores atípicos por ese límite superior.
                    """)
        st.markdown('Volvemos a visualizar los valores, y vemos que la distribución de la variable ya no tiene valores atípicos:')

        fig = go.Figure()
        fig.add_trace(go.Box(x=df["Fare"], name='Tarifa', showlegend=False, opacity=None, marker=dict(color='#B098E3', line=dict(color='#665784', width=1))))
        fig.update_layout(
            xaxis_title="Tarifa"
        )
        st.plotly_chart(fig)
        

    st.markdown("### Visualización del dataframe preprocesado:")
    st.dataframe(df.head())
    
    
# ---------------------PAGE 3----------------------#

if st.sidebar.button("Análisis exploratorio de los datos"):

    st.markdown("Una vez preprocesado el dataset, procedemos al análisis y visualización de la distribución de los datos:")
    
    st.markdown('### **Distribución de las variables**')
    st.write('En primer lugar, nos interesa saber la distribución de las diferentes variables del dataset:')
    
    
    # ---------------------TABS (pestañas)----------------------#
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ['Género','Clase', 'Tarifa', 'Embarque','Supervivencia']) 
    
    st.write('----------------------------')

    # ---------------------PESTAÑA GÉNERO----------------------#

    with tab1: 
        col1, col2 = st.columns(2)
        with col1:
            st.write('Vemos que los hombres fueron casi el doble que las mujeres que embarcaron en el Titanic:')
            # Creamos el gráfico de pastel para ver las proporciones
            colors = ['#A9DFBF', '#F9E79F']
            fig = px.pie(values=df['Sex'].value_counts(), names=['Hombres','Mujeres'], color_discrete_sequence=colors)
            fig.update_layout(width=500, height=500, showlegend=True, title='Distribución de los pasajeros por género', title_x=0.15, title_font_size=18,
                            legend=dict(
                    orientation='h',  # Orientación horizontal
                    y=0,  # Desplazamiento vertical desde el gráfico (0-1)
                    xanchor='center',  # Ancla en el centro horizontal
                    x=0.5  # Desplazamiento horizontal desde el gráfico (0-1)
                ))
            fig.update_traces(textinfo='percent', textfont_size=16, marker = dict(line = dict(color = 'black', width = 0.5)))
            # Mostrar el gráfico de plotly
            st.plotly_chart(fig)
    
    # ---------------------PESTAÑA CLASE----------------------#
    with tab2: 
        col1, col2 = st.columns(2)
        with col1:
            st.write('Vemos que embarcó un mayor porcentaje de personas en tercera clase:')

            # Primero definimos los colores por clase
            colors = {'1': '#F9E79F', '2': '#ABEBC6', '3': '#16A085'}

            # Definimos el tamaño del gráfico y el estilo de fondo
            plt.figure(figsize=(12, 10))
            sns.set_style("white") 
            # Hacemos gráfico countplot y mostramos el porcentaje de pasajeros por cada clase
            ax = sns.countplot(data=df, y='Pclass', palette=colors, edgecolor='black', linewidth=0.5) 
            ax.set_yticklabels(['1ª clase', '2ª clase', '3ª clase'], fontsize=11) # Cambiamos el nombre de las etiquetas del eje x
            ax.set_xlabel('Porcentaje de pasajeros', labelpad=15, fontsize=15) # Establecer el título del eje y y separarlo del eje
            ax.set_ylabel('')
            ax.set_title('Distribución de los pasajeros por clases', fontsize=17, fontweight='bold')  # Establecer el título del gráfico en negrita

            # Mostrar el gráfico de seaborn
            st.pyplot()
        
    # ---------------------PESTAÑA FARE----------------------#
    with tab3: 
        col1, col2 = st.columns(2)
        with col1:
            st.write('Se observa una mayor densidad en las tarifas bajas que en las altas:')

            # Crear el gráfico de densidad de las tarifas

            plt.figure(figsize=(10, 6))
            sns.set_style("ticks") 
            ax = sns.kdeplot(data=df, y='Fare', fill=True, color='#117A65')
            ax.set_xlabel('Densidad', labelpad = 15, fontsize=15)
            ax.set_ylabel('Tarifa', fontsize=15)
            ax.set_title('Densidad de la tarifa pagada', fontsize=18, fontweight='bold')  # Establecer el título del gráfico en negrita

            # Mostrar el gráfico de seaborn
            st.pyplot()
        
    # ---------------------PESTAÑA EMBARKED----------------------#
    with tab4: 
        col1, col2 = st.columns(2)
        with col1:
            st.write('Se observa que el mayor porcentaje de pasajeros embarcaron desde Southampton:')
            # Agregar dos líneas en blanco para alinear con col2
            st.text("")
            st.text("")

            # Hacemos gráfico countplot y mostramos el número de pasajeros por cada puerto de embarque

            plt.figure(figsize=(10, 6))
            sns.set_style("white") 
            colors = {'S': '#F9E79F', 'C': '#ABEBC6', 'Q': '#16A085'}
            ax = sns.countplot(data=df, y='Embarked', hue='Embarked',stat="percent", palette = colors,edgecolor='black',linewidth=0.5,legend=False)
            ax.set_yticklabels(['Southampton', 'Cherbourg', 'Queenstown']) # Cambiamos el nombre de las etiquetas del eje x
            ax.set_xlabel('Porcentaje de pasajeros', labelpad = 15, fontsize=15) # Establecer el título del eje y y separarlo del eje
            ax.set_ylabel('Puerto de embarque',fontsize=15)
            ax.set_title('Cantidad de pasajeros por cada puerto de embarque', fontsize=18, fontweight='bold')  # Establecer el título del gráfico en negrita

            # Mostrar el gráfico de seaborn
            st.pyplot()
        
        with col2:
            st.write('Además, se puede observar como en Southampton y Queenston la mayoría de pasajeros embarcaron en 3ª clase, mientras que en Cherbour hubo mayoría de 1ª clase:')

            plt.figure(figsize=(10, 6))
            sns.set_style("white") 
            colors = {1: '#F9E79F', 2: '#ABEBC6', 3: '#16A085'}
            ax = sns.countplot(data=df, y='Embarked', hue='Pclass',palette = colors, edgecolor='black',linewidth=0.5)

            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
            ax.set_xlabel('Cantidad de pasajeros', labelpad=15,fontsize=15)
            ax.set_ylabel('')

            plt.legend(title='Clase', labels=['1ª Clase', '2ª Clase', '3ª Clase'],loc=4)

            ax.set_title('Cantidad de pasajeros por cada puerto de embarque', fontsize=18, fontweight='bold')  # Establecer el título del gráfico en negrita

            # Mostrar el gráfico de seaborn
            st.pyplot()
            
    # ---------------------PESTAÑA SURVIVED----------------------#

    with tab5: 
        col1, col2 = st.columns(2)
        with col1:
            st.write('Se observa como hubo un mayor porcentaje de pasajeros que no sobrevivieron al accidente:')
                
            # Crear el gráfico de pastel
            colors = ['#493838', '#73C4A8']
            fig = px.pie(values=df['Survived'].value_counts(), names=['No supervivientes','Supervivientes'], color_discrete_sequence=colors, hole=0.3)
            fig.update_layout(width=600, height=500, showlegend=True, title='Supervivientes del Titanic',  title_x=0.3, title_font_size=18, template = 'plotly_white',legend=dict(
                    orientation='h',  # Orientación horizontal
                    y=0,  # Desplazamiento vertical desde el gráfico (0-1)
                    xanchor='center',  # Ancla en el centro horizontal
                    x=0.5  # Desplazamiento horizontal desde el gráfico (0-1)
                ))
            fig.update_traces(textinfo='percent', textfont_size=16, marker = dict(line = dict(color = 'black', width = 0.5)))
            
            # Mostrar el gráfico de plotly
            st.plotly_chart(fig)

            
    
# ---------------------PAGE 4----------------------#
   
    
if st.sidebar.button("Estudio de la supervivencia"):
    
    st.write('Una vez visualizada la distribución de las variables del dataset, vamos a estudiar en más detalle la variable ``Survived`` y las características de las personas que sobrevivieron y que murieron (rango de edad, género, clase, tarifa pagada y título):")')


    # ---------------------TABS (pestañas)----------------------#
    tab1, tab2, tab3, tab4 = st.tabs(
        ['Supervivencia VS Género','Supervivencia VS Edad', 'Supervivencia VS Clase/Tarifa', 'Supervivencia VS título']) 

    # ---------------------PESTAÑA VS GÉNERO----------------------#

    with tab1: 
        col1, col2 = st.columns(2)
        st.markdown('**Más información en porcentajes:**')
    # Texto con formato HTML y estilos CSS
        st.markdown('De entre los supervivivientes, un <span style="color:#DF925E; font-size:22px;">**68.13**</span> % fueron mujeres, y un <span style="color:#73C4A8; font-size:22px;">**31.87**</span> % hombres', unsafe_allow_html=True)
        st.markdown('De todas las mujeres, un <span style="color:#DF925E; font-size:22px;">**74.2**</span> % sobrevivió.\nDe los hombres, solo lo hizo el <span style="color:#73C4A8; font-size:22px;">**18.89**</span> %.',unsafe_allow_html=True)
        with col1:
            st.markdown('**Densidad de pasajeros supervivientes/no supervivientes según su género:**')
            # Crear el gráfico de densidad de pasajeros por género
            plt.figure(figsize=(12, 7))
            colors = {'male': '#73C6B6', 'female': '#EDBB99'}
            ax = sns.kdeplot(data=df, x='Survived', label= 'Sexo', hue='Sex', fill=True, alpha=0.5, palette = colors)
            # Establecer las posiciones de las marcas en el eje x
            ax.set_xticks([0, 1])
            # Cambiar el nombre de las etiquetas del eje x
            ax.set_xticklabels(['No Supervivientes', 'Supervivientes'], fontsize = 14)
                    # plt.title('Densidad de pasajeros supervivientes y no supervivientes por Género')
            ax.set_xlabel('Género', labelpad = 15, fontsize=18)
            ax.set_ylabel('Densidad', labelpad = 15, fontsize=18)
            # Cambiar los nombres dentro de la leyenda
            handles, labels = ax.get_legend_handles_labels()
            new_labels = ['Mujeres', 'Hombres']  # Nuevos nombres para las etiquetas
            ax.legend(handles=handles, labels=new_labels, title='Género',fontsize=14, title_fontsize=14)
    
            # Mostrar el gráfico de seaborn
            st.pyplot()
        
    # ---------------------PESTAÑA VS EDAD----------------------#

    with tab2: 
        col1, col2 = st.columns(2)
        
        with col1:
            st.write('Graficamos la distribución de la edad entre las personas supervivientes y observamos que de entre las personas que no sobrevivieron, había una mayor densidad de personas con edades de entre 20 y 40 años que entre las personas que sobrevivieron:')            
            
            # Filtrar los datos para supervivientes y no supervivientes
            survivors = df[df['Survived'] == 1]
            non_survivors = df[df['Survived'] == 0]

            # Crear el gráfico de densidad para estudiar la distribución de las edades de los supervivientes y no supervivientes
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=survivors['Age'], label='Supervivientes', fill=True, color = '#73C4A8', alpha=0.5)
            sns.kdeplot(data=non_survivors['Age'], label='No supervivientes', fill=True, color = '#493838', alpha=0.3)

            # plt.title('Distribución de Edades de Supervivientes y No Supervivientes')
            plt.xlabel('Edad', labelpad = 15, fontsize=15)
            plt.ylabel('Densidad', labelpad = 15, fontsize=15)
            plt.legend(fontsize='large')
            # Cambiar el tamaño de la fuente de las etiquetas de la leyenda
            ax.set_title('Densidad de pasajeros supervivientes/no supervivientes según su edad', fontsize=18, fontweight='bold')  # Establecer el título del gráfico en negrita
            # Mostrar el gráfico de seaborn
            st.pyplot()
            
            st.markdown('**Más información en pocentajes:**')
            # Texto con formato HTML y estilos CSS
            st.markdown('- Media de edad de los supervivientes: <span style="font-size:22px;">**28.27**</span> años.', unsafe_allow_html=True)
            st.markdown('- Media de edad no supervivientes: <span style="font-size:22px;">**30.01**</span> años.', unsafe_allow_html=True)


        
    # ---------------------PESTAÑA VS CLASE----------------------#

    with tab3: 
        col1, col2 = st.columns(2)
        
        with col1:
            st.write('Graficamos la distribución de la clase entre las personas supervivientes y observamos que la mayoría de los que no sobrevivieron viajaban en 3ª clase, y la mayoría de los supervivientes en 1ª clase:')            
            
            # Calcular los porcentajes de supervivencia por clase
            total_class = df['Pclass'].value_counts()
            # Calcular el número de pasajeros por título y supervivencia
            class_survived = df.groupby(['Pclass', 'Survived']).size() # size es una funcion de agregacion que cuenta el número de filas (pasajeros) en cada combinacion formada por la operación groupby
            #Calculo el porcentaje
            perc_class_survived = class_survived.div(total_class, level='Pclass') * 100
            perc_class_survived = perc_class_survived.reset_index() # Con .reset_index() convierto los índices en columnas para luego poder manipularlas

            plt.figure(figsize=(15, 8))
            sns.set_style("white")
            colors = {1: '#F9E79F', 2: '#ABEBC6', 3: '#16A085'}

            ax = sns.barplot(data=perc_class_survived, x='Survived', y=0, hue='Pclass',palette=colors, edgecolor='black', linewidth=0.5)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No Supervivientes', 'Supervivientes'], fontsize = 20)

            ax.set_ylabel('Porcentaje de pasajeros', labelpad = 15, fontsize=20) 
            ax.set_xlabel('') 

            # Agregar cuadrícula en el eje horizontal
            ax.grid(axis='y')

            plt.legend(title='Clase', fontsize = 18, title_fontsize=18)

            # Mostrar el gráfico de seaborn
            st.pyplot()
        
            st.write('Esta relación entre las clases y la supervivencia puede observarse también al analizar la relación entre los supervivientes y las **tarifas** de los billetes:')
            # Texto con formato HTML y estilos CSS
            st.markdown('Media de las tarifas pagadas por los supervivientes: <span style="font-size:22px;">**18.91**</span> £.', unsafe_allow_html=True)
            st.markdown('Media de las tarifas pagadas por los no supervivientes: <span style="font-size:22px;">**32.28**</span> £.', unsafe_allow_html=True)
        
    # ---------------------PESTAÑA VS TÍTULO----------------------#

    with tab4: 
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('El título ***Miss.*** además de a mujeres no casadas incluye también a las niñas, y ***Master.*** a los niños. Vemos que entre los supervivientes hay mayoría de niños, niñas y mujeres solteras, y entre los no supervivientes, la mayoría son hombres sin título nobiliario:')            
            
            total_Title = df['Title'].value_counts()
            # Calcular el número de pasajeros por título y supervivencia
            Title_Survived = df.groupby(['Title', 'Survived']).size() 
            #Calculamos el porcentaje 
            perc_Title_Survived = Title_Survived.div(total_Title, level='Title') * 100 
            perc_Title_Survived = perc_Title_Survived.reset_index()

            plt.figure(figsize=(10, 6))
            ax2 = sns.barplot(data=perc_Title_Survived, x='Survived', y=0, hue='Title',palette='Set2',edgecolor='grey',linewidth=0.5) 

            ax2.set_xticks([0, 1])

            # Agregar cuadrícula en el eje horizontal
            ax2.grid(axis='y')

            ax2.set_xticklabels(['No supervivientes', 'Supervivientes'], fontsize = 15)
            ax2.set_ylabel('Porcentaje de pasajeros', labelpad = 15, fontsize=15) 
            ax2.set_xlabel('') 
            plt.legend(title='Título', fontsize = 13, title_fontsize=13)
            # Mostrar el gráfico de seaborn
            st.pyplot()
                
# ---------------------PAGE 5----------------------#
      
if st.sidebar.button("Conclusiones"):
    st.markdown('Las conclusiones a las que podemos llegar con el análisis de estos datos son:')
    st.markdown("""
        - Embarcaron más hombres (<span style="font-size:22px;">**64.8 %**</span>) que mujeres (<span style="font-size:22px;">**35.2 %**</span>).

        - El porcentaje de personas que no sobrevivieron fue mayor (<span style="font-size:22px;">**61.6%**</span>). De todas las mujeres, un <span style="font-size:22px;">**74.2%**</span> sobrevivió.

        - De los hombres, fallecieron el <span style="font-size:22px;">**81.1%**</span>, con una media de edad de <span style="font-size:22px;">**30.77**</span> años.

        - De los supervivientes, un <span style="font-size:22px;">**68.13%**</span> fueron mujeres, y un <span style="font-size:22px;">**31.87%**</span> hombres.

        - La mayoría de los que **no sobrevivieron** pertenecían a la <span style="font-size:22px;">**3ª clase**</span>, y la mayoría de **supervivientes** pertenecían a la <span style="font-size:22px;">**1ª clase**</span>.

        - Embarcaron más pasajeros desde <span style="font-size:22px;">**Southampton**</span>. En todos los puertos embarcaron mayoría de pasajeros en 3ª clase excepto en Cherbourg que fueron mayoría de 1ª clase.
""", unsafe_allow_html=True)

# ---------------------PAGE 6----------------------#


if st.sidebar.button("Predicción de la supervivencia"):
    st.markdown("### Predicción de la supervivencia de los pasajeros")
    
    ## -- Descarga de archivos
    scaler = load('outputs/scaler_classif.pkl')
    model = load('models/survived_RFC.pkl') 
    with open('outputs/encoder_sex.json', 'r') as f:
        encoder_sex = json.load(f)
    with open('outputs/decoder_sex.json', 'r') as f:
        decoder_sex = json.load(f)

 # --------------------------------------------------------------------------------------
    
    # Definición de los valores para la interfaz
    fare_min = 0
    fare_max = 100
    sex_options = ['male','female']
    
    with st.form("prediction_form"): 
        pclass = st.number_input('Clase en la que viaja el pasajero:', min_value=1, max_value=3, value=1)
        sex = st.selectbox('Género del pasajero:', sex_options)
        fare = st.slider('Tarifa del billete (£):', fare_min, fare_max)
        age = st.number_input('Edad del pasajero:', min_value=0, value=1)
        submit_button = st.form_submit_button(label='¿Sobrevivirá?')

    if submit_button:
        input_data = pd.DataFrame([[pclass, sex, fare, age]],
                                columns=['Pclass', 'Sex', 'Fare', 'Age']) 

        # 1- Primero codifico a números lo que ingresa el usuario utilizando el json de mapeo
        input_data['Sex'] = input_data['Sex'].replace(encoder_sex)

        # 2 - Después normalizo los datos de entrada
        input_data_scaled = scaler.transform(input_data)

        # 3 - Realiza la predicción con el modelo
        prediction = model.predict(input_data_scaled)

        # 4 - Diccionario de mapeo para la predicción
        prediction_mapping = {0: "No sobrevive", 1: "Sobrevive"}
            
        # 4 - Por último decodifico utilizando los diccionarios de mapeo inverso
        input_data['Sex'] = input_data['Sex'].replace(decoder_sex)
        
        
        # Asegurémonos de acceder al nombre correcto de la columna de predicciones
        predicted_survived= prediction[-1]  # Generalmente, la predicción está en la última columna
        # Mapear la predicción a los valores deseados
        prediction_text = prediction_mapping[predicted_survived]
        st.write(f"### La predicción de la supervivencia de ese pasajero es que: {prediction_text}")



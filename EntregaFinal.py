from tkinter import Tk, Label, OptionMenu, StringVar, Button
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
###################################################################################################################
#El objetivo del proyecto es desarrollar un modelo de regresión lineal para predecir el número de
# transferencias de vehículos por provincias de la Argentina. El modelo se basa en datos 
# históricos de transferencias desde el año 2000 al 2023.
####################################################################################################################
#Variables disponibles:
    
# tipo_vehiculo columna 0
# anio_transferencia columna 1
# mes_transferencia columna 2
# provincia_transferencia columna 3
# letra_provincia_transferencia columna 4
# cantidad_transferencias columna 5
# provincia_id columna 6     

class ProyectoTransferencias:
     # Cargamos el archivo CSV en un DataFrame. 
     # Este dataset contiene informacion sobre las transferencias historicas de las provincias desde el año 2000
    def __init__(self):
        self.file_path = 'transferencias.csv'
        self.df = pd.read_csv(self.file_path, sep=',', encoding='UTF-8', low_memory=False)
        #Dropeamos las columnas que no vamos a usar
        #Dejamos las columnas año de transferencia, mes, provincia y cantidad de transferencias 
        #para poder calcular la prediccion en base a estas cuatro columnas      
        self.df.drop([0, 4, 6], inplace=True)
        self.provincia_elegida = None
        self.anio_elegido = None


    def cargar_datos(self):
        # Muestra los primeros registros del DataFrame
        print(f"\n Se muestra por pantalla el dataset: \n\n {self.df} \n\n")
        # Detectamos si hay valores nulos o NaN en el DataFrame
        print("\n\n Detecto si hay valores nulos/nans \n\n")
        print(self.df.isnull().any())
        print("\n Imprimo los primeros 10 registros: \n ")
        print(self.df.head(10))   

    def mostrar_interfaz(self):
        root = Tk()
        root.title("Proyecto Transferencias")

        # Label para seleccionar una provincia
        label = Label(root, text="Selecciona una provincia:")
        label.pack()

        # Opciones de provincias
        provincias_disponibles = self.df['provincia_transferencia'].unique()
        provincia_seleccionada = StringVar(root)
        provincia_seleccionada.set(provincias_disponibles[0])  # Valor por defecto

        # Menú desplegable
        opciones_provincias = OptionMenu(root, provincia_seleccionada, *provincias_disponibles)
        opciones_provincias.pack()

        # Botón
        button = Button(root, text="Seleccionar Provincia", command=lambda: self.seleccionar_provincia(root, provincia_seleccionada.get()))
        button.pack()

        root.mainloop()

    def seleccionar_provincia(self, root, provincia):
        self.provincia_elegida = provincia
        root.destroy()  # Cerraramos la ventana después de seleccionar la provincia

        # Filtramos por la provincia elegida
        df_eleccion = self.df[self.df['provincia_transferencia'] == self.provincia_elegida]

        # Agrupamos los datos filtrados por año y calcula el total de transferencias por año
        total_transferencias_por_anio = df_eleccion.groupby('anio_transferencia')['cantidad_transferencias'].sum().reset_index()

        # Configuramos el estilo de seaborn
        sns.set(style="whitegrid", palette="pastel")

        # Creamos el gráfico de barras horizontal
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x='anio_transferencia', y='cantidad_transferencias', data=total_transferencias_por_anio, color='skyblue')
        barplot.set(xlabel='Año de Transferencia', ylabel='Total de Transferencias', title=f'Total de Transferencias por Año - {self.provincia_elegida}')

        # Muestramos la cantidad de transferencias en las barras por provincia elegida en su historico
        for index, value in enumerate(total_transferencias_por_anio['cantidad_transferencias']):
            barplot.text(index, value, str(value), ha='center', va='bottom', fontsize=10)

        # Añade estilo adicional
        sns.despine(left=True, bottom=True)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.show()

        # Preguntamos por el año para el segundo gráfico en una nueva ventana
        self.mostrar_menu_anio()

    def mostrar_menu_anio(self):
        root = Tk()
        root.title("Proyecto Transferencias - Selección de Año")

        # Label para seleccionar un año a predecir
        label = Label(root, text="Selecciona un año para predecir:")
        label.pack()

        # Opciones de años
        anios_disponibles = self.df['anio_transferencia'].unique()
        anio_seleccionado = StringVar(root)
        anio_seleccionado.set(anios_disponibles[0])  # Valor por defecto

        # Menú desplegable
        opciones_anios = OptionMenu(root, anio_seleccionado, *anios_disponibles)
        opciones_anios.pack()

        # Botón
        button = Button(root, text="Aceptar", command=lambda: self.ingresar_anio(root, anio_seleccionado.get()))
        button.pack()

        root.mainloop()

    def ingresar_anio(self, root, anio):
        self.anio_elegido = int(anio)
        root.destroy()  # Cerrar la ventana después de seleccionar el año
        self.generar_predicciones()

    def generar_predicciones(self):
        # Filtra el DataFrame por el año especificado por el usuario
        df_filtered = self.df[(self.df['provincia_transferencia'] == self.provincia_elegida) & (self.df['anio_transferencia'] == self.anio_elegido)]

        # Definimos las variables independientes (X) y dependientes (y)
        # vamos a tratar de predecir la cantidad de transferencias
        X = df_filtered[['mes_transferencia']]
        y = df_filtered['cantidad_transferencias']

        # Divide los datos en conjuntos de entrenamiento y prueba (80-20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Creamos el modelo de regresión lineal
        model = LinearRegression()

        # Entrenamos el modelo
        model.fit(X_train, y_train)

        # Realizamos la predicción para el conjunto completo
        df_filtered = df_filtered.copy()
        df_filtered['predicciones'] = model.predict(df_filtered[['mes_transferencia']])

        # Visualizamos los resultados con leyenda
        plt.scatter(df_filtered['mes_transferencia'], df_filtered['cantidad_transferencias'], color='black', label='Valores reales')
        plt.plot(df_filtered['mes_transferencia'], df_filtered['predicciones'], color='blue', linewidth=3, label='Predicciones')
        plt.title(f'Regresión Lineal - Transferencias en {self.provincia_elegida} para el año {self.anio_elegido}')
        plt.xlabel('Mes')
        plt.ylabel('Cantidad de Transferencias')
        plt.legend()

        # Mostramos el total de transferencias reales y predichas de la provincia escogida
        total_reales = df_filtered['cantidad_transferencias'].sum()
        total_predicciones = df_filtered['predicciones'].sum()
        plt.text(0.5, 0.9, f'Total Reales: {total_reales}\nTotal Predicciones: {total_predicciones}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

        plt.show()

        print(f'Total de transferencias reales para {self.anio_elegido} en {self.provincia_elegida}: {total_reales}')
        print(f'Total de transferencias predichas para {self.anio_elegido} en {self.provincia_elegida}: {total_predicciones}')

        # Realizamos la predicción para el conjunto de prueba
        y_pred = model.predict(X_test)

        # Mostramos métricas de evaluación del modelo
        print('Error Absoluto Medio:', metrics.mean_absolute_error(y_test, y_pred))
        print('Error Cuadrático Medio:', metrics.mean_squared_error(y_test, y_pred))
        print('Raíz del Error Cuadrático Medio:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        self.grupo()

    def grupo(self):
        # Miembros del grupo
        print("\n****************************************************")
        print("*************** Gracias por usar el sistema *********")
        print("****************************************************")
        print("******************* Miembros del equipo: **************")
        print("****************** PRIETO SEBASTIAN ISIDRO ***********")
        print("******************** REY BRIENZA AGUSTINA ************")
        print("******************** FERRETTI EMILIANO ***************")
        print("************************* UNSAM **********************")
        print("******************* 2do Cuatrimestre 2023 ************")

    def ejecutar_proyecto(self):
        self.cargar_datos()
        self.mostrar_interfaz()
# Creamos una instancia de ProyectoTransferencias y ejecutamos el proyecto
proyecto_transferencias = ProyectoTransferencias()
proyecto_transferencias.ejecutar_proyecto()
###############################################Conclusión:#########################################################
# En conclusión, el proyecto desarrollado es un primer paso para crear una herramienta que pueda utilizarse 
# para predecir el número de transferencias de vehículos por provincias de la Argentina. Los resultados del
# proyecto son alentadores, pero es importante señalar que el modelo se basa en datos históricos. 
# Es posible que el modelo no sea tan preciso para predecir el número de transferencias en el 
# futuro, si se producen cambios en las condiciones económicas o en las políticas gubernamentales.





































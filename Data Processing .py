# Databricks notebook source
# MAGIC %md
# MAGIC # Procesammiento de datos usando PySpark
# MAGIC Este modulo vamos a estudiar los diferentes pasos para preprocesar y manejar datos en PySpark. Las técnicas de preprocesamiento ciertamente pueden variar de un caso a otro, y se pueden usar muchos métodos diferentes para masajear los datos en la forma deseada. La idea de este modulo es exponer algunas de las técnicas más comunes para manejar big data en Spark. En este modulo, vamos a repasar los diferentes pasos involucrados en el preprocesamiento de datos, como el manejo de valores faltantes, la fusión de conjuntos de datos, la aplicación de funciones, las agregaciones y la clasificación. Una parte importante del preprocesamiento de datos es la transformación de columnas numéricas en categóricas y viceversa, que veremos en los próximos modulos y se basa en el **Machine learning**. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Creación de un objeto SparkSession
# MAGIC El primer paso es crear un objeto SparkSession para usar Spark. También importamos todas las funciones y tipos de datos requeridos desde spark.sql:

# COMMAND ----------

from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('data_processing').getOrCreate()

# COMMAND ----------

# crear un dataframe de cliente declarando el esquema y pasando valores
import pyspark.sql.functions as F
from pyspark.sql.types import *


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Ahora, en lugar de leer directamente un archivo para crear un marco de datos, repasamos el proceso de creación de un marco de datos, pasando valores clave. La forma en que creamos un marco de datos en Spark es declarando su esquema y pasando los valores de las columnas.

# COMMAND ----------




# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Creación de marcos de datos
# MAGIC En el siguiente ejemplo, estamos creando un nuevo marco de datos con cinco columnas de ciertos tipos de datos (cadena y entero). Como puede ver, cuando llamamos a mostrar en el nuevo marco de datos, se crea con tres filas y cinco columnas que contienen los valores que pasamos.

# COMMAND ----------

schema=StructType().add("user_id","string").add("country","string").add("browser", "string").add("OS",'string').add("age", "integer")

# COMMAND ----------

#pass the values
df=spark.createDataFrame([("A203",'Nicaragua',"Chrome","WIN",33),("A201",'Panama',"Safari","MacOS",35),("A205",'Brazil',"Mozilla","Linux",25)],schema=schema)
df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Valores nulos
# MAGIC Es muy común tener valores nulos como parte de los datos generales. Por lo tanto, se vuelve fundamental agregar pipeline al procesamiento de datos para manejar los valores nulos. En Spark, podemos lidiar con valores nulos reemplazándolos con algún valor específico o quitando las filas/columnas que contienen valores nulos.
# MAGIC 
# MAGIC Primero, creamos un nuevo marco de datos (df_na) que contiene valores nulos en dos de sus columnas (el esquema es el mismo que en el marco de datos anterior). Por el primer enfoque para tratar con valores nulos, llenamos todos los valores nulos en el marco de datos actual con un valor de 0, lo que ofrece una solución rápida. Usamos la función `fillna` para reemplazar todos los valores nulos en el marco de datos con 0. Por el segundo enfoque, reemplazamos los valores nulos en columnas específicas (país, navegador) respectivamente.

# COMMAND ----------

#create a new dataframe with nukll values 
df_na=spark.createDataFrame([("A203",None,"Chrome","WIN",33),("A201",'Panama',None,"MacOS",35),("A205",'Brazil',"Mozilla","Linux",25)],schema=schema)


# COMMAND ----------

df_na.show()

# COMMAND ----------

# Usamos la función fillna para reemplazar los datos nulos con 0
df_na.fillna('0').show()

# COMMAND ----------

#fill null values with specific value
df_na.fillna( { 'country':'Mexico', 'browser':'Explorador' } ).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Para eliminar las filas con valores nulos, simplemente podemos usar la funcionalidad na.drop en PySpark. Mientras que si es necesario hacer esto para columnas específicas, también podemos pasar el conjunto de nombres de columna, como se muestra en el siguiente ejemplo:

# COMMAND ----------

#Return new df omitting rows with null values
df_na.na.drop().show()

# COMMAND ----------

df_na.na.drop(subset='country').show()

# COMMAND ----------

# MAGIC %md
# MAGIC Otro paso muy común en el procesamiento de datos es reemplazar algunos puntos de datos con valores particulares. Podemos usar la función replace para esto, como se muestra en el siguiente ejemplo. Para soltar la columna de un marco de datos, podemos usar la función de soltar de PySpark.

# COMMAND ----------

df_na.replace("Chrome","Google Chrome").show()

# COMMAND ----------

#deleting column 
df.drop('user_id').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cargando dataset
# MAGIC Ahora que hemos visto cómo crear un marco de datos pasando un valor y cómo tratar los valores faltantes, podemos crear un marco de datos de Spark leyendo un archivo (.csv, parquet, etc.). El conjunto de datos contiene un total de siete columnas y 2000 filas. La función de resumen nos permite ver las medidas estadísticas del conjunto de datos, como el mínimo, el máximo y la media de los datos numéricos presentes en el marco de datos.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/customer_data-1.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# COMMAND ----------

df=spark.read.csv(file_location,header=True,inferSchema=True)

# COMMAND ----------

df.count()

# COMMAND ----------

len(df.columns)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.show(3)

# COMMAND ----------

df.columns

# COMMAND ----------

df.summary().show()

# COMMAND ----------

# MAGIC %md
# MAGIC La mayoría de las veces, no usaremos todas las columnas presentes en el marco de datos, ya que algunas pueden ser redundantes y tener muy poco valor en términos de proporcionar información útil. Por lo tanto, subdividir el marco de datos se vuelve fundamental para tener los datos adecuados para el análisis. Cubriré esto en la siguiente sección.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subset 
# MAGIC Se puede crear un subconjunto de un marco de datos, en función de múltiples condiciones en las que seleccionamos algunas filas, columnas o datos con ciertos filtros establecidos. En los siguientes ejemplos, verá cómo podemos crear un subconjunto del marco de datos original, según ciertas condiciones, para demostrar el proceso de filtrado de registros.
# MAGIC * Select
# MAGIC * Filter
# MAGIC * Where

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Select
# MAGIC En este ejemplo, tomamos una de las columnas del `DataFrame`, 'Avg_Salary', y creamos un subconjunto del dataframe original, usando `select()``. Podemos pasar cualquier número de columnas que deben estar presentes en el subconjunto. Luego aplicamos un filtro en el dataframe para extraer los registros, en función de un cierto umbral (Avg_Salary> 1000000). Una vez filtrados, podemos tomar el recuento total de registros presentes en el subconjunto o tomarlo para su posterior procesamiento.

# COMMAND ----------

df.select(['Customer_subtype','Avg_Salary']).show()

# COMMAND ----------

df.filter(df['Avg_Salary'] > 1000000).count()

# COMMAND ----------

df.filter(df['Avg_Salary'] > 1000000).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter
# MAGIC También podemos aplicar más de un filtro en el marco de datos, al incluir más condiciones, como se muestra a continuación. Esto se puede hacer de dos maneras: primero, aplicando filtros consecutivos, luego usando operandos (&, o) con una instrucción where.

# COMMAND ----------

df.filter(df['Avg_Salary'] > 500000).filter(df['Number_of_houses'] > 2).show()

# COMMAND ----------

df.where((df['Avg_Salary'] > 500000) & (df['Number_of_houses'] > 2)).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregations 
# MAGIC Cualquier tipo de agregación se puede dividir simplemente en tres etapas, en el siguiente orden:
# MAGIC 
# MAGIC * Split
# MAGIC * Apply
# MAGIC * Combine
# MAGIC 
# MAGIC El primer paso es dividir los datos, en función de una columna o grupo de columnas, y luego realizar la operación en esos pequeños grupos individuales (recuento, máximo, promedio, etc.). Una vez que se obtienen los resultados para cada conjunto de grupos, el último paso es combinar todos estos resultados. En el siguiente ejemplo, agregamos los datos, según el 'Subtipo de cliente', y simplemente contamos la cantidad de registros en cada categoría. Usamos la función groupBy en PySpark. El resultado de esto no está en ningún orden en particular, ya que no hemos aplicado ninguna clasificación a los resultados. Por tanto, también veremos cómo podemos aplicar cualquier tipo de clasificación a los resultados finales. Debido a que tenemos siete columnas en el marco de datos, todas son columnas categóricas excepto una (Avg_Salary), podemos iterar sobre cada columna y aplicar la agregación como en el siguiente ejemplo:

# COMMAND ----------

df.groupBy('Customer_subtype').count().show()

# COMMAND ----------

for col in df.columns:
    if col !='Avg_Salary':
        print(f" *** Aggregation for  {col} ***")
        df.groupBy(col).count().orderBy('count',ascending=False).show(truncate=False)

    

# COMMAND ----------

# MAGIC %md 
# MAGIC Como se mencionó, podemos tener diferentes tipos de operaciones en grupos de registros, como
# MAGIC * Mean
# MAGIC * Max
# MAGIC * Min
# MAGIC * Sum
# MAGIC 
# MAGIC Los siguientes ejemplos cubren algunos de estos, basados en diferentes agrupaciones. F se refiere a la función Spark sql aquí.

# COMMAND ----------

df.groupBy('Customer_main_type').agg(F.min('Avg_Salary')).show()

# COMMAND ----------

df.groupBy('Customer_main_type').agg(F.max('Avg_Salary')).show()

# COMMAND ----------

df.groupBy('Customer_main_type').agg(F.sum('Avg_Salary')).show()

# COMMAND ----------

df.groupBy('Customer_main_type').agg(F.mean('Avg_Salary')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sorting 
# MAGIC A veces, simplemente existe la necesidad de ordenar los datos con agregación o sin ningún tipo de agregación. Ahí es donde podemos hacer uso de la funcionalidad `sort` y `orderBy` de PySpark, para reorganizar los datos en un orden particular, como se muestra en los siguientes ejemplos:

# COMMAND ----------

df.sort("Avg_salary", ascending=False).show()

# COMMAND ----------

df.groupBy('Customer_subtype').agg(F.avg('Avg_Salary').alias('mean_salary')).orderBy('mean_salary',ascending=False).show(50,False)

# COMMAND ----------

df.groupBy('Customer_subtype').agg(F.max('Avg_Salary').alias('max_salary')).orderBy('max_salary',ascending=False).show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Collect 
# MAGIC En algunos casos, también debemos recopilar la lista de valores para grupos particulares o para categorías individuales. Por ejemplo, supongamos que un cliente va a una tienda en línea y accede a diferentes páginas del sitio web de la tienda. Si tenemos que recopilar todas las actividades del cliente en una lista, podemos usar la funcionalidad de recopilación en PySpark. Podemos recopilar valores de dos maneras diferentes:
# MAGIC * Collect List
# MAGIC * Collect Set

# COMMAND ----------

# Collect _set 
df.groupby("Customer_subtype").agg(F.collect_set("Number_of_houses")).show()

# COMMAND ----------

#collect list 
df.groupby("Customer_subtype").agg(F.collect_list("Number_of_houses")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC La necesidad de crear una nueva columna con un valor constante puede ser muy común. Por lo tanto, podemos hacer eso en PySpark, usando la función 'encendido'. En el siguiente ejemplo, creamos una nueva columna con un valor constante:

# COMMAND ----------

# creando una nueva columna con valor constante

df=df.withColumn('constant',F.lit('finance'))

# COMMAND ----------

df.select('Customer_subtype','constant').show()

# COMMAND ----------

# MAGIC %md
# MAGIC Debido a que estamos tratando con marcos de datos, es un requisito común aplicar ciertas funciones personalizadas en columnas específicas y obtener el resultado. Por lo tanto, hacemos uso de UDF para aplicar funciones de Python en una o más columnas.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### User-Defined Functions (UDFs)
# MAGIC En este ejemplo, intentamos nombrar las categorías de edad y crear una función Python estándar (categoría_edad) para las mismas. Para aplicar esto en el dataframe de Spark, creamos un objeto UDF usando esta función de Python. El único requisito es mencionar el tipo de retorno de la función. En este caso, es simplemente un valor de cadena.

# COMMAND ----------

from pyspark.sql.functions import udf
df.groupby("Avg_age").count().show()

# COMMAND ----------

# crear una función para asignar categorías
def age_category(age):
    if age  == '20-30 years':
        return 'Young'
    elif age== '30-40 years':
        return 'Mid Aged' 
    elif ((age== '40-50 years') or (age== '50-60 years')) :
        return 'Old'
    else:
        return 'Very Old'



# COMMAND ----------

# creamos age categorica udf 
age_udf=udf(age_category,StringType())
#create the bucket column by applying udf
df=df.withColumn('age_category',age_udf(df['Avg_age']))

# COMMAND ----------

df.select('Avg_age','age_category').show()

# COMMAND ----------

df.groupby("age_category").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Los UDF de Pandas son otro avance reciente, así que repasémoslos ahora.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas UDF
# MAGIC Las UDF de Pandas son mucho más rápidas y eficientes, en términos de procesamiento y tiempo de ejecución, en comparación con las UDF estándar de Python. La principal diferencia entre una UDF de Python normal y una UDF de Pandas es que una UDF de Python se ejecuta fila por fila y, por lo tanto, realmente no ofrece la ventaja de un marco distribuido. Puede llevar más tiempo, en comparación con Pandas UDF, que se ejecuta bloque por bloque y ofrece resultados más rápidos. Hay tres tipos diferentes de UDF de Pandas: escalar, mapa agrupado y agregado agrupado. La única diferencia en el uso de una UDF de Pandas en comparación con una UDF tradicional radica en la declaración. En el siguiente ejemplo, intentamos escalar los valores de Avg_Salary aplicando la escala. Primero tomamos los valores mínimo y máximo de Avg_Salary, restamos de cada valor el salario mínimo de cada valor y luego lo dividimos por la diferencia entre el máximo y el mínimo.
# MAGIC 
# MAGIC $$\frac{X-x_{min}}{X_{max}-X{min}}$$

# COMMAND ----------

df.select('Avg_Salary').summary().show()

# COMMAND ----------

min_sal=1361
max_sal=48919896

# COMMAND ----------

### Pandas udf 
from pyspark.sql.functions import pandas_udf, PandasUDFType

def scaled_salary(salary):
    scaled_sal=(salary-min_sal)/(max_sal-min_sal)
    return scaled_sal

# COMMAND ----------

scaling_udf = pandas_udf(scaled_salary, DoubleType())
df.withColumn("scaled_salary", scaling_udf(df['Avg_Salary'])).show(10,False)



# COMMAND ----------

# MAGIC %md
# MAGIC Así es como podemos usar tanto las UDF convencionales como las de Pandas para aplicar diferentes condiciones en el marco de datos, según sea necesario.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Joins
# MAGIC Combinar diferentes conjuntos de datos es un requisito muy genérico presente en la mayoría de las canalizaciones de procesamiento de datos en el mundo de los grandes datos. PySpark ofrece una forma muy conveniente de fusionar y pivotar los valores de su marco de datos, según sea necesario. En el siguiente ejemplo, creamos un marco de datos fabricado con algunos valores de código de región ficticios para todos los tipos de clientes. La idea es combinar este marco de datos con el marco de datos original, para tener estos códigos de región como parte del marco de datos original, como una columna.

# COMMAND ----------

df.groupby("Customer_main_type").count().show(50,False)

# COMMAND ----------

region_data = spark.createDataFrame([('Family with grown ups','PN'),
                                    ('Driven Growers','GJ'),
                                    ('Conservative families','DD'),
                                    ('Cruising Seniors','DL'),
                                    ('Average Family ','MN'),
                                    ('Living well','KA'),
                                    ('Successful hedonists','JH'),
                                    ('Retired and Religious','AX'),
                                   ('Career Loners','HY'),('Farmers','JH')],schema=StructType().add("Customer_main_type","string").add("Region Code","string"))

# COMMAND ----------

region_data.show()

# COMMAND ----------


new_df=df.join(region_data,on='Customer_main_type')



# COMMAND ----------

new_df.groupby("Region Code").count().show(50,False)

# COMMAND ----------

# MAGIC %md
# MAGIC Tomamos el recuento regional después de unir el marco de datos original (df) con el marco de datos region_data recién creado en la columna Customer_main_type.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pivoting 
# MAGIC Podemos usar la función dinámica en PySpark para simplemente crear una vista dinámica del marco de datos para columnas específicas, como se muestra en el siguiente ejemplo. Aquí, estamos agrupando datos, según el tipo de cliente. Las columnas representan diferentes grupos de edad. Los valores dentro de la tabla dinámica son la suma del salario promedio de cada una de estas categorías de tipo de cliente para un grupo de edad en particular. También nos aseguramos de que no haya valores nulos o vacíos, completando todos los valores nulos con 0. En el siguiente ejemplo, creamos una tabla dinámica más, usando la columna de etiqueta y tomando la suma de Salario promedio como los valores dentro de ella.

# COMMAND ----------

df.groupBy('Customer_main_type').pivot('Avg_age').sum('Avg_salary').fillna(0).show()

# COMMAND ----------

df.groupBy('Customer_main_type').pivot('label').sum('Avg_salary').fillna(0).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Dividimos los datos, según la columna Customer_main_type, y tomamos la suma acumulada del `Avg_Salary` de cada uno de los valores de la etiqueta (0,1), usando la función pivote.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Window OperationsWindow Functions or Windowed Aggregates
# MAGIC Esta funcionalidad en PySpark le permite realizar ciertas operaciones en grupos de registros conocidos como "dentro de la ventana". Calcula los resultados para cada fila dentro de la ventana. Un ejemplo clásico del uso de la ventana son las diversas agregaciones para un usuario durante diferentes sesiones. Un visitante puede tener múltiples sesiones en un sitio web en particular y, por lo tanto, la ventana se puede usar para contar las actividades totales del usuario durante cada sesión. PySpark admite tres tipos de funciones de ventana:
# MAGIC 
# MAGIC * Aggregations
# MAGIC * Ranking
# MAGIC * Analytics
# MAGIC 
# MAGIC En el siguiente ejemplo, importamos la función de ventana, además de otras, como número_fila. El siguiente paso es definir la ventana. A veces puede ser simplemente una columna ordenada o, a veces, puede basarse en categorías particulares dentro de una columna. Veremos ejemplos de cada uno de ellos. En el primer ejemplo, definimos la ventana, que solo se basa en la columna Ordenada Salario promedio, y clasificamos estos salarios. Creamos una nueva columna "rango" y asignamos rangos a cada uno de los valores de Salario promedio.

# COMMAND ----------

## Ranking 

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import udf,rank, col,row_number

# COMMAND ----------

#create a window function to order the relevant column( Avg Salary)
win = Window.orderBy(df['Avg_Salary'].desc())

# COMMAND ----------

#create a additonal column with row numbers as rank
df=df.withColumn('rank', row_number().over(win).alias('rank'))

# COMMAND ----------

df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Un requisito común es encontrar los tres valores principales de una categoría. En este caso, la ventana se puede utilizar para obtener los resultados. En el siguiente ejemplo, definimos la ventana y la partición por la columna del subtipo Cliente. Básicamente, lo que hace es ordenar el salario promedio para cada categoría de subtipo de cliente, por lo que ahora podemos usar el filtro para obtener los tres valores de salario principales para cada grupo.

# COMMAND ----------

# Ranking groupwise 
#create a window function to order the relevant column( Avg Salary)
win_1 = Window.partitionBy("Customer_subtype").orderBy(df['Avg_Salary'].desc())

# COMMAND ----------

#create a additonal column with row numbers as rank
df=df.withColumn('rank', row_number().over(win_1).alias('rank'))

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora que tenemos un nuevo rango de columna que consiste en el rango o cada categoría de Customer_subtype, podemos filtrar fácilmente los tres primeros rangos para cada categoría.

# COMMAND ----------

df.groupBy('rank').count().orderBy('rank').show()

# COMMAND ----------

# filter top 3 customers from every group
df.filter(col('rank') < 4).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusión
# MAGIC En este modulo, analizamos diferentes técnicas para leer, limpiar y preprocesar datos en PySpark. Vismo los métodos para unir un dataframe y crear una tabla dinámica a partir de él. Las secciones finales del modulo cubrieron las UDF y las operaciones basadas en ventanas en PySpark. Los próximos modulos se centrarán en el manejo de transmisión de datos en PySpark y el **Machine learning** mediante MLlib.

# COMMAND ----------

 

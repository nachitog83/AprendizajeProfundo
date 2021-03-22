# Aprendizaje Profundo

Repositorio oficial de la materia optativa "Aprendizaje Profundo" (Deep Learning) de la Diplomatura en Ciencias de Datos de la UNC.

Para comenzar a instalar y configurar el entorno de trabajo por favor seguir las instrucciones detalladas en el [Notebook 0](./0_set_up.ipynb).

# DiploDatos 2020 -  Aprendizaje Profundo

## Práctico

### Integrantes:
- Ana Rusconi
- Mariano Lucero
- Nazareno Medrano
- Ignacio Grosso

### Detalles de los experimentos:

En este trabajo se ha tratado un problema de clasificación multiclase. 
El DataSet trabajado corresponde al Meli Challenge 2019, en el que se evaluan las descripciones de diferentes articulos, y se trata de obtener una clasificacion de los mismos. 

Se ha trabajado con dos redes neuronales distintas, a las cuales se fue modificando su estructura:

- Multi Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (CNN)

Multi Layer Perceptron (MLP) con CrossEntropyLoss.
Se trabajo probando combinaciones de los siguientes hiperparametros: 

  - Número y tamaño de capas ocultas.
  - Dropout.
  - Learning Rate.
  - Reg L2.
  - Nro de epochs.

Convolutional Neural Network (CNN) con CrossEntropyLoss.
Se trabajo probando combinaciones de los siguientes hiperparametros: 

- Cantidad y tamaño de los filtros.
- Tamaño de perceptron de salida.
- Learning rate.
- Reg L2.

Recurrent Neural Network (RNN - LSTM) con CrossEntropyLoss.
Se trabajo probando combinaciones de los siguientes hiperparametros: 

- Cantidad de capas recurrentes LSTM.
- Tamaño de capas recurrentes.
- Tamaño de perceptron de salida.
- Learning rate.
- Reg L2.

Se trabajo con Embeddings pre entrenados sobre el dataset preprocesado provisto.

De todas las pruebas realizadas, se obtuvieron mejores resultados con la RNN, al obtener valores de accuracy de 87% en validacion y 92% en test.

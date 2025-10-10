#  Proyecto: Predicción de Eficiencia Energética Edilicia

Este proyecto utiliza técnicas de Machine Learning y MLOps para predecir la eficiencia energética de diseños de edificios residenciales. Utiliza la navegación lateral para explorar los artefactos y el desarrollo de la Fase 1.

## Description

###  Problema

El proceso tradicional de diseño de edificios, especialmente al buscar la máxima eficiencia energética, es inherentemente **complejo, iterativo y consume mucho tiempo**. Los diseñadores dependen de simulaciones extensas y costosas, lo que retrasa significativamente la toma de decisiones óptimas.

###  Propuesta de Valor (Solución ML)

Se propone la creación de un **Modelo de Machine Learning de Regresión Supervisada** que sirva como una herramienta de simulación instantánea. Este modelo permitirá a los arquitectos e ingenieros predecir la eficiencia energética (cargas de calefacción y refrigeración) con solo ingresar las características de diseño (inputs X1 a X8).

#### Tarea de Predicción
El objetivo es estimar y cuantificar dos valores numéricos continuos:

* **Y1: Carga de Calefacción (Heating Load - HL)**
* **Y2: Carga de Refrigeración (Cooling Load - CL)**

#### Beneficios Clave
Al automatizar esta predicción, el proyecto busca:

1.  **Optimización Acelerada:** Permitir a los diseñadores testear instantáneamente cientos de geometrías posibles.
2.  **Reducción de Costos y Tiempo:** Acortar el ciclo de diseño al minimizar la necesidad de simulaciones costosas.


## Commands

The Makefile contains the central entry points for common tasks related to this project.


# ==================================
#       settings    
# ==================================
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from pysentimiento import create_analyzer

# ==================================
#       funciones   
# ==================================
# crear funcion para detectar las emociones tanto en ingles y español
def detect_emotion(text, lang = 'es'):
  emotion = create_analyzer(task="emotion", lang=lang)
  result = emotion.predict(text)
  resultado = result.output
  return resultado, result.probas[resultado]

#Hate speech
def detect_hate(text, lang ='es'):
  hate = create_analyzer(task="hate_speech", lang=lang)
  result = hate.predict(text)
  return result.probas

# ==================================
#       Encabezado    
# ==================================
st.title('Análisis de Emociones')

# ==================================
#       Menu y controles   
# ==================================
st.sidebar.subheader('Seleccione el Idioma del Texto a Introducir')
lang = st.sidebar.selectbox('Español o Ingles', ['es', 'en'])


# ==================================
#       Body    
# ==================================
# detectar emociones
text = st.text_input('Introducir su Texto para analisis')
emotion = detect_emotion(text, lang = lang)

# crear lista
lista = [list(emotion)]
resultado = pd.DataFrame(lista, columns = ['Emoción', 'Score'])
st.write(resultado)

#detectar discurso de odio
hate = detect_hate(text, lang = lang)
res= pd.Series(hate)
df = res.to_frame().reset_index()
df.columns = ['Discurso de Odio','Probabilidad']
st.write(df)

#creado por:
st.warning('Realizado por Luis Hernández')
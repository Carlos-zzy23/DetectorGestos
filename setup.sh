#!/bin/bash

# Este script instala las dependencias usando Python 3.11
# y luego ejecuta la aplicación con el intérprete correcto.

# 1. Instala la versión de Python 3.11 (si no está)
sudo apt-get update
sudo apt-get install python3.11 python3.11-dev python3.11-venv -y

# 2. Crea un nuevo entorno virtual con Python 3.11
python3.11 -m venv venv_311

# 3. Activa el entorno virtual
source venv_311/bin/activate

# 4. Actualiza pip y instala las dependencias desde requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

# 5. Ejecuta la aplicación de Streamlit usando el intérprete 3.11
# Streamlit ya está instalado en el venv_311
streamlit run app_gestos.py

import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np # Necesario para la conversión de numpy/PIL

# --- Configuración de MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Función para Clasificar Gestos ---
def classify_gesture(hand_landmarks):
    # (El resto de la lógica de classify_gesture debe ir aquí)
    tip_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
               mp_hands.HandLandmark.RING_FINGER_TIP, 
               mp_hands.HandLandmark.PINKY_TIP]
    
    fingers_up = 0
    
    # Chequeo del Pulgar
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if thumb_tip.x < thumb_ip.x: 
        fingers_up += 1

    # Chequeo de los demás dedos
    for id in tip_ids:
        tip = hand_landmarks.landmark[id]
        pip = hand_landmarks.landmark[id - 2]
        if tip.y < pip.y:
            fingers_up += 1

    # Clasificación simple
    if fingers_up == 5:
        return "🖐️ Mano Abierta"
    elif fingers_up == 1 and hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        return "👆 Apuntando"
    elif fingers_up == 0:
        return "✊ Puño Cerrado"
    else:
        return f"Dedos Arriba: {fingers_up}"

# =========================================================
# --- Configuración de Streamlit ---
# =========================================================
st.title("Reconocimiento de Gestos con Streamlit")
st.markdown("Presiona **Iniciar Webcam** para comenzar el reconocimiento.")

# Placeholder para el frame de video
frame_placeholder = st.empty()
gesture_placeholder = st.empty()
stop_button_placeholder = st.empty()

# Bandera para controlar la ejecución
if 'running' not in st.session_state:
    st.session_state['running'] = False

# Botón de inicio
if st.button("Iniciar Webcam"):
    st.session_state['running'] = True

# Botón de detención (solo visible cuando está corriendo)
if st.session_state['running']:
    if st.button("Detener Webcam"):
        st.session_state['running'] = False

if st.session_state['running']:
    # 1. Iniciar la captura de video
    cap = cv2.VideoCapture(0)
    
    # 2. Bucle principal de procesamiento
    while cap.isOpened() and st.session_state['running']:
        ret, frame = cap.read()
        if not ret:
            st.warning("No se puede acceder a la cámara. Reintentando...")
            cap = cv2.VideoCapture(0) # Reintentar
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_gesture = "Esperando Gesto..."
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_gesture = classify_gesture(hand_landmarks)

        # 5. Mostrar el resultado en el frame de OpenCV
        cv2.putText(frame, current_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        # 6. Conversión para Streamlit y actualización
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        frame_placeholder.image(img, channels="RGB", use_column_width=True)
        gesture_placeholder.markdown(f"**Gesto Detectado:** **{current_gesture}**")
        
    # 7. Liberar recursos al salir del bucle
    cap.release()
    st.session_state['running'] = False # Asegurar que la bandera se resetee al terminar

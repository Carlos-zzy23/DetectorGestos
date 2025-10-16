import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, VideoTransformerBase
from typing import Union, List, Dict, Callable

# --- Configuración de MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar MediaPipe globalmente para rehusar el objeto
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# --- Función para Clasificar Gestos (Igual que antes) ---
def classify_gesture(hand_landmarks):
    tip_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
               mp_hands.HandLandmark.RING_FINGER_TIP, 
               mp_hands.HandLandmark.PINKY_TIP]
    fingers_up = 0
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if thumb_tip.x < thumb_ip.x: 
        fingers_up += 1

    for id in tip_ids:
        tip = hand_landmarks.landmark[id]
        pip = hand_landmarks.landmark[id - 2]
        if tip.y < pip.y:
            fingers_up += 1

    if fingers_up == 5:
        return "🖐️ Mano Abierta"
    elif fingers_up == 1 and hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        return "👆 Apuntando"
    elif fingers_up == 0:
        return "✊ Puño Cerrado"
    else:
        return f"Dedos Arriba: {fingers_up}"


# =========================================================
# --- CLASE TRANSFORMADORA DE VIDEO (OPTIMIZADA Y LIGERA) ---
# =========================================================

class GestureRecognizer(VideoTransformerBase):
    def transform(self, frame):
        # 1. Convierte el frame de video a un array de OpenCV (BGR)
        img = frame.to_ndarray(format="bgr")
        
        # 2. Voltear y convertir a RGB
        img = cv2.flip(img, 1)
        # Reducir la resolución de manera interna (si es necesario)
        img_resized = cv2.resize(img, (320, 240)) 
        rgb_frame = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Procesamiento de MediaPipe
        results = hands.process(rgb_frame) # Usamos el objeto hands global

        current_gesture = "Esperando Gesto..."
        
        # 4. Dibujar y Clasificar
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Usamos el frame original (más grande) para dibujar si quieres alta calidad, 
                # o el frame reducido (img_resized) si quieres consistencia. Usaremos el reducido.
                mp_drawing.draw_landmarks(img_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_gesture = classify_gesture(hand_landmarks)

        # 5. Mostrar el resultado
        cv2.putText(img_resized, current_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # 6. Retorno seguro del frame procesado (pequeño)
        return img_resized


# =========================================================
# --- CONFIGURACIÓN DE STREAMLIT ---
# =========================================================

st.title("Reconocimiento de Gestos con Streamlit (FINAL)")
st.markdown("Permite el acceso a la cámara. El procesamiento debe verse fluido.")

webrtc_streamer(
    key="gesture-recognizer-final",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GestureRecognizer,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320}, 
            "height": {"ideal": 240},
            "frameRate": {"ideal": 10}, # 10 FPS es crucial para que la CPU respire
        }, 
        "audio": False
    },
    async_processing=True,
)

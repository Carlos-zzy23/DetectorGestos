import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Configuración de MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Función para Clasificar Gestos ---
def classify_gesture(hand_landmarks):
    """Clasifica el gesto basándose en la posición de los dedos."""
    
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
# --- CLASE TRANSFORMADORA DE VIDEO (OPTIMIZADA) ---
# =========================================================

class GestureRecognizer(VideoTransformerBase):
    def __init__(self):
        # Inicializa MediaPipe dentro de la clase
        self.hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.display_width = 320  # Nueva resolución más pequeña
        self.display_height = 240 # Nueva resolución más pequeña

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr")
        
        # 1. Reducir la Resolución del Frame (¡CLAVE PARA LA OPTIMIZACIÓN!)
        img = cv2.resize(img, (self.display_width, self.display_height))
        
        # 2. Voltear la imagen y convertir a RGB
        img = cv2.flip(img, 1)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Opcional: Probar escala de grises para ver si mejora la velocidad
        # gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # results = self.hands.process(gray_frame)
        
        # 3. Procesamiento de MediaPipe
        results = self.hands.process(rgb_frame)

        current_gesture = "Esperando Gesto..."
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_gesture = classify_gesture(hand_landmarks)

        # 4. Mostrar el resultado
        cv2.putText(img, current_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # 5. Retorno seguro
        return img
# =========================================================
# --- CONFIGURACIÓN DE STREAMLIT ---
# =========================================================

st.title("Reconocimiento de Gestos con Streamlit (Webcam)")
st.markdown("Permite el acceso a la cámara. El procesamiento se ve en el recuadro.")

# El componente webrtc_streamer maneja la conexión, los botones y el bucle de video.
webrtc_streamer(
    key="gesture-recognizer",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GestureRecognizer,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320},  # Pedir al navegador una resolución baja
            "height": {"ideal": 240},
            "frameRate": {"ideal": 10}, # Pedir solo 10 frames por segundo (¡CLAVE!)
        }, 
        "audio": False
    },
    async_processing=True,
)
)

import cv2
import numpy as np
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R 
import winsound

# --- 1. CONFIGURACIÓN DE MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

TIP_INDEX = mp_hands.HandLandmark.INDEX_FINGER_TIP
TIP_THUMB = mp_hands.HandLandmark.THUMB_TIP
TIP_MIDDLE = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
WRIST = mp_hands.HandLandmark.WRIST

# --- 2. MODELADO Y ESTADO DEL OBJETO 3D (Planeta) ---
object_3d = {
    'name': 'Planeta Tierra', 
    'scale': 1.6,      
    'rotation': [0, 0, 0], 
    'position': [0, 0, 0] 
}

# Configuración de Sonido
def play_sound(action):
    """Reproduce un archivo WAV usando el módulo winsound."""
    sound_file = 'alerta.wav' 
    
    try:
        # CORRECCIÓN CLAVE: Usamos SND_NOSTOP para que el sonido no se corte si se llama rápidamente.
        winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NOSTOP) 
        print(f"🔊 {action} Planeta! (Reproduciendo {sound_file})")
        
    except Exception as e:
        print(f"🔊 {action} Planeta! (ERROR DE SONIDO: {e}). Asegúrate de que el archivo es WAV y existe.")
        pass

# Define la malla de la Esfera (Planeta)
def create_sphere_mesh(scale):
    # Aumentamos la resolución para un mejor color/sombreado
    u = np.linspace(0, 2 * np.pi, 50) 
    v = np.linspace(0, np.pi, 50)
    x = scale * np.outer(np.cos(u), np.sin(v))
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# --- 3. FUNCIÓN DE RENDERIZADO 3D (Matplotlib) ---
fig = plt.figure(figsize=(8, 8), facecolor='black') 
ax = fig.add_subplot(111, projection='3d', facecolor='black') 
ax.set_title(f"{object_3d['name']} (Escala: {object_3d['scale']:.1f})", color='white')

AXIS_RANGE = 1.5 
ax.set_box_aspect([1, 1, 1]) 

def clean_plot(ax):
    """Limpia los ejes, la cuadrícula y el fondo para el efecto 'espacio'."""
    ax.set_axis_off() 
    ax.grid(False) 
    
    # >>> CORRECCIÓN DE ATTRIBUTEERROR: Usamos .pane en lugar de .w_xaxis <<<
    ax.xaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.set_xlim([-AXIS_RANGE, AXIS_RANGE])
    ax.set_ylim([-AXIS_RANGE, AXIS_RANGE])
    ax.set_zlim([-AXIS_RANGE, AXIS_RANGE])
    
def update_3d_plot():
    """Actualiza la posición, escala y rotación del objeto 3D."""
    ax.cla() 

    # --- Aplicar la Estética ---
    clean_plot(ax) 
    
    # 1. Crear el objeto con la escala actual
    x, y, z = create_sphere_mesh(object_3d['scale'] * 0.5) 
    
    # 2. Aplicar ROTACIÓN
    r = R.from_euler('xyz', object_3d['rotation'], degrees=True)
    rotated_coords = r.apply(np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1))
    x_rotated = rotated_coords[:, 0].reshape(x.shape)
    y_rotated = rotated_coords[:, 1].reshape(y.shape)
    z_rotated = rotated_coords[:, 2].reshape(z.shape)
    
    # 3. Aplicar MOVIMIENTO (Traslación)
    pos_x, pos_y, pos_z = object_3d['position']
    
    # CLAVE PARA TEXTURA SIMULADA: Usamos los valores Z para el color
    # Almacenamos Z_rotada para usarla como dato de color (cstride y rstride=1 son importantes)
    color_data = z_rotated 
    
    # Dibujar la superficie del planeta
    # ELIMINAR: cmap='ocean'
    ax.plot_surface(x_rotated + pos_x, y_rotated + pos_y, z_rotated + pos_z, 
                     facecolors=plt.cm.terrain(color_data), # <-- CAMBIO CLAVE
                     rstride=1, cstride=1, 
                     alpha=1.0, edgecolor='none', linewidth=0, shade=True) # shade=True para mejor efecto

    ax.set_title(f"Planeta (Escala: {object_3d['scale']:.1f})", color='white')
    # Ajustamos la vista para que no siempre mire al frente
    ax.view_init(elev=30, azim=object_3d['rotation'][2] * 0.5) 
    ax.figure.canvas.draw()
    
# --- 4. FUNCIÓN DE DETECCIÓN DE GESTOS ---

def get_gesture(hand_landmarks):
    y_thumb = hand_landmarks.landmark[TIP_THUMB].y
    y_index = hand_landmarks.landmark[TIP_INDEX].y
    y_middle = hand_landmarks.landmark[TIP_MIDDLE].y
    y_base_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    dist_x_norm = abs(hand_landmarks.landmark[TIP_INDEX].x - hand_landmarks.landmark[TIP_THUMB].x)
    is_index_up = y_index < y_base_index * 0.9 
    is_middle_up = y_middle < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * 0.9

    if dist_x_norm < 0.05: # PINZA (Escala)
        y_wrist = hand_landmarks.landmark[WRIST].y
        return "DISMINUIR" if y_wrist < 0.4 else "AMPLIAR"

    if is_index_up and not is_middle_up and dist_x_norm > 0.1: # Señalar (Girar)
        return "GIRAR"

    if is_index_up and is_middle_up and dist_x_norm > 0.1: # Palma Abierta (Mover)
        return "MOVER"

    return "NINGUNO"

# --- 5. BUCLE PRINCIPAL DE CAPTURA Y ACTUALIZACIÓN ---
cap = cv2.VideoCapture(0)
last_scale = object_3d['scale']
last_action_time = 0

def capture_and_process(i):
    """Captura el frame, procesa MediaPipe y actualiza el estado 3D."""
    global object_3d, last_scale, last_action_time
    
    ret, frame = cap.read()
    if not ret: return
    
    frame = cv2.flip(frame, 1)
    cam_h, cam_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    gesto_detectado = "NINGUNO"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_index_norm = hand_landmarks.landmark[TIP_INDEX].x
            y_index_norm = hand_landmarks.landmark[TIP_INDEX].y
            
            gesto = get_gesture(hand_landmarks)
            gesto_detectado = gesto
            
            current_time = time.time()
            sound_delay = (current_time - last_action_time) > 0.3

            if gesto == 'MOVER':
                # Mover el objeto en los ejes X e Y del espacio 3D
                object_3d['position'][0] = 2 * (x_index_norm - 0.5) 
                object_3d['position'][1] = -2 * (y_index_norm - 0.5) 
                object_3d['position'][2] = 0.0
                
            elif gesto == 'AMPLIAR' or gesto == 'DISMINUIR':
                # GESTO: ESCALA (ÚNICO LUGAR DONDE DEBE HABER SONIDO)
                scale_change = 0.1
                if gesto == 'AMPLIAR':
                    object_3d['scale'] = np.clip(object_3d['scale'] + scale_change, 0.3, 3.0)
                else:
                    object_3d['scale'] = np.clip(object_3d['scale'] - scale_change, 0.3, 3.0)
                    
                if sound_delay:
                    play_sound(gesto)
                    last_action_time = current_time # Resetea el tiempo para evitar spam
                    
                last_scale = object_3d['scale'] # Actualiza la escala después del cambio
            
            # ELIMINA CUALQUIER LLAMADA A play_sound() AQUÍ
            elif gesto == 'GIRAR':
                # Rotación continua en el eje Z (SIN SONIDO)
                object_3d['rotation'][2] = (object_3d['rotation'][2] + 10) % 360
            
    # Mostrar el estado en la ventana de la cámara
    cv2.putText(frame, f"ALGORITMO: DETECTOR", (10, cam_h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"GESTO: {gesto_detectado}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Control de Objeto 3D ', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        plt.close(fig)
        cap.release()
        cv2.destroyAllWindows()
        exit()
        
    update_3d_plot() 

# Usar FuncAnimation para mantener el plot 3D interactivo
ani = FuncAnimation(fig, capture_and_process, interval=50, cache_frame_data=False)
plt.show()

cap.release()
cv2.destroyAllWindows()
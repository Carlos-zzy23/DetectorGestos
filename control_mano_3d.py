import cv2
import mediapipe as mp
import numpy as np
import time
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pygame
from PIL import Image # Necesario para cargar texturas

# --- CONFIGURACIÓN DE LÍMITES GLOBALES ---
MAX_MOVE = 2.5
MIN_MOVE = -2.5
MAX_SCALE = 2.0
MIN_SCALE = 0.5
CAP_WIDTH, CAP_HEIGHT = 800, 600
TEXTURE_ID = None

# Inicialización de Pygame y Sonido
pygame.init()
pygame.mixer.init()
try:
    sound = pygame.mixer.Sound("Alerta.wav") 
except pygame.error:
    print("ERROR: No se encontró Alerta.wav. El sonido no funcionará.")
    sound = None
    
# --- VARIABLES DE ESTADO ---
state = {
    "rotate_x": 0,
    "rotate_y": 0,
    "scale_factor": 1.5,
    "move_x": 0.0,
    "move_y": 0.0,
    "gesture": "ninguno",
    "last_sound_time": 0.0
}

# Iniciar mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Puntos de referencia clave para la lógica de gestos
TIP_INDEX = mp_hands.HandLandmark.INDEX_FINGER_TIP
TIP_THUMB = mp_hands.HandLandmark.THUMB_TIP
WRIST = mp_hands.HandLandmark.WRIST

# --- 1. CARGADOR DE MODELO OBJ (CORRECCIÓN DE NORMALES Y UV) ---
class OBJModel:
    def __init__(self, filename):
        self.vertices = []
        self.texcoords = []
        self.normals = [] # Nueva lista para almacenar las normales
        self.faces = []
        self.load(filename)

    def load(self, filename):
        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('v '):
                        self.vertices.append([float(x) for x in line.split()[1:]])
                    elif line.startswith('vt '):
                        parts = line.split()[1:]
                        if len(parts) >= 2:
                            self.texcoords.append([float(parts[0]), float(parts[1])])
                    elif line.startswith('vn '): # Cargar normales
                        self.normals.append([float(x) for x in line.split()[1:]])
                    elif line.startswith('f '):
                        parts = line.split()[1:]
                        face = []
                        for p in parts:
                            if p:
                                # f v/vt/vn
                                face.append([int(i) - 1 if i else -1 for i in p.split('/')])
                        self.faces.append(face)
            print(f"✅ Archivo OBJ ({filename}) cargado.")
        except FileNotFoundError:
            print(f"⚠️ ERROR: No se encontró {filename}. Usando fallback.")
        except Exception as e:
            print(f"Error al parsear {filename}: {e}")

# Usamos los archivos del dinosaurio
dino_model = OBJModel("11677_dinosaur_v1_L3.obj") 

if dino_model.vertices:
    dino_vertices_np = np.array(dino_model.vertices)
    max_val = np.max(np.abs(dino_vertices_np))
    dino_model.vertices = (dino_vertices_np / (max_val * 2.5)).tolist() 
    print("✅ Modelo OBJ normalizado.")
else:
    print("⛔️ Usando Cubo de Fallback (Modelo no cargado).")

# --- 2. Lógica de Carga de Textura y OpenGL ---
def load_texture(filename):
    global TEXTURE_ID
    texture_path = "dinosaur_diffuse.jpg" 
    
    try:
        image = Image.open(texture_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM) 
        
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        
        image_data = np.array(list(image.getdata()), np.uint8)
        
        TEXTURE_ID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, TEXTURE_ID)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.size[0], image.size[1], 0, 
                     GL_RGB, GL_UNSIGNED_BYTE, image_data)
        
        glEnable(GL_TEXTURE_2D)
        print("✅ Textura cargada correctamente.")
    except FileNotFoundError:
        print(f"⚠️ ERROR: No se encontró la textura {texture_path}. Usando color sólido.")
        TEXTURE_ID = None
    except Exception as e:
        print(f"⚠️ Error al cargar la textura: {e}. Usando color sólido.")
        TEXTURE_ID = None


def init_gl(width, height):
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE) # Para dibujar solo las caras visibles
    
    # 🌟 CORRECCIÓN CLAVE 1: Configuración de Iluminación
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
    # Definir la posición y color de la luz (Light source)
    glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0]) # Luz ambiental
    
    # Habilitar el material y el color
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    load_texture("dinosaur_diffuse.jpg")


def draw_dino():
    global state, dino_model

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Aplicar transformaciones
    glTranslatef(state['move_x'], state['move_y'], -6) 
    glRotatef(-90.0, 0.0, 0.0, 1.0)
    glScalef(state['scale_factor'], state['scale_factor'], state['scale_factor'])
    
    if dino_model.vertices:
        if TEXTURE_ID:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, TEXTURE_ID)
            glColor3f(1.0, 1.0, 1.0) # Asegurar color blanco para ver la textura
        else:
            glDisable(GL_TEXTURE_2D)
            glColor3f(0.2, 0.8, 0.3) # Fallback verde

        #  Renderizado Sólido (Usando GL_QUADS o GL_POLYGON)
        glBegin(GL_TRIANGLES) 
        
        for face in dino_model.faces:
            # Lógica para manejar caras de 3 o 4 vértices
            if len(face) == 4: # Cuadrado (convertir a dos triángulos)
                indices = [0, 1, 2, 0, 2, 3] 
            elif len(face) == 3: # Triángulo simple
                indices = [0, 1, 2]
            else:
                continue

            for i in indices:
                v_idx, tc_idx, n_idx = face[i]
                
                # Normales (CRUCIAL para la iluminación)
                if n_idx != -1 and n_idx < len(dino_model.normals):
                    glNormal3fv(dino_model.normals[n_idx])

                # Coordenadas de Textura
                if TEXTURE_ID and tc_idx != -1 and tc_idx < len(dino_model.texcoords):
                    glTexCoord2fv(dino_model.texcoords[tc_idx])
                
                # Vértices (Geometría)
                if v_idx != -1 and v_idx < len(dino_model.vertices):
                    glVertex3fv(dino_model.vertices[v_idx])
                
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
    else:
        # Dibujo de fallback (Cubo Rojo)
        glColor3f(1.0, 0.0, 0.0)
        glutWireCube(1.0) 
        
    glutSwapBuffers()

# --- 3. Lógica de Gestos y Bucle ---

def detectar_gesto(hand_landmarks):
    tip_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    tip_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    
    distancia_pulgar_indice = np.linalg.norm(np.array([
        hand_landmarks.landmark[TIP_THUMB].x - hand_landmarks.landmark[TIP_INDEX].x,
        hand_landmarks.landmark[TIP_THUMB].y - hand_landmarks.landmark[TIP_INDEX].y,
        hand_landmarks.landmark[TIP_THUMB].z - hand_landmarks.landmark[TIP_INDEX].z
    ]))

    if distancia_pulgar_indice < 0.03: return "escalar" 

    if tip_index < palm * 0.8 and tip_middle < palm * 0.8: return "mover"

    if tip_index < palm * 0.8: return "girar" 
    
    return "ninguno"


def detectar_y_actualizar_gesto():
    global state
    
    ret, frame = cap.read()
    if not ret: return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = "ninguno"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            current_gesture = detectar_gesto(hand_landmarks)
            cx = hand_landmarks.landmark[TIP_INDEX].x
            cy = hand_landmarks.landmark[TIP_INDEX].y

            if current_gesture == "girar":
                state['rotate_y'] = (state['rotate_y'] + 5) % 360

            elif current_gesture == "mover":
                state['move_x'] = np.interp(cx, [0.1, 0.9], [MIN_MOVE, MAX_MOVE])
                state['move_y'] = np.interp(cy, [0.1, 0.9], [MAX_MOVE, MIN_MOVE]) 

            elif current_gesture == "escalar":
                scale_factor_raw = 1.0 - cy 
                new_scale = np.interp(scale_factor_raw, [0.1, 0.8], [MIN_SCALE, MAX_SCALE])
                
                if abs(new_scale - state['scale_factor']) > 0.05 and time.time() - state['last_sound_time'] > 0.3:
                    state['scale_factor'] = new_scale
                    if sound:
                         sound.play()
                    state['last_sound_time'] = time.time()
                    
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    state['gesture'] = current_gesture
    
    cv2.putText(frame, f"Gesto: {state['gesture']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Escala: {state['scale_factor']:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camara', frame)
    cv2.waitKey(1)

def update_scene(value):
    detectar_y_actualizar_gesto()
    glutPostRedisplay()
    glutTimerFunc(10, update_scene, 0)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(CAP_WIDTH, CAP_HEIGHT)
    glutCreateWindow(b"Dinosaurio 3D con Control de Mano - PyOpenGL")
    init_gl(CAP_WIDTH, CAP_HEIGHT)
    
    glutDisplayFunc(draw_dino)
    
    glutTimerFunc(10, update_scene, 0) 
    
    glutMainLoop()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Ha ocurrido un error en la ejecución: {e}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

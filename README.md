# ✋ Detección y Control de Gestos 3D

Proyecto de visión por computadora que implementa la detección de gestos de la mano en tiempo real (utilizando MediaPipe) para **controlar la manipulación de un objeto 3D** (un dinosaurio) renderizado con OpenGL.

## 🌟 Características

- **Detección de la Mano:** Rastreo de 21 puntos de referencia (landmarks) en la mano a través de la webcam.
- **Visualización 3D:** Renderizado del modelo `11677_dinosaur_v1_L3.obj` en un contexto OpenGL/Pygame.
- **Control por Gesto:** La posición y/o rotación del modelo 3D es controlada por un gesto específico detectado en la mano.
- **Feedback Auditivo:** Reproducción de `Alerta.wav` como señal de un evento o detección.

## ⚙️ Estructura del Proyecto

| Archivo/Carpeta | Descripción |
| :--- | :--- |
| `control_mano_3d.py` | Script prueba con Dino 3D que integra MediaPipe, OpenGL, Pygame y la lógica de gestos. |
| `gestos_3d_mediapipe_final` | Script principal funcional con Planeta 3D que integra MediaPipe, OpenGL, Pygame y la lógica de gestos. |
| `11677_dinosaur_v1_L3.obj` | Archivo del modelo 3D (Object File) del dinosaurio. |
| `11677_dinosaur_v1_L3.mtl` | Archivo de materiales que define la apariencia del modelo 3D. |
| `dinosaur_diffuse.jpg` | Textura (imagen) del dinosaurio. |
| `Alerta.wav` | Archivo de sonido para notificaciones de eventos. |
| `requirements.txt` | Lista de dependencias de Python y sus versiones. |
| `.gitignore` | Ignora archivos temporales y carpetas de entornos virtuales. |

## 🚀 Guía de Instalación y Ejecución

### Requisitos

Asegúrate de tener **Python 3.x** instalado y una **webcam** funcional.

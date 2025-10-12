# EmoVisionPy - Sistema de Detección de Emociones Faciales

Aplicación web desarrollada en Python que detecta emociones faciales en tiempo real usando inteligencia artificial. Utiliza la cámara web para análisis continuo.

## 🚀 Características

- **Detección en Tiempo Real**: Acceso a la cámara desde el navegador
- **Análisis Continuo**: Detección automática cada 2 segundos
- **Detección Facial con OpenCV**: Cuadro verde alrededor del rostro detectado
- **Interfaz Moderna**: Diseño responsive con Bootstrap
- **Sin Base de Datos**: Procesamiento completamente en memoria
- **7 Emociones**: Feliz, triste, enojado, sorprendido, neutral, asustado, disgustado

## 🛠️ Tecnologías

- **Backend**: Python, Flask
- **IA**: DeepFace, OpenCV, TensorFlow
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5

## 📋 Requisitos

- Python 3.8+
- Navegador moderno con soporte para getUserMedia API
- Cámara web (para detección en tiempo real)
- 4GB RAM mínimo

## 🔧 Instalación

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar aplicación
python app.py

# 3. Abrir navegador
http://localhost:5000
```

## 📱 Uso

1. Abre la aplicación en tu navegador
2. Haz clic en "Iniciar Cámara"
3. Permite acceso a la cámara cuando se solicite
4. Las emociones se detectarán automáticamente cada 2 segundos
5. Los resultados se muestran en tiempo real sobre el video

## 📁 Estructura

```
PythonReconocimientoFacial/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias
├── README.md             # Documentación
├── templates/
│   └── index.html        # Interfaz web
└── static/uploads/       # Archivos temporales
```

## 🐛 Problemas Comunes

- **Error de cámara**: Permite acceso a la cámara en el navegador
- **Error de memoria**: Cierra otras aplicaciones, necesitas 4GB+ RAM
- **Primera carga lenta**: 30-60 segundos para descargar modelo de IA

## 🔒 Privacidad

- Procesamiento 100% local
- No se envían datos a servidores externos
- Archivos temporales se eliminan automáticamente

---

**¡Disfruta detectando emociones con EmoVisionPy! 🎉**

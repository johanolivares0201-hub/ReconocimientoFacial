# EmoVisionPy - Sistema de Detección de Emociones y Formas Geométricas

Aplicación web desarrollada en Python que detecta emociones faciales en tiempo real y analiza formas geométricas y colores en videos. Utiliza inteligencia artificial y visión por computadora para análisis automático.

## 🚀 Características

### Detección de Emociones
- **Detección en Tiempo Real**: Acceso a la cámara desde el navegador
- **Análisis Continuo**: Detección automática cada 2 segundos
- **Detección Facial con OpenCV**: Cuadro verde alrededor del rostro detectado
- **7 Emociones**: Feliz, triste, enojado, sorprendido, neutral, asustado, disgustado

### Detección de Formas y Colores
- **Análisis de Videos**: Procesamiento frame por frame de videos
- **Formas Geométricas**: Círculo, triángulo, cuadrado, rectángulo, rombo
- **Reconocimiento de Colores**: Detección de colores dominantes
- **Estadísticas Detalladas**: Conteo y porcentajes de formas y colores
- **Múltiples Formatos**: MP4, AVI, MOV, MKV, WMV

### Interfaz
- **Interfaz Moderna**: Diseño responsive con Bootstrap
- **Navegación por Pestañas**: Separación clara de funcionalidades
- **Sin Base de Datos**: Procesamiento completamente en memoria

## 🛠️ Tecnologías

- **Backend**: Python, Flask
- **IA**: DeepFace, OpenCV, TensorFlow
- **Visión por Computadora**: OpenCV, NumPy, SciPy
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Procesamiento**: NumPy, Pillow, Matplotlib

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

### Detección de Emociones
1. Abre la aplicación en tu navegador
2. Ve a la pestaña "Detección de Emociones"
3. Haz clic en "Iniciar Cámara"
4. Permite acceso a la cámara cuando se solicite
5. Las emociones se detectarán automáticamente cada 2 segundos
6. Los resultados se muestran en tiempo real sobre el video

### Detección de Formas y Colores
1. Ve a la pestaña "Detección de Formas y Colores"
2. Arrastra y suelta un video o haz clic para seleccionar
3. Haz clic en "Procesar Video"
4. Espera a que se complete el análisis (puede tomar varios minutos)
5. Revisa los resultados detallados de formas y colores detectados

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
- **Video no procesa**: Verifica que el formato sea compatible (MP4, AVI, MOV, MKV, WMV)
- **Procesamiento lento**: Videos largos pueden tomar varios minutos

## 🔒 Privacidad

- Procesamiento 100% local
- No se envían datos a servidores externos
- Archivos temporales se eliminan automáticamente

---

**¡Disfruta detectando emociones y formas con EmoVisionPy! 🎉**

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import base64
import os

app = Flask(__name__)

# Configuración para archivos temporales
UPLOAD_FOLDER = 'static/uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def detect_emotion_with_face_box(image_path):
    """Detecta emociones y dibuja cuadro alrededor del rostro"""
    try:
        # Verificar que el archivo existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"El archivo {image_path} no existe")
        
        # Cargar imagen con OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("No se pudo cargar la imagen")
        
        # Analizar emociones y obtener región del rostro
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # Normalizar resultado si es una lista
        if isinstance(result, list):
            result = result[0]
        
        if 'emotion' not in result:
            raise ValueError("No se encontraron emociones en el resultado")
        
        emotions = result['emotion']
        
        # Encuentra la emoción dominante
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        # Dibujar cuadro alrededor del rostro si se detectó
        face_box = None
        if 'region' in result:
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Dibujar rectángulo verde alrededor del rostro
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Agregar texto con la emoción detectada
            emotion_text = f"{dominant_emotion[0]}: {dominant_emotion[1]:.1f}%"
            cv2.putText(image, emotion_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            face_box = {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        
        # Guardar imagen con el cuadro dibujado
        output_path = image_path.replace('.jpg', '_detected.jpg')
        cv2.imwrite(output_path, image)
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'confidence': round(dominant_emotion[1], 2),
            'all_emotions': emotions,
            'face_box': face_box,
            'output_image': output_path
        }
    except Exception as e:
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.0,
            'all_emotions': {},
            'face_box': None,
            'error': str(e)
        }

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/detect_emotion_frame', methods=['POST'])
def detect_emotion_frame():
    """Endpoint para detectar emociones en frame de video"""
    temp_path = None
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No se encontró imagen'}), 400
        
        # Decodifica la imagen base64
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Error al procesar la imagen'}), 400
        
        # Guarda temporalmente la imagen
        temp_filename = f"temp_frame_{np.random.randint(1000, 9999)}.jpg"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Asegurar que el directorio existe
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        if not cv2.imwrite(temp_path, image):
            return jsonify({'error': 'Error al guardar frame temporal'}), 500
        
        # Detecta emociones y dibuja cuadro del rostro
        result = detect_emotion_with_face_box(temp_path)
        
        # Convertir imagen con cuadro a base64 para enviar al frontend
        if 'output_image' in result and result['output_image']:
            try:
                with open(result['output_image'], 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    result['image_with_face_box'] = f"data:image/jpeg;base64,{img_data}"
                
                # Limpiar archivo de imagen con cuadro
                os.remove(result['output_image'])
            except Exception as e:
                print(f"Error procesando imagen con cuadro: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500
    
    finally:
        # Limpia el archivo temporal
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

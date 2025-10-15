from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import base64
import os
import json
from scipy import ndimage
from collections import Counter

app = Flask(__name__)

# Configuración para archivos temporales
UPLOAD_FOLDER = 'static/uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_shapes_and_colors(image):
    """Detecta formas geométricas y colores en una imagen"""
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes con Canny - usar múltiples umbrales
        edges1 = cv2.Canny(blurred, 30, 100)  # Más sensible
        edges2 = cv2.Canny(blurred, 50, 150)  # Estándar
        edges3 = cv2.Canny(blurred, 100, 200) # Menos sensible
        
        # Combinar los bordes detectados
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, edges3)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        colors = []
        
        for contour in contours:
            # Filtrar contornos muy pequeños
            area = cv2.contourArea(contour)
            if area < 500:  # Área mínima
                continue
            
        # Aproximar el contorno - usar múltiples niveles de precisión
        epsilon1 = 0.02 * cv2.arcLength(contour, True)
        epsilon2 = 0.03 * cv2.arcLength(contour, True)
        
        # Intentar con diferentes niveles de aproximación
        approx1 = cv2.approxPolyDP(contour, epsilon1, True)
        approx2 = cv2.approxPolyDP(contour, epsilon2, True)
        
        # Usar la aproximación que tenga 4 vértices si está disponible
        if len(approx1) == 4:
            approx = approx1
        elif len(approx2) == 4:
            approx = approx2
        else:
            approx = approx1  # Usar la más precisa por defecto
        
        # Detectar forma
        shape = detect_shape(approx, area)
        
        if shape:
            # Obtener el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extraer región de color
            roi = image[y:y+h, x:x+w]
            color = detect_dominant_color(roi)
            
            shapes.append({
                'shape': shape,
                'area': int(area),
                'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'color': color
            })
            
            colors.append(color)
        
        # Método adicional para detectar rectángulos usando detección de rectas
        rectangles = detect_rectangles_alternative(image)
        shapes.extend(rectangles)
        
        # Contar colores dominantes en toda la imagen
        dominant_colors = get_dominant_colors(image)
        
        return {
            'shapes': shapes,
            'colors_in_shapes': colors,
            'dominant_colors': dominant_colors,
            'total_shapes': len(shapes)
        }
        
    except Exception as e:
        return {
            'shapes': [],
            'colors_in_shapes': [],
            'dominant_colors': [],
            'total_shapes': 0,
            'error': str(e)
        }

def detect_rectangles_alternative(image):
    """Método alternativo para detectar rectángulos usando detección de rectas"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro bilateral para preservar bordes
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detectar bordes
        edges = cv2.Canny(filtered, 50, 150, apertureSize=3)
        
        # Detectar líneas usando HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        rectangles = []
        
        if lines is not None:
            # Agrupar líneas que forman rectángulos
            for i, line1 in enumerate(lines):
                for j, line2 in enumerate(lines[i+1:], i+1):
                    x1, y1, x2, y2 = line1[0]
                    x3, y3, x4, y4 = line2[0]
                    
                    # Verificar si las líneas forman un rectángulo
                    if is_rectangle_lines(line1[0], line2[0]):
                        # Calcular el rectángulo
                        rect = calculate_rectangle_from_lines(line1[0], line2[0])
                        if rect:
                            x, y, w, h = rect
                            
                            # Verificar que el rectángulo sea válido
                            if w > 20 and h > 20:
                                # Extraer región de color
                                roi = image[y:y+h, x:x+w]
                                color = detect_dominant_color(roi)
                                
                                rectangles.append({
                                    'shape': 'rectángulo',
                                    'area': int(w * h),
                                    'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                                    'color': color
                                })
        
        return rectangles
        
    except Exception as e:
        return []

def is_rectangle_lines(line1, line2):
    """Verifica si dos líneas pueden formar un rectángulo"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Verificar si las líneas son aproximadamente perpendiculares o paralelas
    # y están cerca entre sí
    dist1 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
    dist2 = np.sqrt((x2-x4)**2 + (y2-y4)**2)
    
    # Si las líneas están cerca, podrían formar un rectángulo
    return dist1 < 100 or dist2 < 100

def calculate_rectangle_from_lines(line1, line2):
    """Calcula las coordenadas del rectángulo formado por dos líneas"""
    try:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calcular los puntos extremos
        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)
        
        width = max_x - min_x
        height = max_y - min_y
        
        return (min_x, min_y, width, height)
    except:
        return None

def detect_shape(approx, area):
    """Detecta la forma geométrica basada en la aproximación del contorno"""
    vertices = len(approx)
    
    if vertices == 3:
        return "triángulo"
    elif vertices == 4:
        # Verificar si es cuadrado o rectángulo
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        
        # Calcular el área del rectángulo delimitador
        rect_area = w * h
        contour_area = cv2.contourArea(approx)
        
        # Verificar si el contorno llena bien el rectángulo delimitador
        area_ratio = contour_area / rect_area if rect_area > 0 else 0
        
        # Si el contorno llena al menos el 80% del rectángulo delimitador
        if area_ratio >= 0.8:
            if 0.95 <= aspect_ratio <= 1.05:
                return "cuadrado"
            else:
                return "rectángulo"
        else:
            # Podría ser un rombo o forma cuadrilátera irregular
            return "rombo"
    
    elif vertices >= 5:
        # Verificar si es círculo
        (x, y), radius = cv2.minEnclosingCircle(approx)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Calcular el área del círculo
        circle_area = np.pi * radius * radius
        contour_area = cv2.contourArea(approx)
        
        # Si el área del contorno es similar al área del círculo, es un círculo
        if abs(contour_area - circle_area) / circle_area < 0.3:
            return "círculo"
        else:
            return "rombo"  # Forma compleja
    
    return None

def detect_dominant_color(roi):
    """Detecta el color dominante en una región de interés"""
    try:
        # Convertir a HSV para mejor detección de color
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Definir rangos de color en HSV
        color_ranges = {
            'rojo': [(0, 50, 50), (10, 255, 255)],
            'rojo2': [(170, 50, 50), (180, 255, 255)],
            'verde': [(40, 50, 50), (80, 255, 255)],
            'azul': [(100, 50, 50), (130, 255, 255)],
            'amarillo': [(20, 50, 50), (40, 255, 255)],
            'naranja': [(10, 50, 50), (20, 255, 255)],
            'morado': [(130, 50, 50), (170, 255, 255)],
            'rosa': [(160, 50, 50), (180, 255, 255)],
            'blanco': [(0, 0, 200), (180, 30, 255)],
            'negro': [(0, 0, 0), (180, 255, 50)],
            'gris': [(0, 0, 50), (180, 30, 200)]
        }
        
        max_pixels = 0
        dominant_color = 'desconocido'
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            pixel_count = cv2.countNonZero(mask)
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color_name
        
        return dominant_color
        
    except Exception as e:
        return 'desconocido'

def get_dominant_colors(image, num_colors=5):
    """Obtiene los colores dominantes de toda la imagen"""
    try:
        # Redimensionar imagen para procesamiento más rápido
        small_image = cv2.resize(image, (150, 150))
        
        # Convertir a RGB
        rgb_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
        
        # Aplanar la imagen
        pixels = rgb_image.reshape(-1, 3)
        
        # Contar colores únicos (aproximados)
        color_counts = Counter()
        
        for pixel in pixels:
            # Redondear valores para agrupar colores similares
            rounded_color = tuple((pixel // 32) * 32)
            color_counts[rounded_color] += 1
        
        # Obtener los colores más comunes
        dominant_colors = []
        for color, count in color_counts.most_common(num_colors):
            if count > len(pixels) * 0.01:  # Al menos 1% de la imagen
                # Convertir RGB a nombre de color aproximado
                color_name = rgb_to_color_name(color)
                dominant_colors.append({
                    'color': color_name,
                    'rgb': [int(c) for c in color],  # Convertir a lista de enteros
                    'percentage': round((count / len(pixels)) * 100, 2)
                })
        
        return dominant_colors
        
    except Exception as e:
        return []

def rgb_to_color_name(rgb):
    """Convierte valores RGB a nombres de colores"""
    # Asegurar que rgb sea una tupla o lista
    if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
        r, g, b = rgb[:3]
    else:
        return 'desconocido'
    
    # Definir colores de referencia
    colors = {
        'rojo': (255, 0, 0),
        'verde': (0, 255, 0),
        'azul': (0, 0, 255),
        'amarillo': (255, 255, 0),
        'naranja': (255, 165, 0),
        'morado': (128, 0, 128),
        'rosa': (255, 192, 203),
        'blanco': (255, 255, 255),
        'negro': (0, 0, 0),
        'gris': (128, 128, 128)
    }
    
    min_distance = float('inf')
    closest_color = 'desconocido'
    
    for color_name, color_rgb in colors.items():
        distance = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

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

@app.route('/process_video', methods=['POST'])
def process_video():
    """Endpoint para procesar video y detectar formas/colores"""
    temp_video_path = None
    try:
        # Verificar si hay archivo de video
        if 'video' not in request.files:
            return jsonify({'error': 'No se encontró archivo de video'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        # Verificar extensión del archivo
        allowed_extensions = ['mp4', 'avi', 'mov', 'mkv', 'wmv']
        file_extension = video_file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            return jsonify({'error': f'Formato no soportado. Use: {", ".join(allowed_extensions)}'}), 400
        
        # Guardar archivo temporal
        temp_filename = f"temp_video_{np.random.randint(1000, 9999)}.{file_extension}"
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Asegurar que el directorio existe
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        video_file.save(temp_video_path)
        
        # Procesar video
        results = process_video_frames(temp_video_path)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Error procesando video: {str(e)}'}), 500
        
    finally:
        # Limpiar archivo temporal
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

def process_video_frames(video_path, sample_rate=10):
    """Procesa frames del video para detectar formas y colores"""
    try:
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'error': 'No se pudo abrir el video'}
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Procesar frames muestreados
        frame_count = 0
        processed_frames = 0
        all_results = []
        
        # Resumen global
        global_shapes = Counter()
        global_colors = Counter()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar cada N frames (sample_rate)
            if frame_count % sample_rate == 0:
                # Redimensionar frame para procesamiento más rápido
                small_frame = cv2.resize(frame, (640, 480))
                
                # Detectar formas y colores
                result = detect_shapes_and_colors(small_frame)
                
                if 'error' not in result:
                    result['frame_number'] = frame_count
                    result['timestamp'] = round(frame_count / fps, 2)
                    all_results.append(result)
                    
                    # Acumular estadísticas globales
                    for shape_info in result['shapes']:
                        global_shapes[shape_info['shape']] += 1
                        global_colors[shape_info['color']] += 1
                    
                    processed_frames += 1
            
            frame_count += 1
            
            # Limitar procesamiento para videos muy largos
            if frame_count > 300:  # Máximo 300 frames
                break
        
        cap.release()
        
        # Calcular estadísticas finales
        total_shapes_detected = sum(global_shapes.values())
        
        # Obtener formas más comunes
        most_common_shapes = []
        for shape, count in global_shapes.most_common(5):
            percentage = round((count / total_shapes_detected) * 100, 2) if total_shapes_detected > 0 else 0
            most_common_shapes.append({
                'shape': shape,
                'count': count,
                'percentage': percentage
            })
        
        # Obtener colores más comunes
        most_common_colors = []
        for color, count in global_colors.most_common(5):
            percentage = round((count / total_shapes_detected) * 100, 2) if total_shapes_detected > 0 else 0
            most_common_colors.append({
                'color': color,
                'count': count,
                'percentage': percentage
            })
        
        # Convertir a tipos JSON serializables
        serializable_results = []
        for result in all_results[-10:]:  # Últimos 10 frames procesados
            serializable_shapes = []
            
            # Serializar formas
            for shape in result.get('shapes', []):
                serializable_shape = {
                    'shape': str(shape.get('shape', '')),
                    'area': int(shape.get('area', 0)),
                    'color': str(shape.get('color', '')),
                    'position': {
                        'x': int(shape.get('position', {}).get('x', 0)),
                        'y': int(shape.get('position', {}).get('y', 0)),
                        'width': int(shape.get('position', {}).get('width', 0)),
                        'height': int(shape.get('position', {}).get('height', 0))
                    }
                }
                serializable_shapes.append(serializable_shape)
            
            serializable_result = {
                'frame_number': int(result.get('frame_number', 0)),
                'timestamp': float(result.get('timestamp', 0)),
                'total_shapes': int(result.get('total_shapes', 0)),
                'shapes': serializable_shapes
            }
            serializable_results.append(serializable_result)
        
        return {
            'success': True,
            'video_info': {
                'total_frames': int(total_frames),
                'processed_frames': int(processed_frames),
                'fps': float(round(fps, 2)),
                'duration': float(round(duration, 2)),
                'sample_rate': int(sample_rate)
            },
            'summary': {
                'total_shapes_detected': int(total_shapes_detected),
                'most_common_shapes': most_common_shapes,
                'most_common_colors': most_common_colors,
                'unique_shapes': int(len(global_shapes)),
                'unique_colors': int(len(global_colors))
            },
            'frame_results': serializable_results,
            'all_shapes_found': [str(shape) for shape in global_shapes.keys()],
            'all_colors_found': [str(color) for color in global_colors.keys()]
        }
        
    except Exception as e:
        return {'error': f'Error procesando video: {str(e)}'}


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

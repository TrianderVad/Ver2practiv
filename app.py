from flask import Flask, render_template, request, redirect, url_for, jsonify
from ultralytics import YOLO
import cv2
import os
import json
import time
from datetime import datetime
from collections import defaultdict
from werkzeug.utils import secure_filename
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

model = YOLO("tf.pt").to('cuda') if torch.cuda.is_available() else YOLO("tf.pt")
processing_status = {'progress': 0, 'is_processing': False}

# Создаем папки при запуске
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

@app.template_filter('datetimeformat')
def datetimeformat_filter(ts, format='%d.%m.%Y %H:%M'):
    return datetime.fromtimestamp(ts).strftime(format)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_rename(src_path, dst_dir):
    """Безопасное перемещение файла с генерацией уникального имени"""
    filename = os.path.basename(src_path)
    name, ext = os.path.splitext(filename)
    counter = 1
    
    while True:
        dst_path = os.path.join(dst_dir, filename)
        if not os.path.exists(dst_path):
            break
        filename = f"{name}_{counter}{ext}"
        counter += 1
    
    os.rename(src_path, dst_path)
    return dst_path

def process_video(path, name):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    unique_lights = {}
    light_counter = 1
    threshold_distance = 50
    output_path = os.path.join(f'static/{name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                        (int(cap.get(3)), int(cap.get(4))))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)
        current_frame_lights = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                if not box.id:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                class_name = results[0].names[box.cls.item()]
                current_frame_lights.append((x_center, y_center, class_name, box.id.item()))

        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        updated_lights = {}
        for x, y, cls, track_id in current_frame_lights:
            match = next((k for k, v in unique_lights.items() if v['track_id'] == track_id), None)
            
            if not match:
                match = next((k for k, v in unique_lights.items() if 
                            ((v['last_x'] - x)**2 + (v['last_y'] - y)**2)**0.5 < threshold_distance), None)
            
            if match:
                if cls == 'red':
                    if unique_lights[match]['state'] != 'red':
                        unique_lights[match]['start_stop_time'] = frame_count / fps
                        unique_lights[match]['state'] = 'red'
                    unique_lights[match]['stop_time'] += 1/fps
                else:
                    if unique_lights[match]['state'] == 'red':
                        unique_lights[match]['start_drive_time'] = frame_count / fps
                    unique_lights[match]['state'] = cls
                
                updated_lights[match] = unique_lights[match]
            else:
                new_light = {
                    'track_id': track_id,
                    'last_x': x,
                    'last_y': y,
                    'stop_time': 1/fps if cls == 'red' else 0,
                    'state': cls,
                    'start_stop_time': frame_count/fps if cls == 'red' else None,
                    'start_drive_time': None
                }
                
                if cls == 'red':
                    new_light['start_stop_time'] = frame_count/fps
                else:
                    new_light['start_stop_time'] = None
                    new_light['start_drive_time'] = frame_count/fps if cls in ['green', 'yellow'] else None

                updated_lights[light_counter] = new_light
                light_counter += 1
        
        unique_lights = updated_lights
        frame_count += 1
        processing_status['progress'] = int((frame_count / total_frames)*100)
    
    cap.release()
    out.release()
    return {k: {
        'time': v['stop_time'],
        'x': v['last_x'],
        'y': v['last_y'],
        'start_stop_time': v['start_stop_time'],
        'start_drive_time': v['start_drive_time']
    } for k, v in unique_lights.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Генерируем безопасное имя файла
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Сохраняем во временную папку
        file.save(filepath)
        
        res_file = f"results_{int(time.time())}"
        # Обработка видео
        processing_status['is_processing'] = True
        try:
            stats = process_video(filepath, res_file)
        except Exception as e:
            print(f"Ошибка обработки: {e}")
            return redirect(url_for('index'))
        finally:
            processing_status['is_processing'] = False
        
    
        # Генерируем уникальное имя для результатов
        result_filename = f"{res_file}.json"
        with open(os.path.join('static', result_filename), 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Безопасное перемещение видео
        try:
            video_path = safe_rename(filepath, 'static')
        except Exception as e:
            print(f"Ошибка перемещения файла: {e}")
            return redirect(url_for('index'))
        
        return redirect(url_for('show_results', filename=result_filename))
    
    return redirect(url_for('index'))

@app.route('/results/<filename>')
def show_results(filename):
    with open(os.path.join('static', filename)) as f:
        stop_times = json.load(f)
    stop_times = {int(k): v for k, v in stop_times.items()}
    video_file = filename.replace('.json', '')+'.mp4'
    
    print(video_file, filename)
    return render_template('results.html', 
                         stop_times=stop_times,
                         video_file=video_file, json_file = filename)

@app.route('/history')
def show_history():
    logs = []
    for f in os.listdir('static'):
        if f.startswith('results_') and f.endswith('.json'):
            try:
                # Извлекаем временную метку из имени файла
                timestamp = int(f.split('_')[1].split('.')[0])
                
                # Получаем метаданные видео
                video_file = f.replace('.json', '.mp4')
                
                
                # Собираем данные для отображения
                logs.append({
                    'result_file': f,
                    'video_file': video_file,
                    'timestamp': timestamp,
                    'duration': get_processing_duration(f)
                })
            except Exception as e:
                print(f"Ошибка обработки файла {f}: {e}")
    
    # Сортируем по времени создания
    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template('history.html', logs=logs)

def get_processing_duration(filename):
    """Получение длительности обработки из файла результатов"""
    try:
        with open(os.path.join('static', filename)) as f:
            data = json.load(f)
            return round(sum(item['time'] for item in data.values()), 2)
    except:
        return 0.0

@app.route('/progress')
def get_progress():
    return jsonify(processing_status)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

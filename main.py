import cv2
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import streamlit as st
import tempfile


# Carrega a rede YOLO
net = cv2.dnn.readNet("yolov3-wider_16000.weights", "yolov3-face.cfg")

# Define os nomes das camadas
layer_names = net.getLayerNames()

# Obtém as camadas de saída
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    # Compatibilidade para versões mais antigas do OpenCV
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Configurações gerais
QUANTOS_FRAMES_PARA_SALVAR = 5
PROCESSAR_A_CADA_N_FRAMES = 2

# Inicializa o Streamlit
st.title("Detecção e Rastreamento de Faces (YOLO + DeepSort)")
st.sidebar.header("Configurações")
input_source = st.sidebar.selectbox("Escolha a fonte de vídeo", ["Webcam", "Arquivo de Vídeo"])
uploaded_file = None

if input_source == "Arquivo de Vídeo":
    uploaded_file = st.sidebar.file_uploader("Envie o arquivo de vídeo", type=["mp4", "avi", "mov"])

# Inicializa a captura de vídeo
if input_source == "Webcam":
    video_source = 0
else:
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_source = tfile.name
    else:
        st.warning("Envie um arquivo de vídeo.")
        st.stop()

cap = cv2.VideoCapture(video_source)

# Pasta para salvar as imagens das pessoas
if not os.path.exists('saved_faces'):
    os.makedirs('saved_faces')

# Inicializa o rastreador DeepSort
tracker = DeepSort(max_age=5, embedder_gpu=False)  # GPU desabilitada
seen_ids_data = {}  
already_saved = set()

def boxes_overlap(box1, box2):
    """Verifica se duas caixas se sobrepõem."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2)

def salvar_imagem_nao_mais_trackeada(tracks):
    """Salva imagens de faces que não estão mais sendo rastreadas."""
    for track_id, datas in seen_ids_data.items():
        if track_id in already_saved:
            continue
        if track_id in [track.track_id for track in tracks] and len(datas) < QUANTOS_FRAMES_PARA_SALVAR:
            continue

        max_confidence = 0
        max_confidence_face_chip = None
        for data in datas:
            if data['confidence'] > max_confidence:
                max_confidence = data['confidence']
                max_confidence_face_chip = data['face_chip']

        if max_confidence_face_chip is not None and max_confidence_face_chip.size > 0:
            face_filename = f'saved_faces/face_{track_id}_confidence_{max_confidence:.2f}.jpg'
            cv2.imwrite(face_filename, max_confidence_face_chip)
            already_saved.add(track_id)

frame_count = 0

stframe = st.empty()  # Elemento do Streamlit para exibir o vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        salvar_imagem_nao_mais_trackeada([])
        break

    frame_count += 1
    if frame_count % PROCESSAR_A_CADA_N_FRAMES != 0:
        continue

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            confidence = detection[4]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    indices = np.array(indices)
    filtered_boxes = []
    filtered_confidences = []
    for i in indices.flatten():
        filtered_boxes.append(boxes[i])
        filtered_confidences.append(confidences[i])

    detections = [(box, conf, 0) for box, conf in zip(filtered_boxes, filtered_confidences)]
    tracks = tracker.update_tracks(detections, frame=frame)

    salvar_imagem_nao_mais_trackeada(tracks)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if track_id in already_saved:
            continue

        confidence = None
        for box, conf, _ in detections:
            if boxes_overlap(box, [x1, y1, x2 - x1, y2 - y1]):
                confidence = conf
                break

        if confidence:
            face_chip = frame[y1:y2, x1:x2]
            if track_id not in seen_ids_data:
                seen_ids_data[track_id] = []
            seen_ids_data[track_id].append({'confidence': confidence, 'face_chip': face_chip})

    stframe.image(frame, channels="BGR")

cap.release()


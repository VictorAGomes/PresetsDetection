import cv2
import os
import logging
from ultralytics import YOLO
from sort import Sort
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

PASTA_PRESETS = 'presets_mid'
VIDEO_SOURCE = "video_2min.mp4"
NUM_FEATURES = 1000
GOOD_MATCH_RATIO = 0.80
MIN_GOOD_MATCHES = 10
FRAMES_PARA_CONFIRMACAO = 2
LARGURA_PROCESSAMENTO = 640
ALTURA_PROCESSAMENTO = 480

def detectar_movimento_simples(frame_anterior_cinza, frame_atual_cinza, limiar=0.15):
    if frame_anterior_cinza is None:
        return False
    diff = cv2.absdiff(frame_anterior_cinza, frame_atual_cinza)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    proporcao = cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])
    return proporcao > limiar

def analisar_presets_referencia(pasta_presets):
    orb = cv2.ORB_create(nfeatures=NUM_FEATURES)
    descritores_presets = {}
    for nome_arquivo in os.listdir(pasta_presets):
        caminho_arquivo = os.path.join(pasta_presets, nome_arquivo)
        nome_preset = os.path.splitext(nome_arquivo)[0]
        img = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, descriptors = orb.detectAndCompute(img, None)
            if descriptors is not None:
                descritores_presets[nome_preset] = descriptors
    return orb, descritores_presets

def identificar_preset_features(frame, orb, descritores_presets):
    _, descriptors_frame = orb.detectAndCompute(frame, None)
    if descriptors_frame is None:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    melhor_preset, max_matches = None, 0
    for nome_preset, descriptors_preset in descritores_presets.items():
        matches = bf.knnMatch(descriptors_frame, descriptors_preset, k=2)
        good_matches = [m for m, n in matches if m.distance < GOOD_MATCH_RATIO * n.distance] if matches else []
        if len(good_matches) > max_matches:
            melhor_preset, max_matches = nome_preset, len(good_matches)
    if max_matches >= MIN_GOOD_MATCHES:
        return melhor_preset, max_matches
    return None, max_matches

# Carregar modelo YOLO pré-treinado para detecção de carros
modelo_yolo = YOLO('yolov8n.pt')  # Use yolov8n.pt para testes rápidos, ou yolov8s.pt/yolov8m.pt para mais precisão

def main():
    orb_detector, descritores_referencia = analisar_presets_referencia(PASTA_PRESETS)
    if not descritores_referencia:
        logging.error("Nenhuma imagem de referência foi carregada. Encerrando.")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logging.error(f"Erro ao abrir a fonte de vídeo: {VIDEO_SOURCE}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(int(1000 / fps), 10)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('video_processado2.mp4', fourcc, fps/2, (largura, altura))

    estado_camera = "BUSCANDO_PRESET"
    preset_atual = preset_candidato = None
    contador_confirmacao = 0
    frame_anterior_cinza = None
    frame_count = 0
    nomes_presets = [f"preset {i+1}" for i in range(4)]
    tempo_presets = {nome: 0.0 for nome in nomes_presets}
    contagem_carros_presets = {nome: 0 for nome in nomes_presets}
    tracker = Sort()
    carros_ids_presets = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        frame_processamento = cv2.resize(frame, (LARGURA_PROCESSAMENTO, ALTURA_PROCESSAMENTO))
        frame_cinza = cv2.cvtColor(frame_processamento, cv2.COLOR_BGR2GRAY)
        movimento_camera = detectar_movimento_simples(frame_anterior_cinza, frame_cinza)
        frame_anterior_cinza = frame_cinza.copy()
        camera_estavel = not movimento_camera

        if movimento_camera:
            estado_camera = "BUSCANDO_PRESET"
            preset_atual = preset_candidato = None
            contador_confirmacao = 0

        if estado_camera == "BUSCANDO_PRESET" and camera_estavel:
            identificado, n_matches = identificar_preset_features(frame_cinza, orb_detector, descritores_referencia)
            if identificado == preset_candidato:
                contador_confirmacao += 1
            else:
                preset_candidato = identificado
                contador_confirmacao = 1 if identificado else 0
            if contador_confirmacao >= FRAMES_PARA_CONFIRMACAO and identificado:
                preset_atual = preset_candidato
                estado_camera = "PRESET_DEFINIDO"
        if estado_camera == "PRESET_DEFINIDO" and preset_atual:
            nome_preset_exibicao = str(preset_atual)
            # Garante que o preset está no dicionário
            if nome_preset_exibicao not in contagem_carros_presets:
                contagem_carros_presets[nome_preset_exibicao] = 0

            # --- Contagem de carros usando YOLO ---
            resultados = modelo_yolo(frame)
            carros = [det for det in resultados[0].boxes.cls if int(det) == 2]
            num_carros = len(carros)
            contagem_carros_presets[nome_preset_exibicao] += num_carros

            # Adicione no início do seu main()
            if nome_preset_exibicao not in carros_ids_presets:
                carros_ids_presets[nome_preset_exibicao] = set()

            # Pegue as detecções do YOLO (classe 2 = carro)
            boxes = []
            for box, cls in zip(resultados[0].boxes.xyxy.cpu().numpy(), resultados[0].boxes.cls.cpu().numpy()):
                if int(cls) == 2:
                    x1, y1, x2, y2 = box
                    boxes.append([x1, y1, x2, y2, 1.0])  # 1.0 = score

            # Atualize o tracker
            tracks = tracker.update(np.array(boxes)) if boxes else []

            # Adicione IDs únicos ao set do preset atual
            for track in tracks:
                track_id = int(track[4])
                carros_ids_presets[nome_preset_exibicao].add(track_id)

            num_carros = len(carros_ids_presets[nome_preset_exibicao])

            texto_carros = f"Carros únicos detectados: {num_carros}"
            cv2.putText(frame, texto_carros, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        texto_status = (
            f"PRESET ATUAL: {preset_atual} (ESTAVEL)" if estado_camera == "PRESET_DEFINIDO"
            else f"Candidato: {preset_candidato} ({contador_confirmacao}/{FRAMES_PARA_CONFIRMACAO})"
            if preset_candidato else "CAMERA EM MOVIMENTO..."
        )
        cor_status = (0, 255, 0) if estado_camera == "PRESET_DEFINIDO" else (0, 0, 255)
        cv2.putText(frame, texto_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_status, 2)

        # Exibe a contagem acumulada de carros por preset
        if estado_camera == "PRESET_DEFINIDO" and preset_atual:
            nome_preset_exibicao = str(preset_atual)
            total_carros = contagem_carros_presets.get(nome_preset_exibicao, 0)
            texto_contador = f"Total carros {nome_preset_exibicao}: {total_carros}"
            cv2.putText(frame, texto_contador, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Identificador de Preset - Tempo Real", frame)
        out.write(frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
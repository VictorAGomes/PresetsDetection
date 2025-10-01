# Imports e logging
import cv2
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

PASTA_PRESETS = 'presets'
VIDEO_SOURCE = "https://rtsptoweb.fiware.interjato.com.br/stream/20aadc3f-8621-47ec-8c00-550474635331/channel/0/hlsll/live/index.m3u8"
NUM_FEATURES = 2000
GOOD_MATCH_RATIO = 0.75
MIN_GOOD_MATCHES = 10
FRAMES_PARA_CONFIRMACAO = 3
LARGURA_PROCESSAMENTO = 640
ALTURA_PROCESSAMENTO = 480

def detectar_movimento_camera_diff(frame_anterior_cinza, frame_atual_cinza, limiar_area=0.20, limiar_contorno=0.18):
    if frame_anterior_cinza is None:
        return False

    diff = cv2.absdiff(frame_anterior_cinza, frame_atual_cinza)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)  # threshold mais alto

    # Não use dilatação para não aumentar áreas pequenas
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maior_area = 0
    soma_areas = 0
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        soma_areas += area
        if area > maior_area:
            maior_area = area
    total_pixels = thresh.shape[0] * thresh.shape[1]
    porcentagem_maior_contorno = maior_area / total_pixels
    porcentagem_soma_areas = soma_areas / total_pixels

    logging.debug(f"Maior contorno: {porcentagem_maior_contorno:.4f} | Soma áreas: {porcentagem_soma_areas:.4f}")

    # Agora só movimentos realmente grandes vão ser detectados
    return (porcentagem_maior_contorno > limiar_contorno) or (porcentagem_soma_areas > limiar_area)

def analisar_presets_referencia(pasta_presets: str):
    logging.debug("Iniciando análise das imagens de referência...")
    orb = cv2.ORB_create(nfeatures=NUM_FEATURES)
    descritores_presets = {}
    for nome_arquivo in os.listdir(pasta_presets):
        caminho_arquivo = os.path.join(pasta_presets, nome_arquivo)
        nome_preset = os.path.splitext(nome_arquivo)[0]
        try:
            img = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if descriptors is not None:
                descritores_presets[nome_preset] = (keypoints, descriptors)
                logging.debug(f"'{nome_preset}' processado com sucesso.")
            else:
                logging.warning(f"AVISO: Nenhuma característica encontrada em '{nome_preset}'.")
        except Exception as e:
            logging.error(f"ERRO ao processar '{caminho_arquivo}': {e}")
    logging.debug("Análise das imagens de referência concluída.")
    return orb, descritores_presets

def identificar_preset_features(frame, orb, descritores_presets: dict):
    logging.debug("Chamando identificar_preset_features")
    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame, None)
    if descriptors_frame is None:
        logging.debug("Nenhum descritor encontrado no frame atual.")
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_por_preset = {}
    for nome_preset, (_, descriptors_preset) in descritores_presets.items():
        matches = bf.knnMatch(descriptors_frame, descriptors_preset, k=2)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < GOOD_MATCH_RATIO * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass
        matches_por_preset[nome_preset] = len(good_matches)
        logging.debug(f"{nome_preset}: {len(good_matches)} good matches")

    if not matches_por_preset:
        logging.debug("Nenhum preset de referência disponível.")
        return None, 0

    melhor_preset = max(matches_por_preset, key=matches_por_preset.get)
    num_melhores_matches = matches_por_preset[melhor_preset]
    logging.debug(f"Melhor preset: {melhor_preset} com {num_melhores_matches} matches")

    if num_melhores_matches >= MIN_GOOD_MATCHES:
        logging.debug(f"{melhor_preset} atingiu o mínimo de matches.")
        return melhor_preset, num_melhores_matches
    else:
        logging.debug("Nenhum preset atingiu o mínimo de matches.")
        return None, num_melhores_matches

# Função principal
def main():
    logging.debug("Iniciando script principal")
    orb_detector, descritores_referencia = analisar_presets_referencia(PASTA_PRESETS)
    if not descritores_referencia:
        logging.error("Nenhuma imagem de referência foi carregada. Encerrando.")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logging.error(f"Erro ao abrir a fonte de vídeo: {VIDEO_SOURCE}")
        return

    estado_camera = "BUSCANDO_PRESET"
    preset_atual = None
    preset_candidato = None
    contador_confirmacao = 0
    frame_anterior_cinza = None
    estado_anterior = None
    camera_estavel = False

    logging.info("Iniciando captura de vídeo. Pressione 'q' para sair.")
    logging.info("Pressione 'm' para simular o movimento da câmera e voltar ao modo de busca.")

    frame_count = 0
    while True:
        logging.debug("Novo loop do frame")
        ret, frame = cap.read()
        if not ret:
            logging.info("Fim do vídeo ou erro de captura.")
            break
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        frame_processamento = cv2.resize(frame, (LARGURA_PROCESSAMENTO, ALTURA_PROCESSAMENTO))
        frame_cinza = cv2.cvtColor(frame_processamento, cv2.COLOR_BGR2GRAY)

        movimento_camera = detectar_movimento_camera_diff(frame_anterior_cinza, frame_cinza)
        frame_anterior_cinza = frame_cinza.copy()

        if movimento_camera:
            if estado_camera != "BUSCANDO_PRESET":
                logging.info("Movimento da câmera detectado. Buscando novo preset...")
                estado_camera = "BUSCANDO_PRESET"
                preset_atual = None
                preset_candidato = None
                contador_confirmacao = 0
            camera_estavel = False
        else:
            camera_estavel = True

        if estado_camera != estado_anterior:
            logging.info(f"Estado mudou de {estado_anterior} para {estado_camera}")
            estado_anterior = estado_camera

        # Só identifica preset se a câmera estiver estável (sem movimento)
        if estado_camera == "BUSCANDO_PRESET" and camera_estavel:
            logging.debug("Estado: BUSCANDO_PRESET (câmera estável)")
            texto_status = "BUSCANDO PRESET..."
            cor_status = (0, 0, 255)
            identificado, n_matches = identificar_preset_features(frame_cinza, orb_detector, descritores_referencia)
            logging.debug(f"identificador: {identificado}, n_matches: {n_matches}")

            if identificado:
                if identificado == preset_candidato:
                    contador_confirmacao += 1
                else:
                    preset_candidato = identificado
                    contador_confirmacao = 1

                texto_status = f"Candidato: {preset_candidato} ({contador_confirmacao}/{FRAMES_PARA_CONFIRMACAO})"
                logging.debug(f"Candidato atual: {preset_candidato}, contador: {contador_confirmacao}")

                if contador_confirmacao >= FRAMES_PARA_CONFIRMACAO:
                    preset_atual = preset_candidato
                    estado_camera = "PRESET_DEFINIDO"
                    logging.info(f"--- PRESET CONFIRMADO: {preset_atual} ---")
            else:
                preset_candidato = None
                contador_confirmacao = 0

        elif estado_camera == "PRESET_DEFINIDO":
            logging.debug("Estado: PRESET_DEFINIDO")
            texto_status = f"PRESET ATUAL: {preset_atual} (ESTAVEL)"
            cor_status = (0, 255, 0)

        cv2.putText(frame, texto_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_status, 2)
        cv2.imshow("Identificador de Preset - Tempo Real", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logging.info("Tecla 'q' pressionada, encerrando.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.debug("Script finalizado")

# Execução
if __name__ == "__main__":
    main()
# PresetsDetection

Sistema de detecção automática de presets de câmera em tempo real usando visão computacional e OpenCV.

## Descrição

Este projeto implementa um sistema capaz de identificar automaticamente diferentes presets (posições pré-definidas) de câmeras de segurança através da análise de características visuais das imagens. O sistema detecta movimentos da câmera e identifica quando ela se estabiliza em uma posição conhecida.

## Funcionalidades

- **Detecção de movimento da câmera**: Identifica quando a câmera está se movendo
- **Identificação de presets**: Reconhece posições pré-definidas da câmera usando ORB features
- **Confirmação por frames**: Confirma a identificação através de múltiplos frames consecutivos
- **Processamento em tempo real**: Análise de vídeo ao vivo ou arquivos gravados
- **Interface visual**: Exibe o status atual da detecção na tela
- **Gravação de vídeo processado**: Salva o resultado com anotações

## Estrutura do Projeto

```
PresetsDetection/
├── detector.py            # Script principal de processamento
├── README.md            # Este arquivo
├── presets/             # Imagens de referência dos presets 
├── video_2min.mp4       # Vídeo de exemplo para teste
└── video_processado2.mp4 # Vídeo de saída com anotações
```

## Pré-requisitos

### Instalação das Dependências

```bash
pip install opencv-python numpy
```

### Versões Testadas

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.20+

## Como Usar

### 1. Preparação das Imagens de Referência

1. Crie capturas das diferentes posições da câmera (presets)
2. Salve as imagens na parta presets
3. Nomeie os arquivos de forma descritiva (ex: `preset1.png`, `entrada_principal.jpg`)

### 2. Configuração do Vídeo de Entrada

Coloque seu arquivo de vídeo na pasta do projeto ou altere a variável `VIDEO_SOURCE` no código:

```python
VIDEO_SOURCE = "video_2min.mp4"  
```

### 3. Executar o Sistema

```bash
python detector.py
```

### 4. Saída

O sistema gerará:
- Janela de visualização em tempo real
- Arquivo `video_processado2.mp4` com as anotações
- Log das operações no terminal

## Parâmetros Configuráveis

| Parâmetro | Descrição | Valor Padrão | Recomendações |
|-----------|-----------|--------------|---------------|
| `PASTA_PRESETS` | Pasta com imagens de referência | `'presets'` | Ajuste conforme sua estrutura |
| `VIDEO_SOURCE` | Arquivo de vídeo de entrada | `"video_2min.mp4"` | Caminho para seu vídeo |
| `NUM_FEATURES` | Número de features ORB | `1000` | 500-2000 dependendo da complexidade |
| `GOOD_MATCH_RATIO` | Razão para filtrar matches | `0.80` | 0.75-0.85 para maior/menor precisão |
| `MIN_GOOD_MATCHES` | Mínimo de matches válidos | `10` | 5-20 dependendo da qualidade |
| `FRAMES_PARA_CONFIRMACAO` | Frames para confirmar preset | `2` | 2-5 para maior estabilidade |
| `LARGURA_PROCESSAMENTO` | Largura para processamento | `640` | Menor = mais rápido |
| `ALTURA_PROCESSAMENTO` | Altura para processamento | `480` | Menor = mais rápido |

### Exemplo de Personalização

```python
# Para câmeras de alta resolução
NUM_FEATURES = 2000
MIN_GOOD_MATCHES = 15

# Para processamento mais rápido
LARGURA_PROCESSAMENTO = 320
ALTURA_PROCESSAMENTO = 240

# Para maior precisão
FRAMES_PARA_CONFIRMACAO = 3
GOOD_MATCH_RATIO = 0.75
```

## Controles

- **'q'**: Sair da aplicação e finalizar processamento
- **ESC**: Alternativamente, fechar a janela

## Estados do Sistema

### 1. BUSCANDO_PRESET
- **Cor**: Vermelho
- **Situação**: Câmera em movimento ou procurando preset
- **Ação**: Analisando frames para identificar posição estável

### 2. PRESET_DEFINIDO
- **Cor**: Verde
- **Situação**: Preset identificado e confirmado
- **Ação**: Monitorando para detectar próximo movimento

## Como Funciona

### Fluxo do Algoritmo

1. **Inicialização**
   - Carrega imagens de referência dos presets
   - Extrai features ORB de cada imagem
   - Configura captura de vídeo

2. **Processamento por Frame**
   - Redimensiona frame para otimizar performance
   - Converte para escala de cinza
   - Detecta movimento comparando com frame anterior

3. **Detecção de Movimento**
   - Calcula diferença absoluta entre frames
   - Aplica threshold binário
   - Determina proporção de pixels alterados

4. **Identificação de Preset**
   - Quando câmera está estável, extrai features do frame atual
   - Compara com features dos presets de referência
   - Aplica filtro de Lowe para matches de qualidade

5. **Confirmação**
   - Confirma identificação através de frames consecutivos
   - Atualiza estado apenas após confirmação

## Algoritmos Utilizados

### ORB (Oriented FAST and Rotated BRIEF)
- **Função**: Detecção e descrição de features
- **Vantagem**: Rápido e robusto a rotações
- **Uso**: Identificação de características únicas dos presets

### Brute Force Matcher
- **Função**: Correspondência de features
- **Método**: k-NN com k=2 para filtro de Lowe
- **Uso**: Encontrar matches entre frame atual e referências

### Detecção de Movimento por Diferença
- **Função**: Identificar movimento da câmera
- **Método**: Diferença absoluta + threshold
- **Uso**: Determinar quando analisar presets

## Troubleshooting

### Problema: Nenhum preset detectado
**Soluções:**
- Reduza `MIN_GOOD_MATCHES`
- Aumente `GOOD_MATCH_RATIO`
- Verifique qualidade das imagens de referência

### Problema: Muitas detecções falsas
**Soluções:**
- Aumente `MIN_GOOD_MATCHES`
- Reduza `GOOD_MATCH_RATIO`
- Aumente `FRAMES_PARA_CONFIRMACAO`

### Problema: Sistema muito lento
**Soluções:**
- Reduza `NUM_FEATURES`
- Diminua resolução de processamento
- Use vídeo com FPS menor

### Problema: Movimento não detectado
**Soluções:**
- Ajuste threshold na função `detectar_movimento_simples`
- Reduza o limiar de movimento (padrão: 0.15)

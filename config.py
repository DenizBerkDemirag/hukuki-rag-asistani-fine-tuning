"""
Türk Hukuku Asistanı - Merkezi Konfigürasyon Dosyası
Tüm model, eğitim ve yol ayarları burada tanımlanır.
"""

import os

# ─── Yollar ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ham veri ve data/ klasörü
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train_data.json")

# İşlenmiş eğitim verileri (messages formatı)
PROCESSED_TRAIN_FILE = os.path.join(DATA_DIR, "train_messages.json")
PROCESSED_VAL_FILE = os.path.join(DATA_DIR, "val_messages.json")
PROCESSED_TEST_FILE = os.path.join(DATA_DIR, "test_messages.json")

# Mevcut Dataset/ klasörü yolları
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
TRAIN_FILE = os.path.join(DATASET_DIR, "train.json")
TEST_FILE = os.path.join(DATASET_DIR, "test.json")
MAIN_SET_FILE = os.path.join(DATASET_DIR, "main_set.json")

# Mevzuat belgeleri
MEVZUAT_DIR = os.path.join(DATA_DIR, "mevzuat")

# Eğitilmiş model çıktı dizini
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "hukuk_lora")
STATUS_FILE = os.path.join(OUTPUT_DIR, "training_status.json")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "training_history.json")

# ─── Model Ayarları ───────────────────────────────────────────────────────────

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ─── QLoRA Parametreleri ──────────────────────────────────────────────────────

LORA_R = 16                 # LoRA rank
LORA_ALPHA = 32             # LoRA alpha
LORA_DROPOUT = 0.05         # LoRA dropout oranı
LORA_TARGET_MODULES = [     # LoRA uygulanan katmanlar (q_proj ve v_proj ile 3.6M parametre; MLP yükünden arındırıldı)
    "q_proj", "v_proj",
]

# BitsAndBytes 4-bit kuantizasyon
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_USE_DOUBLE_QUANT = True

# ─── Eğitim Hiperparametreleri ────────────────────────────────────────────────

TRAIN_EPOCHS = 1                # 11.884 soru için 1 epoch tam öğrenme sağlar ve süreyi yarıya indirir
TRAIN_BATCH_SIZE = 1            # 4 GB VRAM (RTX 3050 Ti) için optimize edildi
GRADIENT_ACCUMULATION_STEPS = 4 # Efektif batch size = 1 * 4 = 4 (Adım başına bekleme süresini yarıya indirir)
LEARNING_RATE = 0.0002
MAX_SEQ_LENGTH = 256            # Veri analizi: %99 soru-cevap 256 token altındadır; matris çarpımını 2-4x hızlandırır
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 20
EVAL_STRATEGY = "no"            # Eğitim sırasında 1485 doğrulamayı bekletmez; süreyi devasa hızlandırır
EVAL_STEPS = None
SAVE_STRATEGY = "no"            # Yalnızca eğitim bitiminde son LoRA adaptörünü kaydeder
SAVE_TOTAL_LIMIT = 1

# ─── Çıkarım (Inference) Ayarları ────────────────────────────────────────────

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.15

# ─── Sistem İstemi ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Sen Türk hukuku alanında uzmanlaşmış bir yapay zekâ asistanısın. "
    "Kullanıcının hukuki sorularına Türkiye Cumhuriyeti Anayasası ve yürürlükteki "
    "kanunlar çerçevesinde doğru, anlaşılır ve gerekçeli yanıtlar ver. "
    "Yanıtlarında ilgili kanun maddelerini belirt."
)

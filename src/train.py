"""
Türk Hukuku Asistanı - QLoRA (4-bit) + SFT Model Eğitimi Modülü.

Temel Model: Qwen/Qwen2.5-3B-Instruct
Eğitim Yöntemi: QLoRA (NF4 4-bit) + Supervised Fine-Tuning (SFT)
Kütüphaneler: PyTorch, Transformers, PEFT, TRL, bitsandbytes, Datasets

Özellikler:
1. CUDA & GPU bellek uygunluk kontrolü (CPU eğitimi kesinlikle engellenir).
2. Eğitim ve doğrulama veri kümesi doğrulaması.
3. 4-bit NF4 kuantizasyonlu Qwen2.5-3B yükleme.
4. LoRA adaptör yapılandırması (yalnızca LoRA adaptörü kaydedilir).
5. Eğitim ve doğrulama kayıplarını adım adım takip etme.
6. Canlı durum (training_status.json) ve geçmiş (training_history.json) kaydı.
7. Çift çalıştırmayı engelleyen kilit mekanizması (process lock).
"""

import os
import sys
import json
import time
import psutil
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Proje kökünü sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows konsolunda UTF-8 desteği
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import config
from src.data_utils import load_json, prepare_training_data, process_and_split_pipeline


# ─── 1. GPU ve Sistem Kontrolleri ─────────────────────────────────────────────

def get_gpu_info() -> Dict[str, Any]:
    """
    GPU ve CUDA durumunu detaylı olarak sorgular.
    VRAM kapasitesi ve eğitim uygunluğunu döner.
    """
    info = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "Bulunamadı",
        "vram_total_gb": 0.0,
        "vram_allocated_gb": 0.0,
        "vram_free_gb": 0.0,
        "torch_version": "Yüklü değil",
        "is_suitable": False,
        "message": ""
    }

    try:
        import torch
        info["torch_version"] = torch.__version__

        if not torch.cuda.is_available():
            info["message"] = (
                "CUDA destekli GPU bulunamadı. QLoRA eğitimi için NVIDIA GPU gereklidir. "
                "Büyük dil modelleri CPU üzerinde eğitilemez."
            )
            return info

        info["cuda_available"] = True
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)

        # VRAM sorgulama
        total_mem = torch.cuda.get_device_properties(0).total_memory
        allocated_mem = torch.cuda.memory_allocated(0)
        reserved_mem = torch.cuda.memory_reserved(0)

        info["vram_total_gb"] = round(total_mem / (1024 ** 3), 2)
        info["vram_allocated_gb"] = round(allocated_mem / (1024 ** 3), 2)
        info["vram_free_gb"] = round((total_mem - reserved_mem) / (1024 ** 3), 2)

        # 4-bit 3B model için minimum ~3.2 GB VRAM gerekir
        if info["vram_total_gb"] >= 3.5:
            info["is_suitable"] = True
            info["message"] = f"GPU uygun: {info['device_name']} ({info['vram_total_gb']} GB VRAM)"
        else:
            info["is_suitable"] = False
            info["message"] = (
                f"GPU VRAM yetersiz ({info['vram_total_gb']} GB). "
                f"Qwen2.5-3B 4-bit eğitimi için en az 3.5 GB VRAM önerilir."
            )

    except ImportError:
        info["message"] = "PyTorch kütüphanesi kurulu değil."
    except Exception as e:
        info["message"] = f"GPU sorgulama hatası: {str(e)}"

    return info


def check_gpu():
    """CLI ve script çalıştırmaları için zorunlu GPU kontrolü."""
    gpu = get_gpu_info()
    if not gpu["cuda_available"]:
        print("=" * 60)
        print("❌ CUDA destekli GPU bulunamadı!")
        print("=" * 60)
        print("QLoRA eğitimi için NVIDIA GPU gereklidir.")
        print("CPU üzerinde büyük model eğitimi başlatılamaz.")
        print(f"  PyTorch sürümü : {gpu['torch_version']}")
        print(f"  CUDA mevcut    : {gpu['cuda_available']}")
        print("=" * 60)
        sys.exit(1)

    print(f"✅ GPU bulundu: {gpu['device_name']} ({gpu['vram_total_gb']} GB VRAM)")
    return True


# ─── 2. Durum ve Kilit (Lock) Yönetimi ─────────────────────────────────────────

def get_training_status() -> Dict[str, Any]:
    """Mevcut eğitim durumunu STATUS_FILE üzerinden okur."""
    if not os.path.exists(config.STATUS_FILE):
        return {
            "status": "idle",
            "message": "Henüz eğitim başlatılmadı.",
            "current_step": 0,
            "total_steps": 0,
            "progress_percent": 0.0,
            "train_loss": None,
            "eval_loss": None,
            "history": []
        }

    try:
        with open(config.STATUS_FILE, "r", encoding="utf-8") as f:
            status = json.load(f)

        # Eğer status running görünüyorsa PID'nin gerçekten aktif olup olmadığını doğrula
        if status.get("status") == "running":
            pid = status.get("pid")
            if pid and not psutil.pid_exists(pid):
                status["status"] = "failed"
                status["message"] = "Eğitim süreci beklenmedik şekilde kapandı."
                update_training_status(status)

        return status
    except Exception:
        return {"status": "idle", "message": "Durum okunamadı."}


def update_training_status(data: Dict[str, Any]):
    """Eğitim durumunu STATUS_FILE dosyasına atomik olarak yazar."""
    os.makedirs(os.path.dirname(config.STATUS_FILE), exist_ok=True)
    data["last_update"] = datetime.now().isoformat()
    temp_file = config.STATUS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # Windows'ta üzerine yazma için replace
    if os.path.exists(config.STATUS_FILE):
        os.remove(config.STATUS_FILE)
    os.rename(temp_file, config.STATUS_FILE)


def is_training_running() -> bool:
    """Devam eden aktif bir eğitim olup olmadığını kontrol eder."""
    status = get_training_status()
    if status.get("status") == "running":
        pid = status.get("pid")
        if pid and psutil.pid_exists(pid):
            return True
    return False


def stop_training_process() -> bool:
    """Çalışan eğitim sürecini sonlandırır."""
    status = get_training_status()
    if status.get("status") == "running" and status.get("pid"):
        pid = status.get("pid")
        try:
            p = psutil.Process(pid)
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        status["status"] = "stopped"
        status["message"] = "Eğitim kullanıcı tarafından durduruldu."
        update_training_status(status)
        return True
    return False


# ─── 3. TRL / Transformers Callback (İlerleme Takibi) ─────────────────────────

def create_training_callback():
    """Eğitim adımlarını ve kayıplarını yakalayan callback sınıfını üretir."""
    from transformers import TrainerCallback

    class LiveProgressCallback(TrainerCallback):
        def __init__(self, history_file: str, status_file: str):
            self.history_file = history_file
            self.status_file = status_file
            self.logs_history = []
            self.start_time = time.time()

        def on_train_begin(self, args, state, control, **kwargs):
            update_training_status({
                "status": "running",
                "pid": os.getpid(),
                "start_time": datetime.now().isoformat(),
                "current_step": 0,
                "total_steps": state.max_steps,
                "current_epoch": 0.0,
                "total_epochs": args.num_train_epochs,
                "progress_percent": 0.0,
                "train_loss": None,
                "eval_loss": None,
                "learning_rate": args.learning_rate,
                "message": "Eğitim başladı, model ağırlıkları işleniyor..."
            })

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return

            train_loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            learning_rate = logs.get("learning_rate")
            epoch = logs.get("epoch", state.epoch)
            step = state.global_step
            total_steps = state.max_steps if state.max_steps > 0 else 1

            pct = round((step / total_steps) * 100, 1)

            entry = {
                "step": step,
                "epoch": round(epoch, 3) if epoch else 0,
                "train_loss": round(train_loss, 4) if train_loss is not None else None,
                "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
                "learning_rate": learning_rate,
                "timestamp": datetime.now().isoformat()
            }
            self.logs_history.append(entry)

            # Geçmişi kaydet
            try:
                with open(self.history_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": config.BASE_MODEL_NAME,
                        "history": self.logs_history,
                        "last_update": datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # Durumu güncelle
            update_training_status({
                "status": "running",
                "pid": os.getpid(),
                "current_step": step,
                "total_steps": total_steps,
                "current_epoch": round(epoch, 2) if epoch else 0,
                "total_epochs": args.num_train_epochs,
                "progress_percent": pct,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "learning_rate": learning_rate,
                "message": f"Eğitim devam ediyor (Adım: {step}/{total_steps} - %{pct})"
            })

        def on_train_end(self, args, state, control, **kwargs):
            duration_sec = round(time.time() - self.start_time, 1)
            update_training_status({
                "status": "completed",
                "pid": None,
                "completed_at": datetime.now().isoformat(),
                "current_step": state.global_step,
                "total_steps": state.max_steps,
                "progress_percent": 100.0,
                "duration_seconds": duration_sec,
                "message": f"✅ Model eğitimi başarıyla tamamlandı! (Süre: {duration_sec} sn)"
            })

    return LiveProgressCallback(config.HISTORY_FILE, config.STATUS_FILE)


# ─── 4. Veri Doğrulama ve Hazırlama ───────────────────────────────────────────

def validate_and_prepare_datasets(sample_size: Optional[int] = None) -> Tuple[Any, Any]:
    """
    Eğitim ve doğrulama verilerini kontrol eder.
    sample_size belirtilirse eğitimi hızlandırmak için belirtilen adette örnek alır.
    """
    from datasets import Dataset

    # Dataset/ klasöründeki hazır ayrılmış train ve test dosyaları
    train_source = None
    val_source = None

    if os.path.exists(config.TRAIN_FILE):
        train_source = config.TRAIN_FILE
        val_source = config.TEST_FILE if os.path.exists(config.TEST_FILE) else None
        print(f"Dataset/ klasöründeki veriler kullanılıyor: {train_source}")
        if val_source:
            print(f"Doğrulama/Test verisi kullanılıyor: {val_source}")
    else:
        raise FileNotFoundError(
            f"Eğitim verisi bulunamadı! Lütfen {config.TRAIN_FILE} dosyasının var olduğundan emin olun."
        )

    # Verileri oku ve hazırla
    train_raw = load_json(train_source)
    train_prepared = prepare_training_data(train_raw, config.SYSTEM_PROMPT)

    if not train_prepared:
        raise ValueError("Eğitim kümesinde geçerli kayıt bulunamadı.")

    # Hızlı eğitim için örnek kümesi seçimi
    if sample_size and 0 < sample_size < len(train_prepared):
        print(f"⚡ Hızlı eğitim modu: Toplam {len(train_prepared)} örnekten ilk {sample_size} örnek seçildi.")
        train_prepared = train_prepared[:sample_size]

    val_prepared = []
    if val_source and os.path.exists(val_source):
        val_raw = load_json(val_source)
        val_prepared = prepare_training_data(val_raw, config.SYSTEM_PROMPT)
        # Doğrulama aşamasının eğitimi kilitlemesini önlemek için max 50 örnekle sınırla
        if len(val_prepared) > 50:
            val_prepared = val_prepared[:50]

    # Eğer val_source yoksa train verisinden küçük bir pay ayır
    if not val_prepared and len(train_prepared) > 20:
        val_split_idx = int(len(train_prepared) * 0.95)
        val_prepared = train_prepared[val_split_idx:val_split_idx + 50]
        train_prepared = train_prepared[:val_split_idx]

    train_dataset = Dataset.from_list(train_prepared)
    val_dataset = Dataset.from_list(val_prepared) if val_prepared else None

    print(f"  -> Eğitim örnek sayısı   : {len(train_dataset)}")
    if val_dataset:
        print(f"  -> Doğrulama örnek sayısı: {len(val_dataset)} (Hızlı doğrulama)")

    return train_dataset, val_dataset


# ─── 5. Ana Eğitim Fonksiyonu (QLoRA + SFT) ───────────────────────────────────

def train(overrides: Optional[Dict[str, Any]] = None):
    """
    QLoRA + SFT model eğitimini başlatır.
    İsteğe bağlı hiperparametre override sözlüğü kabul eder.
    """
    # 1. Adım: GPU Kontrolü
    gpu_info = get_gpu_info()
    if not gpu_info["cuda_available"]:
        err_msg = (
            "❌ CUDA destekli GPU bulunamadı! QLoRA eğitimi yalnızca NVIDIA GPU ile çalışır. "
            "CPU üzerinde büyük model eğitimi başlatılamaz."
        )
        print(err_msg)
        update_training_status({"status": "failed", "message": err_msg})
        raise RuntimeError(err_msg)

    # 2. Adım: Çift çalıştırma engeli
    if is_training_running():
        msg = "⚠️ Zaten devam eden bir eğitim süreci mevcut! İkinci bir eğitim başlatılamaz."
        print(msg)
        raise RuntimeError(msg)

    # Parametreleri hazırla
    epochs = overrides.get("epochs", config.TRAIN_EPOCHS) if overrides else config.TRAIN_EPOCHS
    batch_size = overrides.get("batch_size", config.TRAIN_BATCH_SIZE) if overrides else config.TRAIN_BATCH_SIZE
    grad_accum = overrides.get("grad_accum", config.GRADIENT_ACCUMULATION_STEPS) if overrides else config.GRADIENT_ACCUMULATION_STEPS
    lr = overrides.get("learning_rate", config.LEARNING_RATE) if overrides else config.LEARNING_RATE
    lora_r = overrides.get("lora_r", config.LORA_R) if overrides else config.LORA_R
    lora_alpha = overrides.get("lora_alpha", config.LORA_ALPHA) if overrides else config.LORA_ALPHA
    lora_dropout = overrides.get("lora_dropout", config.LORA_DROPOUT) if overrides else config.LORA_DROPOUT
    max_seq_len = overrides.get("max_seq_length", config.MAX_SEQ_LENGTH) if overrides else config.MAX_SEQ_LENGTH

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    update_training_status({
        "status": "initializing",
        "pid": os.getpid(),
        "start_time": datetime.now().isoformat(),
        "model": config.BASE_MODEL_NAME,
        "message": f"Model ve kütüphaneler yükleniyor: {config.BASE_MODEL_NAME} (4-bit NF4)..."
    })

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
        from trl import SFTTrainer

        # Veri kümelerini doğrula ve yükle
        sample_size = overrides.get("sample_size") if overrides else None
        train_dataset, val_dataset = validate_and_prepare_datasets(sample_size=sample_size)

        # 4-bit NF4 Kuantizasyon Yapılandırması
        compute_dtype = getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE, torch.bfloat16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.BNB_USE_DOUBLE_QUANT,
        )

        # Tokenizer
        print(f"Tokenizer yükleniyor: {config.BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL_NAME,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 4-bit Model Yükleme (PyTorch SDPA Flash Attention hızlandırmalı)
        print(f"Model 4-bit olarak yükleniyor (NF4 + SDPA): {config.BASE_MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        # k-bit eğitimi için hazırla (Gradient checkpointing kapatıldı; geri yayılımı 3-4x hızlandırır)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

        # LoRA Adaptör Yapılandırması
        print(f"LoRA adaptörü oluşturuluyor (r={lora_r}, alpha={lora_alpha})...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.LORA_TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Eğitim Argümanları (Transformers v5 ve TRL 1.x uyumlu)
        eval_strategy = config.EVAL_STRATEGY
        total_train_steps = max(1, (len(train_dataset) // (batch_size * grad_accum)) * epochs)
        warmup_steps = max(10, int(total_train_steps * getattr(config, "WARMUP_RATIO", 0.03)))

        import inspect
        try:
            from trl import SFTConfig
            sft_config_params = inspect.signature(SFTConfig.__init__).parameters
            config_kwargs = {
                "output_dir": config.OUTPUT_DIR,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "learning_rate": lr,
                "weight_decay": config.WEIGHT_DECAY,
                "lr_scheduler_type": config.LR_SCHEDULER,
                "logging_steps": config.LOGGING_STEPS,
                "eval_strategy": eval_strategy,
                "eval_steps": None,
                "save_strategy": config.SAVE_STRATEGY,
                "save_total_limit": config.SAVE_TOTAL_LIMIT,
                "bf16": True,
                "optim": config.OPTIMIZER,
                "gradient_checkpointing": False,
                "report_to": "none",
            }
            if "warmup_steps" in sft_config_params:
                config_kwargs["warmup_steps"] = warmup_steps
            elif "warmup_ratio" in sft_config_params:
                config_kwargs["warmup_ratio"] = config.WARMUP_RATIO

            if "max_length" in sft_config_params:
                config_kwargs["max_length"] = max_seq_len
            elif "max_seq_length" in sft_config_params:
                config_kwargs["max_seq_length"] = max_seq_len

            if "dataset_text_field" in sft_config_params:
                config_kwargs["dataset_text_field"] = "text"

            training_args = SFTConfig(**config_kwargs)
        except Exception:
            from transformers import TrainingArguments
            t_params = inspect.signature(TrainingArguments.__init__).parameters
            config_kwargs = {
                "output_dir": config.OUTPUT_DIR,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "learning_rate": lr,
                "weight_decay": config.WEIGHT_DECAY,
                "lr_scheduler_type": config.LR_SCHEDULER,
                "logging_steps": config.LOGGING_STEPS,
                "eval_strategy": eval_strategy,
                "eval_steps": config.EVAL_STEPS if val_dataset is not None else None,
                "save_strategy": config.SAVE_STRATEGY,
                "save_total_limit": config.SAVE_TOTAL_LIMIT,
                "bf16": True,
                "optim": config.OPTIMIZER,
                "report_to": "none",
            }
            if "warmup_steps" in t_params:
                config_kwargs["warmup_steps"] = warmup_steps
            elif "warmup_ratio" in t_params:
                config_kwargs["warmup_ratio"] = config.WARMUP_RATIO

            training_args = TrainingArguments(**config_kwargs)

        # Callback ve SFTTrainer
        progress_cb = create_training_callback()

        trainer_kwargs = {
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "args": training_args,
            "callbacks": [progress_cb],
        }

        sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in sft_trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in sft_trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer

        if "dataset_text_field" in sft_trainer_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sft_trainer_params:
            trainer_kwargs["max_seq_length"] = max_seq_len

        trainer = SFTTrainer(**trainer_kwargs)

        print("🚀 Eğitim başlıyor...")
        trainer.train()

        # 8. Yalnızca LoRA Adaptörünü Kaydet (Tüm modeli değil!)
        print(f"LoRA adaptörü kaydediliyor: {config.OUTPUT_DIR}")
        trainer.model.save_pretrained(config.OUTPUT_DIR)
        tokenizer.save_pretrained(config.OUTPUT_DIR)

        print("✅ QLoRA eğitimi başarıyla tamamlandı!")
        return True

    except Exception as e:
        error_msg = f"Eğitim hatası: {str(e)}"
        if "out of memory" in str(e).lower() or "cuda oom" in str(e).lower():
            error_msg = (
                "❌ CUDA Out of Memory (GPU Bellek Yetersizliği) Hatası!\n"
                "Önerilen Çözümler:\n"
                "1) Batch size değerini 1 olarak tutun.\n"
                "2) Gradient Accumulation değerini artırın (örn. 8 veya 16).\n"
                "3) Max Sequence Length değerini düşürün (örn. 256 veya 384)."
            )
        print(error_msg)
        update_training_status({
            "status": "failed",
            "pid": None,
            "failed_at": datetime.now().isoformat(),
            "message": error_msg
        })
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Türk Hukuku Asistanı QLoRA Eğitimi")
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS, help="Eğitim epoch sayısı (varsayılan: 1)")
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE, help="Batch size (varsayılan: 1)")
    parser.add_argument("--grad-accum", type=int, default=config.GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation adımı (varsayılan: 4)")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Öğrenme oranı")
    parser.add_argument("--samples", type=int, default=None, help="Hızlı eğitim için kullanılacak örnek sayısı (örn. 1500)")
    parser.add_argument("--max-seq-len", type=int, default=config.MAX_SEQ_LENGTH, help="Maksimum token uzunluğu (varsayılan: 256)")
    args = parser.parse_args()

    train(overrides={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "sample_size": args.samples,
        "max_seq_length": args.max_seq_len,
    })

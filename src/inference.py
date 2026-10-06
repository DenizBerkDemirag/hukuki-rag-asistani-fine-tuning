"""
Türk Hukuku Asistanı - Çıkarım (Inference) ve Sohbet Modülü.

Temel Model: Qwen/Qwen2.5-3B-Instruct
Adaptör: models/hukuk_lora/ (LoRA)

Özellikler:
1. Temel modeli 4-bit NF4 olarak GPU belleğine optimize yükleme.
2. Eğitilmiş LoRA adaptörünü temel modele bağlama.
3. Model önbellekleme (In-memory singleton): Modeli her soru için yeniden yüklemez.
4. Yapılandırılabilir temperature, top_p, max_new_tokens parametreleri.
5. Akışlı (streaming) veya tek seferlik yanıt üretme.
6. Hata yakalama ve bellek yönetimi.
7. Adaptör mevcudiyet kontrolü.
"""

import os
import sys
import threading
from typing import Generator, Dict, List, Optional, Tuple, Any

# Proje kökünü sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ─── Bellekteki Model Önbelleği (In-Memory Singleton) ─────────────────────────

_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_IS_LORA_ACTIVE = False
_MODEL_LOCK = threading.Lock()
_LOAD_ERROR = None


def is_model_ready(adapter_dir: Optional[str] = None) -> bool:
    """
    Eğitilmiş LoRA adaptör dosyalarının (adapter_config.json ve safetensors/bin)
    mevcut olup olmadığını doğrular.
    """
    path = adapter_dir or config.OUTPUT_DIR
    if not os.path.exists(path):
        return False

    cfg_file = os.path.join(path, "adapter_config.json")
    weights_safetensors = os.path.join(path, "adapter_model.safetensors")
    weights_bin = os.path.join(path, "adapter_model.bin")

    has_config = os.path.exists(cfg_file)
    has_weights = os.path.exists(weights_safetensors) or os.path.exists(weights_bin)

    return has_config and has_weights


def get_model_status_info() -> Dict[str, Any]:
    """Modelin yüklenme ve adaptör durumunu döner."""
    global _CACHED_MODEL, _LOAD_ERROR, _IS_LORA_ACTIVE
    ready = is_model_ready()
    loaded = _CACHED_MODEL is not None

    if loaded:
        mode_text = "LoRA Fine-Tuned (Türk Hukuku)" if _IS_LORA_ACTIVE else "Qwen2.5-3B-Instruct (Temel Model)"
    elif ready:
        mode_text = "LoRA Adaptörü Hazır (Belleğe Yüklenmedi)"
    else:
        mode_text = "Temel Model Kullanılabilir"

    return {
        "is_ready": ready,
        "is_loaded": loaded,
        "is_lora_active": _IS_LORA_ACTIVE,
        "mode": mode_text,
        "base_model": config.BASE_MODEL_NAME,
        "adapter_path": config.OUTPUT_DIR,
        "last_error": _LOAD_ERROR
    }


def load_model(
    adapter_path: Optional[str] = None,
    force_reload: bool = False,
    allow_base_fallback: bool = True
) -> Tuple[Any, Any]:
    """
    Temel Qwen2.5 modelini ve eğitilmiş LoRA adaptörünü GPU belleğine uygun biçimde yükler.
    Eğer LoRA adaptörü henüz eğitilmemişse ve allow_base_fallback=True ise temel modeli yükler.
    Modeli bellekte tutar; her çağrıda tekrar diskten okumaz.
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER, _LOAD_ERROR, _IS_LORA_ACTIVE

    with _MODEL_LOCK:
        if not force_reload and _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None:
            return _CACHED_MODEL, _CACHED_TOKENIZER

        target_adapter = adapter_path or config.OUTPUT_DIR
        has_lora = is_model_ready(target_adapter)

        if not has_lora and not allow_base_fallback:
            _LOAD_ERROR = f"Eğitilmiş LoRA adaptörü bulunamadı: {target_adapter}"
            raise FileNotFoundError(
                f"Eğitilmiş model adaptörü henüz mevcut değil!\n"
                f"Hedef yol: {target_adapter}\n"
                f"Lütfen önce terminalden 'python src/train.py' ile eğitimi tamamlayın."
            )

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig
            )

            # 4-bit NF4 kuantizasyon (4 GB VRAM GPU için bellek tasarrufu)
            compute_dtype = getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE, torch.bfloat16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=config.BNB_USE_DOUBLE_QUANT,
            )

            # Tokenizer
            print(f"[Inference] Tokenizer yükleniyor: {config.BASE_MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.BASE_MODEL_NAME,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Temel Model
            print(f"[Inference] Temel model 4-bit yükleniyor: {config.BASE_MODEL_NAME}")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            if has_lora:
                from peft import PeftModel
                print(f"[Inference] LoRA adaptörü bağlanıyor: {target_adapter}")
                model = PeftModel.from_pretrained(base_model, target_adapter)
                model.eval()
                _IS_LORA_ACTIVE = True
                print("✅ Model ve LoRA adaptörü başarıyla belleğe yüklendi!")
            else:
                model = base_model
                model.eval()
                _IS_LORA_ACTIVE = False
                print("ℹ️ LoRA adaptörü bulunamadı, Qwen2.5-3B-Instruct temel modeli doğrudan kullanıma alındı.")

            _CACHED_MODEL = model
            _CACHED_TOKENIZER = tokenizer
            _LOAD_ERROR = None

            return _CACHED_MODEL, _CACHED_TOKENIZER

        except Exception as e:
            _LOAD_ERROR = str(e)
            error_msg = f"Model yüklenirken hata oluştu: {str(e)}"
            if "out of memory" in str(e).lower():
                error_msg = (
                    "❌ GPU Bellek Yetersizliği (CUDA Out of Memory)!\n"
                    "Başka bir uygulama GPU belleğini kullanıyor olabilir. "
                    "Lütfen diğer GPU işlemlerini sonlandırıp tekrar deneyin."
                )
            print(f"[Inference Hata] {error_msg}")
            raise RuntimeError(error_msg) from e


def unload_model():
    """GPU belleğini boşaltmak için bellekteki modeli kaldırır."""
    global _CACHED_MODEL, _CACHED_TOKENIZER, _IS_LORA_ACTIVE
    with _MODEL_LOCK:
        _CACHED_MODEL = None
        _CACHED_TOKENIZER = None
        _IS_LORA_ACTIVE = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ─── Cevap Üretme Fonksiyonları ──────────────────────────────────────────────

def build_prompt_messages(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Sohbet geçmişi ve yeni soruyu Qwen uyumlu messages listesine dönüştürür.
    """
    sys_prompt = system_prompt or config.SYSTEM_PROMPT
    messages = [{"role": "system", "content": sys_prompt}]

    if chat_history:
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

    # Son soruyu ekle (eğer geçmişte zaten son mesaj değilse)
    if not chat_history or chat_history[-1].get("content") != question:
        messages.append({"role": "user", "content": question})

    return messages


def generate_answer(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    repetition_penalty: float = config.REPETITION_PENALTY,
    system_prompt: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Kullanıcının Türkçe hukuki sorusuna eğitilmiş modelle yanıt üretir.
    Parametreler tamamen yapılandırılabilirdir.
    """
    import torch

    # Model ve tokenizer'ı önbellekten al
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    messages = build_prompt_messages(question, chat_history, system_prompt)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(0.01, float(temperature)),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Yalnızca yeni üretilen token'ları ayrıştır
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip()


def generate_stream(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    repetition_penalty: float = config.REPETITION_PENALTY,
    system_prompt: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Generator[str, None, None]:
    """
    Cevabı kelime kelime / parça parça akış (streaming) olarak üretir.
    Streamlit arayüzünde canlı daktilo efekti sağlar.
    """
    from transformers import TextIteratorStreamer

    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    messages = build_prompt_messages(question, chat_history, system_prompt)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=max(0.01, float(temperature)),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        do_sample=True if temperature > 0 else False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Arka plan iş parçacığında üretimi başlat
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

    thread.join()

"""
Türk Hukuku Asistanı — Modern Yapay Zekâ Sohbet Arayüzü (ChatGPT Deneyimi)
Sade, şık, duyarlı (responsive), çoklu sohbet oturumu ve açık/koyu tema destekli profesyonel hukuk asistanı.
"""

import os
import sys
import uuid
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import streamlit as st

# Proje kökünü sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
from src.inference import (
    is_model_ready,
    load_model,
    generate_stream,
    generate_answer,
    get_model_status_info,
    unload_model,
)
from src.train import get_gpu_info

# ─── Sayfa Yapılandırması ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Türk Hukuku Asistanı",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Oturum ve Çoklu Sohbet Yönetimi ──────────────────────────────────────────

def init_session_state():
    """Tüm oturum durumlarını başlatır ve eksik anahtarları tamamlar."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    if "sessions" not in st.session_state:
        # Başlangıçta varsayılan bir boş oturum oluştur
        initial_id = str(uuid.uuid4())[:8]
        st.session_state.sessions = {
            initial_id: {
                "id": initial_id,
                "title": "Yeni Sohbet",
                "created_at": datetime.now(),
                "messages": [],
            }
        }
        st.session_state.current_session_id = initial_id

    if "current_session_id" not in st.session_state or st.session_state.current_session_id not in st.session_state.sessions:
        st.session_state.current_session_id = next(iter(st.session_state.sessions))

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "editing_session_id" not in st.session_state:
        st.session_state.editing_session_id = None


init_session_state()


def get_current_session() -> Dict[str, Any]:
    """Aktif oturum sözlüğünü döner."""
    cur_id = st.session_state.current_session_id
    if cur_id not in st.session_state.sessions:
        return next(iter(st.session_state.sessions.values()))
    return st.session_state.sessions[cur_id]


def create_new_session() -> str:
    """Yeni bir sohbet oturumu oluşturur ve onu aktif yapar."""
    new_id = str(uuid.uuid4())[:8]
    st.session_state.sessions[new_id] = {
        "id": new_id,
        "title": "Yeni Sohbet",
        "created_at": datetime.now(),
        "messages": [],
    }
    st.session_state.current_session_id = new_id
    st.session_state.pending_prompt = None
    return new_id


def delete_session(session_id: str):
    """Belirtilen oturumu siler."""
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]

    if not st.session_state.sessions:
        create_new_session()
    elif st.session_state.current_session_id == session_id:
        st.session_state.current_session_id = next(iter(st.session_state.sessions))


def auto_title_from_prompt(prompt: str) -> str:
    """İlk kullanıcı sorusundan anlamlı ve kısa bir oturum başlığı üretir."""
    clean = prompt.strip().replace("\n", " ")
    if len(clean) > 36:
        return clean[:34] + "..."
    return clean or "Hukuki Soru"


def group_sessions_by_date(sessions: Dict[str, Dict[str, Any]]) -> List[tuple]:
    """Sohbet oturumlarını Bugün, Dün, Son 7 Gün ve Daha Önce olarak gruplar."""
    now = datetime.now()
    today = now.date()
    yesterday = today - timedelta(days=1)
    seven_days_ago = today - timedelta(days=7)

    groups = {
        "Bugün": [],
        "Dün": [],
        "Son 7 Gün": [],
        "Daha Önce": [],
    }

    # Tarihe göre yeniden eskiye sırala
    sorted_sessions = sorted(
        sessions.values(),
        key=lambda s: s.get("created_at", now),
        reverse=True
    )

    for s in sorted_sessions:
        created_date = s.get("created_at", now).date()
        if created_date == today:
            groups["Bugün"].append(s)
        elif created_date == yesterday:
            groups["Dün"].append(s)
        elif created_date >= seven_days_ago:
            groups["Son 7 Gün"].append(s)
        else:
            groups["Daha Önce"].append(s)

    return [(name, sess_list) for name, sess_list in groups.items() if sess_list]


def extract_legal_citations(text: str) -> List[str]:
    """
    Asistan yanıtındaki kanun, madde numaraları ve mevzuat atıflarını regex ile tespit eder.
    Kaynak uydurmaz; yalnızca metinde gerçekten geçen kanun maddelerini bulur.
    """
    patterns = [
        r'(?:4857\s*sayılı\s*)?İş\s*Kanunu(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:Türk Ceza Kanunu|TCK)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:Türk Borçlar Kanunu|TBK)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:Türk Medeni Kanunu|TMK)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:Hukuk Muhakemeleri Kanunu|HMK)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:Ceza Muhakemesi Kanunu|CMK)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:İcra ve İflas Kanunu|İİK)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:Türk Ticaret Kanunu|TTK)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'Anayasa(?:nın)?(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:6098\s*sayılı\s*)?(?:Borçlar\s*Kanunu)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:4721\s*sayılı\s*)?(?:Medeni\s*Kanun)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
        r'(?:5237\s*sayılı\s*)?(?:Ceza\s*Kanunu)(?:\s*(?:maddesi?|m\.)\s*\d+)?',
    ]
    combined = re.compile('|'.join(patterns), re.IGNORECASE)
    matches = combined.findall(text)

    cleaned = []
    seen = set()
    for m in matches:
        item = m.strip()
        lower_key = item.lower()
        if lower_key not in seen and len(item) > 2:
            seen.add(lower_key)
            cleaned.append(item)
    return cleaned


# ─── Dinamik CSS Tasarım Sistemi (Açık & Koyu Tema) ───────────────────────────

def inject_custom_css(theme: str):
    """Seçilen temaya göre ChatGPT benzeri profesyonel CSS üretir."""
    if theme == "light":
        bg_main = "#f9fafb"
        bg_card = "#ffffff"
        bg_sidebar = "#f3f4f6"
        border_color = "#e5e7eb"
        text_primary = "#111827"
        text_secondary = "#4b5563"
        text_muted = "#9ca3af"
        user_bubble_bg = "#f3f4f6"
        user_bubble_border = "#e5e7eb"
        action_btn_bg = "#f3f4f6"
        hero_gradient = "linear-gradient(135deg, #111827 0%, #374151 100%)"
        accent_blue = "#1d4ed8"
        accent_pill = "rgba(29, 78, 216, 0.08)"
        accent_pill_border = "rgba(29, 78, 216, 0.2)"
        accent_pill_text = "#1e40af"
    else:  # dark
        bg_main = "#0d1117"
        bg_card = "#161b22"
        bg_sidebar = "#090d14"
        border_color = "rgba(255, 255, 255, 0.08)"
        text_primary = "#f0f6fc"
        text_secondary = "#8b949e"
        text_muted = "#6e7681"
        user_bubble_bg = "#1f2937"
        user_bubble_border = "rgba(255, 255, 255, 0.1)"
        action_btn_bg = "rgba(255, 255, 255, 0.04)"
        hero_gradient = "linear-gradient(135deg, #ffffff 40%, #cbd5e1 100%)"
        accent_blue = "#2563eb"
        accent_pill = "rgba(220, 38, 38, 0.12)"
        accent_pill_border = "rgba(239, 68, 68, 0.35)"
        accent_pill_text = "#fca5a5"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: {bg_main} !important;
        color: {text_primary} !important;
    }}

    /* Üst menü ve footer gizleme */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background-color: transparent !important;}}

    /* Sidebar Tasarımı */
    section[data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid {border_color} !important;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }}

    /* Ana sohbet alanı ve maksimum genişlik */
    .block-container {{
        max-width: 820px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 7rem !important;
        margin: 0 auto !important;
    }}

    /* Karşılama (Hero) Alanı */
    .hero-container {{
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        margin-bottom: 1.5rem;
    }}

    .hero-badge {{
        display: inline-block;
        background: {accent_pill};
        border: 1px solid {accent_pill_border};
        color: {accent_pill_text};
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.6px;
        margin-bottom: 1.2rem;
        text-transform: uppercase;
    }}

    .hero-title {{
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: {hero_gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }}

    .hero-subtitle {{
        color: {text_secondary};
        font-size: 1.05rem;
        max-width: 580px;
        margin: 0 auto 1.8rem auto;
        line-height: 1.6;
    }}

    .domain-pills {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-bottom: 2rem;
    }}

    .domain-pill {{
        background: {action_btn_bg};
        border: 1px solid {border_color};
        color: {text_secondary};
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
    }}

    /* Soru Kartları Grid */
    .prompt-grid-title {{
        text-align: center;
        color: {text_muted};
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }}

    /* Chat Mesaj Konteynerleri */
    div[data-testid="stChatMessage"] {{
        background: transparent !important;
        padding: 1.25rem 0 !important;
        border-bottom: 1px solid {border_color} !important;
    }}

    /* Kullanıcı Mesaj Balonu Vurgusu */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {{
        background: transparent !important;
    }}

    /* Yanıt Aksiyon Çubuğu */
    .action-bar {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 0.8rem;
        padding-top: 0.5rem;
    }}

    /* Hukuki Kaynak Kutusu */
    .legal-source-box {{
        background: {action_btn_bg};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 1rem;
        font-size: 0.85rem;
    }}
    .legal-source-badge {{
        display: inline-block;
        background: {accent_pill};
        border: 1px solid {accent_pill_border};
        color: {accent_pill_text};
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }}

    /* Sabit Alt Giriş Alanı Bilgilendirmesi */
    .input-disclaimer {{
        text-align: center;
        font-size: 0.76rem;
        color: {text_muted};
        margin-top: 8px;
    }}

    /* Sidebar Oturum Butonları */
    .session-group-header {{
        color: {text_muted};
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin: 1.2rem 0 0.4rem 4px;
    }}

    /* Buton Tasarımları */
    div.stButton > button {{
        border-radius: 10px !important;
        transition: all 0.15s ease-in-out !important;
        font-weight: 500 !important;
    }}

    /* Yeni Sohbet Butonu */
    div.stButton > button[kind="primary"] {{
        background: {accent_blue} !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        opacity: 0.92 !important;
        transform: translateY(-1px);
    }}

    /* Expander ve Kod Alanları */
    div[data-testid="stExpander"] {{
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        background: {bg_card} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_custom_css(st.session_state.theme)

# ─── Sol Kenar Menüsü (Sidebar) ───────────────────────────────────────────────

with st.sidebar:
    # 1. Başlık ve Logo
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown("### ⚖️ Hukuk Asistanı")
        st.caption("Türk Hukuku Yapay Zekâ Modeli")

    # 2. Belirgin 'Yeni Sohbet' Butonu
    if st.button("➕ Yeni Sohbet", type="primary", use_container_width=True):
        create_new_session()
        st.rerun()

    st.markdown("---")

    # 3. Geçmiş Sohbetler (Tarih Gruplarına Göre: Bugün, Dün, Son 7 Gün, Daha Önce)
    st.markdown("<p style='font-size:0.8rem; font-weight:600; margin-bottom:0.2rem;'>💬 SOHBET GEÇMİŞİ</p>", unsafe_allow_html=True)

    date_groups = group_sessions_by_date(st.session_state.sessions)

    for group_name, sess_list in date_groups:
        st.markdown(f"<div class='session-group-header'>{group_name}</div>", unsafe_allow_html=True)
        for s in sess_list:
            s_id = s["id"]
            is_active = (s_id == st.session_state.current_session_id)
            title = s.get("title", "Sohbet")

            # Aktif oturum için sol ikon
            btn_label = f"📌 {title}" if is_active else f"💬 {title}"

            c_btn, c_opt = st.columns([5, 1])
            with c_btn:
                if st.button(
                    btn_label,
                    key=f"sess_btn_{s_id}",
                    use_container_width=True,
                    help=title,
                ):
                    st.session_state.current_session_id = s_id
                    st.session_state.pending_prompt = None
                    st.rerun()

            with c_opt:
                # Oturum yönetimi için küçük seçenek butonu
                with st.popover("⋮", help="Sohbet Seçenekleri"):
                    new_name = st.text_input("Yeniden Adlandır", value=title, key=f"rename_input_{s_id}")
                    if st.button("Kaydet", key=f"save_rename_{s_id}", use_container_width=True):
                        if new_name.strip():
                            st.session_state.sessions[s_id]["title"] = new_name.strip()
                            st.rerun()

                    st.markdown("---")
                    if st.button("🗑️ Sohbeti Sil", key=f"del_sess_{s_id}", use_container_width=True, type="secondary"):
                        delete_session(s_id)
                        st.rerun()

    st.markdown("---")

    # 4. Model ve Sistem Durumu Rozeti
    status_info = get_model_status_info()
    lora_ready = status_info.get("is_ready", False)
    model_loaded = status_info.get("is_loaded", False)
    mode_text = status_info.get("mode", "Qwen2.5-3B")

    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#888; text-transform:uppercase;'>SİSTEM DURUMU</p>", unsafe_allow_html=True)
    if lora_ready:
        st.success("🟢 **Türk Hukuku LoRA:** Aktif")
    else:
        st.info("ℹ️ **Qwen2.5-3B-Instruct:** Aktif\n*(Temel Model)*")

    gpu_info = get_gpu_info()
    if gpu_info.get("cuda_available"):
        gpu_name = gpu_info.get("device_name", "GPU")
        vram_gb = gpu_info.get("vram_total_gb", 0)
        st.caption(f"⚡ {gpu_name} ({vram_gb:.1f} GB VRAM)")

    # 5. Model Ayarları (Minimalist Açılır Menü)
    with st.expander("⚙️ Model Ayarları"):
        chat_temp = st.slider(
            "Yaratıcılık (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=float(config.TEMPERATURE),
            step=0.05,
            help="Düşük değerler kesin ve tutarlı hukuki çıktılar üretir.",
        )
        chat_max_tokens = st.slider(
            "Maksimum Uzunluk (Tokens)",
            min_value=128,
            max_value=2048,
            value=int(config.MAX_NEW_TOKENS),
            step=64,
        )
        if st.button("🧹 VRAM Belleğini Boşalt", use_container_width=True):
            unload_model()
            st.success("Model belleği boşaltıldı.")
            st.rerun()

    st.markdown("---")

    # 6. Alt Bölüm: Tema Değiştirme ve Profil
    theme_col1, theme_col2 = st.columns([1, 1])
    with theme_col1:
        if st.session_state.theme == "dark":
            if st.button("☀️ Açık Tema", use_container_width=True):
                st.session_state.theme = "light"
                st.rerun()
        else:
            if st.button("🌙 Koyu Tema", use_container_width=True):
                st.session_state.theme = "dark"
                st.rerun()

    with theme_col2:
        st.caption("👤 Hukukçu Modu")

    # Sorumluluk reddi
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #888; line-height: 1.4; margin-top: 8px;">
        ⚖️ Bu asistan hukuki tavsiye veya avukatlık hizmeti vermez; bilgilendirme amaçlıdır.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Aktif Oturum ve Mesajlar ─────────────────────────────────────────────────

current_session = get_current_session()
messages = current_session.get("messages", [])


# ─── Karşılama Ekranı (Sohbet Boşken Gösterilir) ──────────────────────────────

if not messages:
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-badge">⚖️ TÜRK HUKUKU ASİSTANI</span>
            <h1 class="hero-title">Türk Hukuku Asistanı</h1>
            <p class="hero-subtitle">
                Hukuki sorularınızı sorun, mevzuatı keşfedin.
            </p>
            <div class="domain-pills">
                <span class="domain-pill">📘 Türk Ceza Kanunu</span>
                <span class="domain-pill">💼 İş Kanunu</span>
                <span class="domain-pill">🏠 Borçlar & Kira</span>
                <span class="domain-pill">📜 Medeni Kanun</span>
                <span class="domain-pill">🏛️ Arabuluculuk</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<p class='prompt-grid-title'>Örnek sorularla hemen başlayın:</p>", unsafe_allow_html=True)

    # Kullanıcının şart koştuğu 4 kritik hukuki soru kartı:
    card_1 = "İş sözleşmesinin haklı nedenle feshi hangi şartlarda mümkündür?"
    card_2 = "Kira artışında yasal sınırlar nelerdir?"
    card_3 = "Türk Borçlar Kanunu'nda zamanaşımı süreleri nelerdir?"
    card_4 = "İşçilik alacaklarında arabuluculuk zorunlu mudur?"

    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"💼 **İş Hukuku**\n\n{card_1}", key="hero_q1", use_container_width=True):
            st.session_state.pending_prompt = card_1
            st.rerun()

        if st.button(f"⏳ **Borçlar Hukuku**\n\n{card_3}", key="hero_q3", use_container_width=True):
            st.session_state.pending_prompt = card_3
            st.rerun()

    with col2:
        if st.button(f"🏠 **Kira & Gayrimenkul**\n\n{card_2}", key="hero_q2", use_container_width=True):
            st.session_state.pending_prompt = card_2
            st.rerun()

        if st.button(f"⚖️ **Dava & Uyuşmazlık**\n\n{card_4}", key="hero_q4", use_container_width=True):
            st.session_state.pending_prompt = card_4
            st.rerun()


# ─── Mesaj Akışı ve Görüntüleme ───────────────────────────────────────────────

for idx, msg in enumerate(messages):
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            # 1. Asistan Yanıt İçeriği (Markdown)
            st.markdown(content)

            # 2. Hukuki Kaynak ve Mevzuat Kartı (Varsa Göster)
            citations = msg.get("sources")
            if citations is None:
                # Geçmiş mesajlarda kaynak kaydedilmediyse dinamik ayıkla
                citations = extract_legal_citations(content)

            if citations:
                badge_html = " ".join([f"<span class='legal-source-badge'>📜 {c}</span>" for c in citations])
                st.markdown(
                    f"""
                    <div class="legal-source-box">
                        <strong>📚 Tespit Edilen Mevzuat ve Kanun Maddeleri:</strong><br>
                        {badge_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3. Profesyonel Hukuki Danışmanlık ve Bilgilendirme Uyarısı
            st.caption(
                "⚠️ *Bu yanıt yapay zekâ tarafından mevzuat ve içtihat bilgisi taranarak üretilmiştir. "
                "Resmi hukuki danışmanlık yerine geçmez. Somut olaylar bakımından bir avukata danışınız.*"
            )

            # 4. Aksiyon Çubuğu (Kopyala, Yeniden Oluştur, Geri Bildirim)
            act_col1, act_col2, act_col3, act_col4 = st.columns([2, 2, 1, 1])

            with act_col1:
                with st.expander("📋 Yanıtı Kopyala", expanded=False):
                    st.code(content, language="text")

            with act_col2:
                # En son asistan yanıtı ise yeniden oluştur butonu sun
                if idx == len(messages) - 1 and len(messages) >= 2:
                    last_user_q = messages[idx - 1].get("content")
                    if st.button("🔄 Yeniden Oluştur", key=f"regen_{idx}", use_container_width=True):
                        # Son kullanıcı ve asistan mesajını kaldırıp tekrar sor
                        current_session["messages"] = messages[:-2]
                        st.session_state.pending_prompt = last_user_q
                        st.rerun()

            with act_col3:
                feedback = msg.get("feedback")
                btn_up_label = "👍" if feedback != "up" else "✅"
                if st.button(btn_up_label, key=f"fb_up_{idx}", help="Yanıt faydalı oldu"):
                    msg["feedback"] = "up"
                    st.toast("Geri bildiriminiz için teşekkürler! 👍")
                    st.rerun()

            with act_col4:
                feedback = msg.get("feedback")
                btn_down_label = "👎" if feedback != "down" else "❌"
                if st.button(btn_down_label, key=f"fb_down_{idx}", help="Yanıt yetersiz veya hatalı"):
                    msg["feedback"] = "down"
                    st.toast("Geri bildiriminiz kaydedildi. İyileştirme için kullanılacaktır. 👎")
                    st.rerun()


# ─── Mesaj Giriş Alanı ve Yanıt Üretimi ───────────────────────────────────────

user_input = st.chat_input("Hukuki sorunuzu buraya yazın (örn: İhbar tazminatı şartları nelerdir?)...")

prompt_to_process = None
if st.session_state.pending_prompt:
    prompt_to_process = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif user_input and user_input.strip():
    prompt_to_process = user_input.strip()

if prompt_to_process:
    # 1. Oturum başlığını ilk sorudan otomatik güncelle
    if not messages or current_session.get("title") == "Yeni Sohbet":
        current_session["title"] = auto_title_from_prompt(prompt_to_process)

    # 2. Kullanıcı mesajını ekle ve anında göster
    user_entry = {
        "role": "user",
        "content": prompt_to_process,
        "timestamp": datetime.now().strftime("%H:%M"),
    }
    current_session["messages"].append(user_entry)

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_process)

    # 3. Asistan yanıtını streaming (daktilo efekti) ile canlı üret
    with st.chat_message("assistant", avatar="⚖️"):
        response_placeholder = st.empty()
        full_response = ""

        with st.spinner("Mevzuat ve içtihatlar taranıyor, gerekçeli yanıt hazırlanıyor..."):
            try:
                # Geçmiş mesajları modele besle (son kullanıcı mesajı hariç)
                history_for_prompt = current_session["messages"][:-1]

                temp_val = chat_temp if "chat_temp" in locals() else config.TEMPERATURE
                tokens_val = chat_max_tokens if "chat_max_tokens" in locals() else config.MAX_NEW_TOKENS

                for chunk in generate_stream(
                    question=prompt_to_process,
                    chat_history=history_for_prompt,
                    temperature=temp_val,
                    max_new_tokens=tokens_val,
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                # Nihai metni yerleştir
                response_placeholder.markdown(full_response)

                # Yanıttan hukuki mevzuat atıflarını tespit et
                detected_sources = extract_legal_citations(full_response)

                if detected_sources:
                    badge_html = " ".join([f"<span class='legal-source-badge'>📜 {c}</span>" for c in detected_sources])
                    st.markdown(
                        f"""
                        <div class="legal-source-box">
                            <strong>📚 Tespit Edilen Mevzuat ve Kanun Maddeleri:</strong><br>
                            {badge_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Yasal uyarı
                st.caption(
                    "⚠️ *Bu yanıt yapay zekâ tarafından mevzuat ve içtihat bilgisi taranarak üretilmiştir. "
                    "Resmi hukuki danışmanlık yerine geçmez. Somut olaylar bakımından bir avukata danışınız.*"
                )

                # Kopyalama seçeneği
                with st.expander("📋 Yanıtı Kopyala", expanded=False):
                    st.code(full_response, language="text")

                # Oturuma asistan yanıtını ekle
                asst_entry = {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "sources": detected_sources,
                    "feedback": None,
                }
                current_session["messages"].append(asst_entry)

            except Exception as e:
                # Teknik hata detayını maskeleyip kullanıcı dostu ve yol gösterici hata mesajı sun
                user_friendly_error = (
                    "❌ **Yanıt üretilirken bir sorunla karşılaşıldı.**\n\n"
                    "Model şu anda belleğe yüklenemedi veya GPU kaynakları meşgul. "
                    "Lütfen sol menüdeki **'VRAM Belleğini Boşalt'** butonuna tıklayıp sorunuzu tekrar deneyiniz."
                )
                st.error(user_friendly_error)
                st.caption(f"Teknik detay: {str(e)}")
                current_session["messages"].append({
                    "role": "assistant",
                    "content": user_friendly_error,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "sources": [],
                    "feedback": None,
                })


# ─── Sabit Alt Uyarı Metni ───────────────────────────────────────────────────

st.markdown(
    '<p class="input-disclaimer">Yapay zekâ tarafından oluşturulan yanıtları önemli hukuki işlemlerden önce doğrulayın.</p>',
    unsafe_allow_html=True,
)
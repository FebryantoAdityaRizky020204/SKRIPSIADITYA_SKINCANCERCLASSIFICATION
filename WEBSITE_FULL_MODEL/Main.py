import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

# =============================================================================
# KONFIGURASI PATH MODEL
# =============================================================================
# 1. EFFICIENTNET B0
EFFNET_GK_PATH  = ''
EFFNET_DIS_PATH = ''

# 2. HYBRID (EFFICIENTNET + VISION MAMBA)
HYBRID_GK_PATH  = ''
HYBRID_DIS_PATH = ''

# 3. VISION MAMBA MURNI
MAMBA_GK_PATH   = '' 
MAMBA_DIS_PATH  = ''

REF_FOLDER = '' # references image folder
GATEKEEPER_THRESHOLD = 0.6

# =============================================================================
# IMPOR MODUL INFERENSI
# =============================================================================
# 1. IMPOR EFFICIENTNET (Selalu Aman)
from inference_effnet import (
    load_model_gatekeeper_effnet, load_model_disease_effnet, 
    predict_gatekeeper_effnet, predict_disease_effnet,
    CLASS_DESCRIPTIONS, DISEASE_DESCRIPTION, CLASS_RISK, CLASS_RISK_COLOR, CLASS_RISK_ICON
)

# 2. IMPOR HYBRID & MAMBA
try:
    from inference_hybrid import (
        load_model_gatekeeper_hybrid, load_model_disease_hybrid,
        predict_gatekeeper_hybrid, predict_disease_hybrid
    )
    from inference_mamba import (
        load_model_gatekeeper_mamba, load_model_disease_mamba,
        predict_gatekeeper_mamba, predict_disease_mamba
    )
    MAMBA_ENV_READY = True
except ImportError as e:
    MAMBA_ENV_READY = False
    MAMBA_ERROR_MSG = str(e)


st.set_page_config(page_title="Klasifikasi Lesi Kulit", page_icon="🔬", layout="wide")

# =============================================================================
# LOAD MODEL CACHING
# =============================================================================
@st.cache_resource(show_spinner="Memuat model EfficientNet-B0...")
def get_effnet_models():
    gk = load_model_gatekeeper_effnet(EFFNET_GK_PATH)
    dis = load_model_disease_effnet(EFFNET_DIS_PATH)
    return gk, dis

@st.cache_resource(show_spinner="Memuat model Hybrid...")
def get_hybrid_models():
    gk = load_model_gatekeeper_hybrid(HYBRID_GK_PATH)
    dis = load_model_disease_hybrid(HYBRID_DIS_PATH)
    return gk, dis

@st.cache_resource(show_spinner="Memuat model Vision Mamba...")
def get_mamba_models():
    gk = load_model_gatekeeper_mamba(MAMBA_GK_PATH)
    dis = load_model_disease_mamba(MAMBA_DIS_PATH)
    return gk, dis

# =============================================================================
# SESSION STATE & CALLBACK
# =============================================================================
if 'selected_img_path' not in st.session_state:
    st.session_state.selected_img_path = None
if 'result' not in st.session_state:
    st.session_state.result = None

# akan dipanggil setiap kali dropdown model diubah
def reset_hasil():
    st.session_state.result = None

# =============================================================================
# HEADER & INFORMASI MODEL
# =============================================================================
st.title("🔬 Sistem Klasifikasi Lesi Kulit")

model_df = pd.DataFrame({
    'MODEL': ['EfficientNet-B0', 'Vision Mamba', 'Hybrid (EfficientNet-B0 + Vision Mamba)'],
    'Accuracy': ["0,8810 ± 0,0087", "0,7874 ± 0,0180", "0,8732 ± 0,0059 "],
    "Precision": ["0,8429 ± 0,0188", "0,6466 ± 0,0275", "0,8341 ± 0,0215"],
    "Recall": ["0,8137 ± 0,0099", "0,7066 ± 0,0096", "0,7914 ± 0,0152"],
    "F1-Score": ["0,8250 ± 0,0116", "0,6699 ± 0,0145", "0,8095 ± 0,0075"]
})

def highlight_except_model_name(row):
    styles = []
    if row['MODEL'] == 'EfficientNet-B0':
        for col in row.index:
            if col == 'MODEL':
                styles.append('')
            else:
                styles.append('background-color: #3874FF; color: white;') 
    else:
        styles = [''] * len(row)
    return styles

model_styled_df = model_df.style.apply(highlight_except_model_name, axis=1)

with st.expander("Model Information", icon="📃"):
    st.write('''
        Model dilatih menggunakan **Dataset HAM10000 - 7 Kelas Penyakit Kulit**, dengan rincian kinerja model:
    ''')
    st.dataframe(model_styled_df, hide_index=True, width="content")
    st.write('''
        Informasi lebih lanjut klik link berikut: [Github](https://github.com/FebryantoAdityaRizky020204/SKRIPSIADITYA_SKINCANCERCLASSIFICATION.git)
    ''')

# =============================================================================
# PEMILIHAN MODEL HORIZONTAL
# =============================================================================
col_teks, col_dropdown = st.columns([1, 2], vertical_alignment="center")

with col_teks:
    st.markdown("**⚙️ Pilih Arsitektur Model:**")

with col_dropdown:
    pilihan_model = st.selectbox(
        "Pilih Arsitektur Model:",
        [
            "EfficientNet-B0", 
            "Vision Mamba",
            "Hybrid (EfficientNet + Vision Mamba)", 
        ],
        label_visibility="collapsed",
        on_change=reset_hasil
    )


st.caption("⚠️ Hasil ini bersifat eksperimental dan hanya untuk keperluan penelitian. **Tidak menggantikan diagnosis medis profesional.**")
st.markdown("---")

# =============================================================================
# PEMUATAN MODEL BERDASARKAN PILIHAN DROPDOWN
# =============================================================================
model_ready = False

if pilihan_model == "EfficientNet-B0":
    try:
        (gk_model, gk_names, gk_mean, gk_std), (dis_model, dis_names, dis_mean, dis_std) = get_effnet_models()
        model_ready = True
    except Exception as e:
        st.error(f"Gagal memuat model EfficientNet: {e}")

elif pilihan_model in ["Hybrid (EfficientNet + Vision Mamba)", "Vision Mamba"]:
    if not MAMBA_ENV_READY:
        st.error(f"⚠️ **Model {pilihan_model} tidak dapat berjalan di environment ini.**\n\nLibrary yang dibutuhkan tidak lengkap (kemungkinan berjalan di server tanpa GPU). Error: `{MAMBA_ERROR_MSG}`")
    else:
        try:
            if pilihan_model == "Hybrid (EfficientNet + Vision Mamba)":
                (gk_model, gk_names, gk_mean, gk_std), (dis_model, dis_names, dis_mean, dis_std) = get_hybrid_models()
            else:
                (gk_model, gk_names, gk_mean, gk_std), (dis_model, dis_names, dis_mean, dis_std) = get_mamba_models()
            model_ready = True
        except Exception as e:
            st.error(f"Gagal memuat model {pilihan_model}: {e}")

# =============================================================================
# LAYOUT UTAMA
# =============================================================================
col1, col2 = st.columns([3, 1])

# Panel Contoh Gambar (Kanan)
with col2:
    st.subheader("📂 Contoh Gambar")
    with st.container(height=620):
        if os.path.exists(REF_FOLDER):
            file_list = sorted([f for f in os.listdir(REF_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
            for file_name in file_list:
                img_path = os.path.join(REF_FOLDER, file_name)
                st.image(Image.open(img_path), use_container_width=True)
                label = os.path.splitext(file_name)[0].upper()
                if st.button(f"▶ {label}", key=file_name, use_container_width=True):
                    st.session_state.selected_img_path = img_path
                    st.session_state.result = None
                    st.rerun()
        else:
            st.warning("Folder contoh tidak ditemukan.")

# Panel Utama (Kiri)
with col1:
    uploaded_file = st.file_uploader("Unggah Gambar Lesi Kulit", type=["jpg", "jpeg", "png"])
    display_image = None
    
    if uploaded_file is not None:
        display_image = Image.open(uploaded_file).convert('RGB')
        st.session_state.selected_img_path = None
        st.session_state.result = None
    elif st.session_state.selected_img_path is not None:
        display_image = Image.open(st.session_state.selected_img_path).convert('RGB')

    if display_image is not None:
        prev_col, _ = st.columns([1, 1])
        with prev_col:
            st.image(display_image, use_container_width=True)

        st.markdown("")

        if st.button("🔍 Mulai Klasifikasi", type="primary", use_container_width=True, disabled=not model_ready):
            with st.spinner("Memproses gambar..."):
                
                # Routing Inferensi Gatekeeper
                if pilihan_model == "EfficientNet-B0":
                    gk_probs, gk_pred_idx = predict_gatekeeper_effnet(gk_model, display_image, gk_mean, gk_std)
                elif pilihan_model == "Hybrid (EfficientNet + Vision Mamba)":
                    gk_probs, gk_pred_idx = predict_gatekeeper_hybrid(gk_model, display_image, gk_mean, gk_std)
                else:
                    gk_probs, gk_pred_idx = predict_gatekeeper_mamba(gk_model, display_image, gk_mean, gk_std)

                prob_kulit = float(gk_probs[1])
                is_skin    = (gk_pred_idx == 1) and (prob_kulit >= GATEKEEPER_THRESHOLD)

                if not is_skin:
                    st.session_state.result = {'is_skin' : False, 'gk_probs': gk_probs}
                else:
                    # Routing Inferensi Penyakit
                    if pilihan_model == "EfficientNet-B0":
                        probs, pred_idx = predict_disease_effnet(dis_model, display_image, dis_mean, dis_std)
                    elif pilihan_model == "Hybrid (EfficientNet + Vision Mamba)":
                        probs, pred_idx = predict_disease_hybrid(dis_model, display_image, dis_mean, dis_std)
                    else:
                        probs, pred_idx = predict_disease_mamba(dis_model, display_image, dis_mean, dis_std)
                        
                    st.session_state.result = {
                        'is_skin': True, 'gk_probs': gk_probs, 
                        'disease_probs': probs, 'disease_pred_idx': pred_idx,
                        'class_names': dis_names
                    }

        # Tampilkan Hasil
        if st.session_state.result is not None:
            result = st.session_state.result
            st.markdown("---")

            if not result['is_skin']:
                prob_bukan, prob_kulit = result['gk_probs']
                
                # Menampilkan pesan error
                st.error("""
                ### 🚫 MAAF, GAMBAR TIDAK DIKENALI
                :small[*Pastikan Gambar Yang Anda Upload Sesuai Dengan Kriteria 7 Penyakit Kulit Yaitu: (Melanocytic nevi (nv), Melanoma (mel), Benign keratosis-like lesions (bkl), Basal cell carcinoma (bcc), Actinic keratoses and intraepithelial carcinoma (akiec), Vascular lesions (vasc), Dermatofibroma (df) )*]
                
                > Sistem mendeteksi bahwa gambar ini berada di luar cakupan data pelatihan pada penelitian ini ***(HAM10000)***, sehingga proses klasifikasi penyakit tidak dapat dilanjutkan.
                """)

                with st.expander("Detail Analisis Gambar", icon="🔍"):
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.metric(label="Di Luar Dataset Pelatihan (Out-of-Distribution)", value=f"{prob_bukan * 100:.2f}%")
                        st.progress(float(prob_bukan))
                    with c2:
                        st.metric(label="Sesuai Dataset Pelatihan (In-Distribution)", value=f"{prob_kulit * 100:.2f}%")
                        st.progress(float(prob_kulit))
            else:
                probs = result['disease_probs']
                pred_idx = result['disease_pred_idx']
                CLASS_NAMES = result['class_names']
                
                pred_class = CLASS_NAMES[pred_idx]
                risk = CLASS_RISK[pred_class]

                st.caption("⚠️ Hasil ini bersifat eksperimental dan hanya untuk keperluan penelitian. **Tidak menggantikan diagnosis medis profesional.**")
                st.subheader("📊 Hasil Klasifikasi")
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediksi Kelas", pred_class.upper())
                m2.metric("Kepercayaan", f"{probs[pred_idx] * 100:.1f}%")
                m3.metric("Tingkat Risiko", f"{CLASS_RISK_ICON[risk]} {risk}")

                color_map = {'Tinggi': 'error', 'Sedang': 'warning', 'Rendah': 'success'}
                getattr(st, color_map[risk])(f"**{pred_class.upper()}** — {CLASS_DESCRIPTIONS[pred_class]} · {DISEASE_DESCRIPTION[pred_class]}")

                sorted_idx   = np.argsort(probs)
                sorted_names = [CLASS_NAMES[i].upper() for i in sorted_idx]
                sorted_probs = [float(probs[i]) * 100 for i in sorted_idx]
                bar_colors   = ['#E53E3E' if CLASS_RISK[CLASS_NAMES[i]] == 'Tinggi' else '#DD6B20' if CLASS_RISK[CLASS_NAMES[i]] == 'Sedang' else '#38A169' for i in sorted_idx]


                with st.expander("Distribusi Probabilitas", icon="📃"):
                    fig = go.Figure(go.Bar(
                        x=sorted_probs, y=sorted_names, orientation='h',
                        marker=dict(color=bar_colors), text=[f"{p:.1f}%" for p in sorted_probs], textposition='outside'
                    ))
                    fig.update_layout(
                        title="Distribusi Probabilitas/Kepercayaan", xaxis_title="Probabilitas (%)", 
                        xaxis=dict(range=[0, 115]), yaxis=dict(showgrid=False), 
                        height=310, margin=dict(l=10, r=70, t=44, b=30)
                    )
                    st.plotly_chart(fig, use_container_width=True)

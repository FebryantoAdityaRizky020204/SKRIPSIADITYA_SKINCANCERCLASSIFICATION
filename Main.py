import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="Sistem Klasifikasi Penyakit Kulit", layout="wide")

# 1. Inisialisasi Session State
# Ini digunakan untuk menyimpan gambar mana yang sedang aktif (dipilih)
if 'selected_img_path' not in st.session_state:
    st.session_state.selected_img_path = None

# Judul Aplikasi
st.title("KLASIFIKASI CITRA MEDIS")
st.markdown("---")

# 2. Mengatur Proporsi Kolom 75:25
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Contoh Gambar")
    ref_folder = "ref_image"

    with st.container(height=500):
        if os.path.exists(ref_folder):
            file_list = [f for f in os.listdir(ref_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            
            for file_name in file_list:
                img_path = os.path.join(ref_folder, file_name)
                img = Image.open(img_path)
                
                # Tampilkan gambar kecil sebagai preview
                st.image(img, use_container_width=True)
                
                # Tombol untuk memilih gambar ini
                if st.button(f"Gunakan {file_name}", key=file_name):
                    st.session_state.selected_img_path = img_path
                    st.rerun()
        else:
            st.error(f"Folder '{ref_folder}' tidak ditemukan.")


with col1:
    # Uploader gambar
    uploaded_file = st.file_uploader("Unggah gambar penyakit...", type=["jpg", "jpeg", "png"])
    
    display_image = None
    source_name = ""

    if uploaded_file is not None:
        display_image = Image.open(uploaded_file)
        source_name = f"File Unggahan: {uploaded_file.name}"
        # Jika user unggah file baru, hapus pilihan dari referensi
        st.session_state.selected_img_path = None 
    elif st.session_state.selected_img_path is not None:
        display_image = Image.open(st.session_state.selected_img_path)
        source_name = f"Menggunakan Contoh: {os.path.basename(st.session_state.selected_img_path)}"

    # Tampilkan Gambar Utama
    if display_image:
        st.write(f"**Preview ({source_name}):**")
        st.image(display_image, use_container_width=True)
        
        # Di sini Anda bisa menambahkan tombol proses klasifikasi
        if st.button("Mulai Klasifikasi", type="primary"):
            st.info("Proses klasifikasi menggunakan EfficientNet-B0 dan Vision Mamba sedang berjalan...")
    else:
        st.info("Silakan unggah gambar di kolom ini atau pilih salah satu contoh gambar dari daftar di sebelah kanan.")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import folium

# Load data from CSV
file_path = "original_table.csv"
df = pd.read_csv(file_path)

# Sidebar
st.sidebar.title("Pengaturan Klaster")
num_clusters = st.sidebar.slider("Jumlah Klaster", 2, 10, 3)
st.sidebar.markdown("---")

# Kolom untuk pengelompokan (2011-2022)
kolom_pengelompokan = [str(tahun) for tahun in range(2011, 2020)]

# Memastikan tidak ada nilai None di dalam kolom_pengelompokan
df[kolom_pengelompokan] = df[kolom_pengelompokan].astype(float)  # Konversi ke tipe data float

# Menentukan informasi kepadatan penduduk berdasarkan klaster
density_info = {
    0: "Tidak Padat",
    1: "Padat",
    2: "Sangat Padat",
    # Tambahkan sesuai jumlah klaster yang dipilih
}

# Elbow Method untuk menentukan jumlah klaster yang optimal
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(df[kolom_pengelompokan])
    inertia_values.append(kmeans.inertia_)

# Menampilkan plot Elbow Method
fig, ax = plt.subplots()
ax.plot(range(1, 11), inertia_values, marker='o')
ax.set_xlabel('Jumlah Klaster')
ax.set_ylabel('Inertia (Within-cluster Sum of Squares)')
st.sidebar.pyplot(fig)

# Konten utama
st.title("Pengelompokan Kepadatan Penduduk")
st.markdown("----")

# Menampilkan informasi Elbow Method
st.sidebar.write("### Informasi Elbow Method:")
st.sidebar.write("Metode Elbow membantu menentukan jumlah klaster optimal dengan melihat titik di mana penurunan inersia tidak lagi signifikan.")
st.sidebar.markdown("---")

# Sidebar untuk memilih tampilan
tampilan_terpilih = st.sidebar.radio("Pilih Tampilan", ["Data Asli", "Tabel Klaster", "Visualisasi Data", "Peta Folium"])

# Klaster dengan KMeans (gunakan jumlah klaster yang telah ditentukan)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df["Cluster"] = kmeans.fit_predict(df[kolom_pengelompokan])

# Calculate silhouette scores for each row
df["Silhouette Score"] = silhouette_samples(df[kolom_pengelompokan], df["Cluster"])

# Calculate silhouette score for the entire dataset
silhouette_avg = silhouette_score(df[kolom_pengelompokan], df["Cluster"])

# Menampilkan hasil klasterisasi
if tampilan_terpilih == "Data Asli":
    # Display original data table with silhouette scores
    st.write("### Data Asli:")
    st.write(df)

    # Display silhouette score for the entire dataset
    st.write(f"Skor Silhouette Rata-rata untuk Keseluruhan Data: {silhouette_avg}")

    # Display line chart for silhouette scores
    st.write("### Grafik Garis Silhouette:")
    plt.figure(figsize=(10, 5))
    plt.plot(df["Silhouette Score"])
    plt.axhline(y=silhouette_avg, color="red", linestyle="--", label="Skor Rata-rata")
    plt.xlabel("Data Point")
    plt.ylabel("Silhouette Score")
    plt.legend()
    st.pyplot(plt)

elif tampilan_terpilih == "Tabel Klaster":
    # Menampilkan tabel klaster
    for cluster_id in range(num_clusters):
        cluster_df = df[df['Cluster'] == cluster_id]
        st.write(f"### Tabel Klaster {cluster_id + 1}:")
        st.write(cluster_df)

elif tampilan_terpilih == "Visualisasi Data":
    st.write("### Ringkasan Klaster:")
    ringkasan_klaster = df.groupby("Cluster").agg({
        "Desa": "count",
        **{tahun: "mean" for tahun in kolom_pengelompokan}
    }).rename(columns={"Desa": "Jumlah Desa", **{tahun: f"Rata-rata Populasi {tahun}" for tahun in kolom_pengelompokan}})
    st.write(ringkasan_klaster)

    st.markdown("---")

    # Grafik Scatter Plot Latitude dan Longitude sesuai dengan Klaster
    st.write("### Scatter Plot Latitude dan Longitude:")
    st.write("Grafik ini menunjukkan persebaran desa pada peta berdasarkan klaster.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster_id]
        ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'Cluster {cluster_id + 1}')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Scatter Plot Latitude dan Longitude')
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")

    # Grafik Garis Pertumbuhan Penduduk untuk Setiap Klaster
    st.write("### Grafik Garis Pertumbuhan Penduduk untuk Setiap Klaster:")
    st.write("Grafik ini menunjukkan rata-rata pertumbuhan penduduk setiap klaster dari tahun 2012 hingga 2021.")

    plt.figure(figsize=(10, 5))
    for cluster_id in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster_id]
        plt.plot(kolom_pengelompokan, cluster_data[kolom_pengelompokan].mean(), label=f'Cluster {cluster_id + 1}')

    plt.xlabel("Tahun")
    plt.ylabel("Rata-rata Populasi")
    plt.title("Pertumbuhan Penduduk Setiap Klaster")
    plt.legend()
    st.pyplot(plt)


elif tampilan_terpilih == "Peta Folium":
    st.write("### Peta Folium dengan Klaster:")

    # Membuat peta dengan lokasi rata-rata latitude dan longitude
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    # Menambahkan marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Menambahkan penanda untuk setiap klaster
    for i, row in df.iterrows():
        density = density_info.get(row['Cluster'], 'Tidak Diketahui')

        # Menyesuaikan warna ikon berdasarkan tingkat kepadatan penduduk
        icon_color = 'red' if density == 'Sangat Padat' else 'orange' if density == 'Padat' else 'green'

        folium.Marker([row['Latitude'], row['Longitude']],
                      popup=f"Klaster {row['Cluster'] + 1}: {row['Desa']}<br>Kepadatan: {density}",
                      icon=folium.Icon(color=icon_color)).add_to(marker_cluster)

    # Menambahkan peta ke Streamlit
    folium_static(m)

# Sidebar untuk menambahkan kesimpulan
st.sidebar.markdown("---")
st.sidebar.write("### Kesimpulan:")
st.sidebar.write("Dari analisis klaster, dapat disimpulkan bahwa:")

# Menyimpan hasil klaster untuk setiap klaster
cluster_results = []
for cluster_id in range(num_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    cluster_results.append({
        f"Klaster {cluster_id + 1}": cluster_data['Desa'].tolist()
    })

# Menampilkan kesimpulan
for result in cluster_results:
    for key, value in result.items():
        st.sidebar.write(f"{key}: {', '.join(value)}")

# Menambahkan penjelasan kesimpulan berdasarkan hasil klasterisasi, silhouette score, dll.
st.sidebar.write(f"Penggunaan {num_clusters} klaster terlihat sangat optimal, dengan nilai Silhouette Score rata-rata sebesar {silhouette_avg:.2f}. "
                 f"Nilai Silhouette Score yang tinggi menunjukkan bahwa objek dalam satu klaster memiliki kesamaan yang tinggi dan perbedaan yang rendah, "
                 f"mengindikasikan hasil klasterisasi yang baik.")


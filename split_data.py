import pandas as pd
from datetime import datetime

def split_data_by_year(input_file):
    """
    Memisahkan data berdasarkan tahun dari kolom id_time
    
    Parameters:
    input_file (str): Nama file CSV input yang berisi data gabungan
    """
    
    print(f"Membaca file: {input_file}")
    print("=" * 60)
    
    try:
        # Membaca file CSV input
        df = pd.read_csv(input_file)
        
        print(f"Total records dalam file: {len(df)}")
        print(f"Kolom yang ada: {', '.join(df.columns.tolist())}")
        print("\n")
        
        # Convert kolom id_time ke datetime
        # Format: yyyy-MM-dd HH:mm:ss.SSS
        df['id_time'] = pd.to_datetime(df['id_time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        
        # Cek jika ada data yang gagal dikonversi
        null_dates = df['id_time'].isna().sum()
        if null_dates > 0:
            print(f"⚠️  Warning: {null_dates} baris memiliki format tanggal yang tidak valid")
            print("\n")
        
        # Ekstrak tahun dari kolom id_time
        df['tahun'] = df['id_time'].dt.year
        
        # Tampilkan distribusi data per tahun
        print("Distribusi data per tahun:")
        print("-" * 60)
        year_counts = df['tahun'].value_counts().sort_index()
        for year, count in year_counts.items():
            if pd.notna(year):
                print(f"Tahun {int(year)}: {count} records")
        print("\n")
        
        # Filter dan simpan data untuk tahun 2023
        df_2023 = df[df['tahun'] == 2023].copy()
        df_2023 = df_2023.drop(columns=['tahun'])  # Hapus kolom temporary 'tahun'
        output_file_2023 = 'data_2023.csv'
        df_2023.to_csv(output_file_2023, index=False)
        print(f"✓ Data tahun 2023 disimpan ke: {output_file_2023}")
        print(f"  Jumlah records: {len(df_2023)}")
        
        # Filter dan simpan data untuk tahun 2024
        df_2024 = df[df['tahun'] == 2024].copy()
        df_2024 = df_2024.drop(columns=['tahun'])
        output_file_2024 = 'data_2024.csv'
        df_2024.to_csv(output_file_2024, index=False)
        print(f"✓ Data tahun 2024 disimpan ke: {output_file_2024}")
        print(f"  Jumlah records: {len(df_2024)}")
        
        # Filter dan simpan data untuk tahun 2025
        df_2025 = df[df['tahun'] == 2025].copy()
        df_2025 = df_2025.drop(columns=['tahun'])
        output_file_2025 = 'data_2025.csv'
        df_2025.to_csv(output_file_2025, index=False)
        print(f"✓ Data tahun 2025 disimpan ke: {output_file_2025}")
        print(f"  Jumlah records: {len(df_2025)}")
        
        print("\n" + "=" * 60)
        print("Proses selesai!")
        print(f"Total records yang diproses: {len(df_2023) + len(df_2024) + len(df_2025)}")
        
        # Tampilkan rentang tanggal per file
        print("\n" + "Rentang tanggal per file:")
        print("-" * 60)
        
        if len(df_2023) > 0:
            min_date_2023 = df_2023['id_time'].min()
            max_date_2023 = df_2023['id_time'].max()
            print(f"2023: {min_date_2023} hingga {max_date_2023}")
        else:
            print("2023: Tidak ada data")
            
        if len(df_2024) > 0:
            min_date_2024 = df_2024['id_time'].min()
            max_date_2024 = df_2024['id_time'].max()
            print(f"2024: {min_date_2024} hingga {max_date_2024}")
        else:
            print("2024: Tidak ada data")
            
        if len(df_2025) > 0:
            min_date_2025 = df_2025['id_time'].min()
            max_date_2025 = df_2025['id_time'].max()
            print(f"2025: {min_date_2025} hingga {max_date_2025}")
        else:
            print("2025: Tidak ada data")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' tidak ditemukan!")
        print("Pastikan file berada di direktori yang sama dengan script ini.")
        
    except Exception as e:
        print(f"❌ Error: Terjadi kesalahan saat memproses data")
        print(f"Detail error: {str(e)}")


# Contoh penggunaan
if __name__ == "__main__":
    # Ganti 'data_gabungan.csv' dengan nama file input Anda
    input_filename = 'smartmeter 2023 - 2025.csv'
    
    print("PROGRAM SPLIT DATA BERDASARKAN TAHUN")
    print("=" * 60)
    print("\n")
    
    split_data_by_year(input_filename)
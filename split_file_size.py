import pandas as pd
import os

def get_file_size_mb(filepath):
    """Mendapatkan ukuran file dalam MB"""
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_bytes, size_mb

def split_csv_ultra_strict(input_file, max_size_mb=100):
    """
    Split CSV dengan ULTRA STRICT mode - DIJAMIN tidak melebihi 100MB
    """
    
    max_size_bytes = max_size_mb * 1024 * 1024  # 104,857,600 bytes
    
    # KUNCI: Safety buffer yang SANGAT BESAR (5MB) untuk memastikan tidak melebihi
    safety_buffer = 5 * 1024 * 1024  # 5MB = 5,242,880 bytes
    effective_max_size = max_size_bytes - safety_buffer  # ~95MB
    
    print("=" * 70)
    print("SPLIT CSV - ULTRA STRICT MODE")
    print("=" * 70)
    print(f"File input: {input_file}")
    print(f"Batas maksimal: {max_size_mb} MB ({max_size_bytes:,} bytes)")
    print(f"Safety buffer: {safety_buffer / (1024*1024):.1f} MB ({safety_buffer:,} bytes)")
    print(f"Effective max size: {effective_max_size / (1024*1024):.1f} MB ({effective_max_size:,} bytes)")
    print("=" * 70)
    print()
    
    if not os.path.exists(input_file):
        print(f"❌ File '{input_file}' tidak ditemukan!")
        return
    
    input_size_bytes, input_size_mb = get_file_size_mb(input_file)
    print(f"Ukuran file input: {input_size_mb:.2f} MB ({input_size_bytes:,} bytes)\n")
    
    try:
        # Baca header
        with open(input_file, 'r', encoding='utf-8') as f:
            header_line = f.readline()
        
        header_size = len(header_line.encode('utf-8'))
        print(f"Ukuran header: {header_size:,} bytes\n")
        
        # Mulai split
        with open(input_file, 'r', encoding='utf-8') as infile:
            infile.readline()  # Skip header di input
            
            part_number = 1
            current_size = 0
            current_rows = 0
            total_rows = 0
            output_files = []
            
            # Buka file pertama
            output_filename = f'data_2024_part{part_number}.csv'
            outfile = open(output_filename, 'w', encoding='utf-8')
            outfile.write(header_line)
            current_size = header_size
            
            print(f"📝 Membuat: {output_filename}")
            
            for line in infile:
                line_size = len(line.encode('utf-8'))
                
                # KUNCI: Cek SEBELUM menulis
                if current_size + line_size > effective_max_size:
                    # Tutup file sekarang
                    outfile.close()
                    
                    # Verifikasi ukuran
                    file_size_bytes, file_size_mb = get_file_size_mb(output_filename)
                    
                    if file_size_bytes > max_size_bytes:
                        print(f"❌ {output_filename}: {file_size_mb:.4f} MB - MELEBIHI BATAS!")
                    else:
                        margin_mb = (max_size_bytes - file_size_bytes) / (1024 * 1024)
                        print(f"✓ {output_filename}: {file_size_mb:.4f} MB (margin: {margin_mb:.2f} MB)")
                    
                    output_files.append({
                        'name': output_filename,
                        'size': file_size_bytes,
                        'rows': current_rows,
                        'exceeds': file_size_bytes > max_size_bytes
                    })
                    
                    # Buat file baru
                    part_number += 1
                    output_filename = f'data_2024_part{part_number}.csv'
                    outfile = open(output_filename, 'w', encoding='utf-8')
                    outfile.write(header_line)
                    outfile.flush()
                    current_size = header_size
                    current_rows = 0
                    
                    print(f"📝 Membuat: {output_filename}")
                
                # Tulis baris
                outfile.write(line)
                current_size += line_size
                current_rows += 1
                total_rows += 1
                
                # Progress & safety check setiap 5000 baris
                if total_rows % 5000 == 0:
                    outfile.flush()
                    # Double check ukuran file aktual
                    actual_size = os.path.getsize(output_filename)
                    if actual_size > effective_max_size:
                        # Force close jika sudah terlalu besar
                        outfile.close()
                        file_size_bytes, file_size_mb = get_file_size_mb(output_filename)
                        print(f"✓ {output_filename}: {file_size_mb:.4f} MB (force close)")
                        
                        output_files.append({
                            'name': output_filename,
                            'size': file_size_bytes,
                            'rows': current_rows,
                            'exceeds': file_size_bytes > max_size_bytes
                        })
                        
                        part_number += 1
                        output_filename = f'data_2024_part{part_number}.csv'
                        outfile = open(output_filename, 'w', encoding='utf-8')
                        outfile.write(header_line)
                        current_size = header_size
                        current_rows = 0
                        print(f"📝 Membuat: {output_filename}")
                
                if total_rows % 10000 == 0:
                    print(f"  Progress: {total_rows:,} baris diproses...", end='\r')
            
            # Tutup file terakhir
            outfile.close()
            file_size_bytes, file_size_mb = get_file_size_mb(output_filename)
            
            if file_size_bytes > max_size_bytes:
                print(f"❌ {output_filename}: {file_size_mb:.4f} MB - MELEBIHI BATAS!")
            else:
                margin_mb = (max_size_bytes - file_size_bytes) / (1024 * 1024)
                print(f"✓ {output_filename}: {file_size_mb:.4f} MB (margin: {margin_mb:.2f} MB)")
            
            output_files.append({
                'name': output_filename,
                'size': file_size_bytes,
                'rows': current_rows,
                'exceeds': file_size_bytes > max_size_bytes
            })
        
        # Ringkasan
        print("\n" + "=" * 70)
        print("RINGKASAN")
        print("=" * 70)
        print(f"Total baris: {total_rows:,}")
        print(f"Jumlah file: {len(output_files)}\n")
        
        exceeds_count = 0
        for i, f in enumerate(output_files, 1):
            size_mb = f['size'] / (1024 * 1024)
            status = "❌ MELEBIHI" if f['exceeds'] else "✓ OK"
            print(f"{i}. {f['name']}: {size_mb:.2f} MB - {status}")
            if f['exceeds']:
                exceeds_count += 1
        
        print("\n" + "=" * 70)
        if exceeds_count == 0:
            print("✅ SUKSES! SEMUA FILE DI BAWAH 100MB")
        else:
            print(f"⚠️ {exceeds_count} file melebihi batas!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    split_csv_ultra_strict('data_2024.csv', 100)
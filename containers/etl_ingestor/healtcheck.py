import os
import sys
import rasterio

def main():
    bronze_dir = "/data_lake/bronze"
    os.makedirs(bronze_dir, exist_ok=True)
    
    # Создаем маркерный файл для подтверждения монтирования Shared Volume
    marker_path = os.path.join(bronze_dir, ".volume_mounted")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("OK: Shared volume /data_lake/bronze is writable.\n")
        
    print(f"[ETL-INGESTOR] Контейнер 1 успешно инициализирован.")
    print(f"[ETL-INGESTOR] Версия GDAL/Rasterio: {rasterio.__version__}")
    print(f"[ETL-INGESTOR] Хранилище Bronze доступно: {os.path.abspath(bronze_dir)}")

if __name__ == "__main__":
    sys.exit(main())
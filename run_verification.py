import argparse
from pathlib import Path
from convert_dataset_to_yolo_format import convertir_dataset, verificar_conversion

def main():
    dataset_dir = Path("data/doi_nature_original_subset")
    output_dir = Path("tmp_verify_yolo")
    csv_name = "train_class_label_CSV.csv"

    print(f"Iniciando conversión de prueba desde {dataset_dir}...")
    
    # 1. Convertimos un pequeño subset para generar las etiquetas
    convertir_dataset(
        dir_dataset=dataset_dir,
        dir_salida=output_dir,
        proporciones=(0.8, 0.1, 0.1),
        semilla=42,
        nombre_csv=csv_name
    )

    # 2. Generamos las imágenes con Bbox y Keypoints dibujados
    print("\nGenerando imágenes de verificación...")
    verificar_conversion(output_dir, n_muestras=10)
    
    print(f"\n¡Listo! Revisa la carpeta '{output_dir / 'verificacion'}' para ver si los puntos caen exactamente en las esquinas de los dígitos.")

if __name__ == "__main__":
    main()

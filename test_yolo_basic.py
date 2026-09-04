from ultralytics import YOLO
import cv2

def main():
    # Ruta al modelo y a la imagen
    modelo_path = 'best.pt'
    # Utilizaremos una de las imagenes del subset de validacion que generó el script anterior
    #imagen_path = 'tmp_verify_yolo/images/val/train282.png'
    imagen_path = 'train794.png'

    print(f"Cargando modelo desde: {modelo_path}")
    try:
        # 1. Cargar el modelo
        model = YOLO(modelo_path)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el modelo en '{modelo_path}'")
        return

    print(f"Realizando inferencia sobre: {imagen_path}")
    try:
        # 2. Hacer la predicción
        results = model(imagen_path)
    except Exception as e:
        print(f"ERROR: No se pudo realizar la inferencia en '{imagen_path}'. ¿Existe la imagen?")
        print(e)
        return

    # 3. Procesar los resultados
    for result in results:
        # Mostrar el resultado por pantalla (abrirá una ventana de OpenCV)
        # result.show()  # Comentado para evitar que trabe la consola si no hay entorno gráfico
        
        # Guardar la imagen con las detecciones dibujadas en disco
        ruta_salida = 'resultado_ejemplo0.jpg'
        result.save(ruta_salida)
        print(f"\nImagen con detecciones guardada en: {ruta_salida}")
        
        # Imprimir información sobre lo detectado
        print("\n--- Resultados de Detección ---")
        if result.boxes:
            print(f"Se detectaron {len(result.boxes)} objeto(s).")
            print("Cajas (Bboxes normalizadas y confianzas):")
            print(result.boxes.data)
        else:
            print("No se detectó ningún objeto (Bounding Box).")
            
        if result.keypoints:
            print("\nPuntos clave (Keypoints):")
            print(result.keypoints.data)
        else:
             print("\nNo se detectó ningún punto clave (Keypoint).")

if __name__ == '__main__':
    main()

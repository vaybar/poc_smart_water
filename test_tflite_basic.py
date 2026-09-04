from ultralytics import YOLO
import cv2

def main():
    # Ruta al modelo TFLite y a la imagen
    #modelo_path = 'best_yolo_int8.tflite'
    modelo_path = 'mobilenet_model_claude/best_float16.tflite'
    # Utilizaremos una imagen de prueba. Cambia esto por tu imagen real si es distinto.
    imagen_path = 'img_test_final\\test12553.png'
    #ejemplo0_640.png

    print(f"Cargando modelo TFLite desde: {modelo_path}")
    try:
        # 1. Cargar el modelo
        # La librería ultralytics soporta cargar archivos .tflite directamente
        model = YOLO(modelo_path, task='pose') 
    except FileNotFoundError:
        print(f"ERROR: No se encontró el modelo en '{modelo_path}'")
        return
    except Exception as e:
        print(f"ERROR al cargar el modelo: {e}")
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
        # Guardar la imagen con las detecciones (bbox + pose) usando la función propia de YOLO
        ruta_salida = 'img_resultado_deteccion\\resultado_tflite_ejemplo0.jpg'
        result.save(ruta_salida)
        print(f"\nImagen con detecciones de YOLO guardada en: {ruta_salida}")
        
        # Obtener una copia de la imagen original para dibujar los puntos personalizados
        img = result.orig_img.copy()
        
        # Imprimir información sobre lo detectado y dibujar los puntos
        print("\n--- Resultados de Detección ---")
        if result.boxes:
            print(f"Se detectaron {len(result.boxes)} objeto(s).")
            print("Cajas (Bboxes):")
            print(result.boxes.data)
        else:
            print("No se detectó ningún objeto (Bounding Box).")
            
        if result.keypoints:
            print("\nPuntos clave (Keypoints):")
            print(result.keypoints.data)
            
            # Dibujar los puntos sobre la imagen
            # Iterar sobre las detecciones (por si hay más de una)
            for kpts in result.keypoints.xy:
                for idx, pt in enumerate(kpts):
                    # Convertir coordenadas a enteros
                    x, y = int(pt[0]), int(pt[1])
                    if x > 0 and y > 0:  # Si el punto fue detectado
                        # Dibujar un círculo rojo en el punto
                        cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                        # Dibujar el índice del punto (0, 1, 2, 3) para identificar las esquinas
                        cv2.putText(img, str(idx), (x + 10, y + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Guardar la imagen donde resaltamos exclusivamente los puntos y su índice
            ruta_salida_puntos = 'resultado_tflite_puntos_ejemplo0.jpg'
            cv2.imwrite(ruta_salida_puntos, img)
            print(f"\nImagen con SOLO LOS PUNTOS y sus índices guardada en: {ruta_salida_puntos}")
            
        else:
             print("\nNo se detectaron puntos clave (Keypoints).")

if __name__ == '__main__':
    main()

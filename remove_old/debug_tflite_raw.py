import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

def main():
    #modelo_path = 'best_float16.tflite'
    modelo_path = '../best_yolo_int8.tflite'

    imagen_path = '../img_test_final/test12553.png'

    print(f"--- ANALIZANDO MODELO TFLITE: {modelo_path} ---")
    
    # 1. Cargar el intérprete
    try:
        interpreter = tflite.Interpreter(model_path=modelo_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # 2. Detalles de Entrada y Salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n[Detalles de ENTRADA]")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: shape={detail['shape']}, dtype={detail['dtype']}, name={detail['name']}")
        
    print("\n[Detalles de SALIDA]")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}, name={detail['name']}")

    # 3. Preparar imagen
    input_shape = input_details[0]['shape']
    # input_shape usualmente es [1, H, W, 3] o [1, 3, H, W]
    if input_shape[1] == 3:
        # NCHW
        h, w = input_shape[2], input_shape[3]
        nchw = True
    else:
        # NHWC
        h, w = input_shape[1], input_shape[2]
        nchw = False
        
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error al cargar la imagen {imagen_path}")
        return
        
    img_resized = cv2.resize(img, (w, h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalizar a float32 si es necesario
    if input_details[0]['dtype'] == np.float32:
        img_tensor = img_rgb.astype(np.float32) / 255.0
    else:
        img_tensor = img_rgb
        
    if nchw:
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # 4. Inferencia
    print("\n--- EJECUTANDO INFERENCIA RAW ---")
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Shape del tensor de salida RAW: {output_data.shape}")
    
    # Intentar entender la estructura
    # YOLOv8 normalmente saca [1, C, N] donde C = 4 (bbox) + 1 (clase) + K (keypoints)
    # y N es el número de anclas (ej. 2100 u 8400)
    
    out = output_data[0] # Quitar batch
    if out.shape[0] < out.shape[1] and out.shape[0] < 100:
        print("El tensor parece estar en formato [Channels, Anchors]. Transponiendo a [Anchors, Channels]...")
        out = out.transpose()
        
    print(f"Shape procesado (Anchors, Channels): {out.shape}")
    num_anchors, channels = out.shape
    
    print(f"\nTotal de canales por predicción: {channels}")
    print("Desglose esperado:")
    print(" - 4 canales para Bounding Box (cx, cy, w, h)")
    print(f" - Si tienes 1 clase, te quedan {channels - 5} canales para keypoints.")
    print(f" - Si cada keypoint usa 3 valores (x, y, vis), deberías tener {(channels-5)//3} keypoints exactos.")
    print(f" - Si cada keypoint usa 2 valores (x, y), deberías tener {(channels-5)//2} keypoints exactos.")

if __name__ == "__main__":
    main()
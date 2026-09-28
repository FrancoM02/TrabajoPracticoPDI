import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definimos función para mostrar imágenes (la misma vista en clase, Unidad 3)
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


RUTA_IMAGEN = "imagenes/Imagen_con_detalles_escondidos.tif"
CARPETA_SALIDAS = "salidas"


def ecualizacion_local_histograma(img, ventana):
    # img:     Imagen 2D en escala de grises, uint8.
    # ventana: Tamaño de la ventana de procesamiento. Puede ser un entero (ventana cuadrada)
    #          o una tupla (M, N) = (alto, ancho), con M y N enteros positivos.
    # Salida:  Imagen uint8 del mismo tamaño que la de entrada.
    #
    # Para cada pixel: se calcula el histograma de su ventana, se obtiene la transformación de
    # la ecualización (distribución acumulada * 255) y se aplica SOLO al pixel central.
    if isinstance(ventana, int):
        M, N = ventana, ventana
    else:
        M, N = ventana
    if M < 1 or N < 1:
        raise ValueError("M y N deben ser enteros positivos")

    L = 256                                                 # Niveles de intensidad
    h, w = img.shape

    # Agrego bordes para poder centrar la ventana en los pixels de los extremos
    top, left = M // 2, N // 2
    bottom, right = M - top - 1, N - left - 1               # (Si M o N son pares, un lado recibe un pixel menos)
    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

    #Recorro la imagen
    img_out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            ventana_ij = img_pad[i:i+M, j:j+N]                          # Ventana centrada en el pixel (i,j)
            hist = np.bincount(ventana_ij.ravel(), minlength=L)         # Histograma local
            cdf = np.cumsum(hist) / (M*N)                               # Distribución acumulada normalizada
            img_out[i, j] = np.uint8(np.round((L-1) * cdf[img[i, j]]))  # Transformación aplicada al pixel central
    return img_out



if __name__ == "__main__":
    os.makedirs(CARPETA_SALIDAS, exist_ok=True)

    img = cv2.imread(RUTA_IMAGEN, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {RUTA_IMAGEN}")
    print(img.shape, img.dtype, np.unique(img))                 # Valores: 0 a 11 (objetos) y 226 a 228 (fondo)
    imshow(img, title="Imagen Original", ticks=True)

    #  Ecualización global (para comparar) 
    img_eq = cv2.equalizeHist(img)
    cv2.imwrite(f"{CARPETA_SALIDAS}/global.png", img_eq)

    #  Ecualización local con diferentes tamaños de ventana 
    tamanos = [3, 7, 15, 25, 51, 101]
    resultados = {}
    for t in tamanos:
        print(f"Procesando ventana {t}x{t}...")
        resultados[t] = ecualizacion_local_histograma(img, t)
        cv2.imwrite(f"{CARPETA_SALIDAS}/local_{t}x{t}.png", resultados[t])

    #Comparación 
    plt.figure()
    ax = plt.subplot(241)
    imshow(img, new_fig=False, title="Original", colorbar=False)
    plt.subplot(242, sharex=ax, sharey=ax), imshow(img_eq, new_fig=False, title="Ecualización global", colorbar=False)
    for k, t in enumerate(tamanos):
        plt.subplot(2, 4, k+3, sharex=ax, sharey=ax), imshow(resultados[t], new_fig=False, title=f"Local {t}x{t}", colorbar=False)
    plt.suptitle("Ecualización local de histograma - Influencia del tamaño de ventana")
    plt.savefig(f"{CARPETA_SALIDAS}/comparacion_ventanas.png", dpi=110)
    plt.show()
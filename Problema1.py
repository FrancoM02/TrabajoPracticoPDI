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


# Las rutas se arman respecto de la carpeta del script (funciona sin importar desde dónde se ejecute).
# La imagen se busca junto al script o en la subcarpeta "imagenes", con el nombre del archivo entregado
# o con el que figura en el enunciado.
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
NOMBRES_IMAGEN = ["Imagen_con_detalles_escondidos.tif", "Imagen_con_objetos_ocultos.tiff"]
CARPETA_SALIDAS = os.path.join(CARPETA_SCRIPT, "salidas")


def buscar_imagen():
    for carpeta in (CARPETA_SCRIPT, os.path.join(CARPETA_SCRIPT, "imagenes")):
        for nombre in NOMBRES_IMAGEN:
            ruta = os.path.join(carpeta, nombre)
            if os.path.isfile(ruta):
                return ruta
    raise FileNotFoundError(f"No se encontró {' ni '.join(NOMBRES_IMAGEN)} junto al script ni en la carpeta 'imagenes'")


def ecualizacion_local_histograma(img, ventana):
    # img:     Imagen 2D en escala de grises, uint8.
    # ventana: Tamaño de la ventana de procesamiento. Puede ser un entero (ventana cuadrada)
    #          o una tupla (M, N) = (alto, ancho), con M y N enteros positivos.
    # Salida:  Imagen uint8 del mismo tamaño que la de entrada.
    #
    # Para cada pixel: se calcula el histograma de su ventana, se obtiene la transformación de
    # la ecualización (distribución acumulada * 255) y se aplica SOLO al pixel central.
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("La imagen debe ser 2D (escala de grises) y de tipo uint8")
    if isinstance(ventana, (int, np.integer)):
        M, N = ventana, ventana
    elif isinstance(ventana, (tuple, list)) and len(ventana) == 2:
        M, N = ventana
    else:
        raise ValueError("La ventana debe ser un entero o una tupla (M, N) de enteros positivos")
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in (M, N)):
        raise ValueError("M y N deben ser enteros positivos")
    M, N = int(M), int(N)
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

    ruta_imagen = buscar_imagen()
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")
    print(img.shape, img.dtype, np.unique(img))                 # Valores: 0 a 11 (objetos) y 226 a 228 (fondo)

    #  Imagen original e histograma (escala logarítmica: hay solo dos grupos de valores)
    plt.figure(figsize=(10, 4))
    plt.subplot(121), imshow(img, new_fig=False, title="Imagen Original", colorbar=False)
    plt.subplot(122), plt.bar(range(256), np.bincount(img.ravel(), minlength=256), width=1)
    plt.yscale("log"), plt.xlim(-2, 257), plt.title("Histograma (escala logarítmica)"), plt.xlabel("Nivel de intensidad")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDAS, "histograma_original.png"), dpi=110)
    plt.show(block=False)

    #  Ecualización global (para comparar)
    img_eq = cv2.equalizeHist(img)
    cv2.imwrite(os.path.join(CARPETA_SALIDAS, "global.png"), img_eq)

    #  Ecualización local con diferentes tamaños de ventana
    tamanos = [3, 7, 15, 25, 51, 101]
    resultados = {}
    for t in tamanos:
        print(f"Procesando ventana {t}x{t}...")
        resultados[t] = ecualizacion_local_histograma(img, t)
        cv2.imwrite(os.path.join(CARPETA_SALIDAS, f"local_{t}x{t}.png"), resultados[t])

    #Comparación 
    plt.figure()
    ax = plt.subplot(241)
    imshow(img, new_fig=False, title="Original", colorbar=False)
    plt.subplot(242, sharex=ax, sharey=ax), imshow(img_eq, new_fig=False, title="Ecualización global", colorbar=False)
    for k, t in enumerate(tamanos):
        plt.subplot(2, 4, k+3, sharex=ax, sharey=ax), imshow(resultados[t], new_fig=False, title=f"Local {t}x{t}", colorbar=False)
    plt.suptitle("Ecualización local de histograma - Influencia del tamaño de ventana")
    plt.savefig(os.path.join(CARPETA_SALIDAS, "comparacion_ventanas.png"), dpi=110)
    plt.show()
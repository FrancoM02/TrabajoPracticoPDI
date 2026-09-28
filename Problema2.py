import csv
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


# Parámetros 

CARPETA_IMAGENES = "imagenes"
CARPETA_SALIDAS = "salidas"
IDS_PLANILLAS = [1, 2, 3, 4]           # Planillas a procesar (grade_sheet_<id>.png)
CANTIDAD_REGISTROS = 20                # Filas de datos de cada planilla
TH_LINEAS = 100                        # Umbral para detectar las lineas de la tabla (solo pixels casi negros)
TH_TEXTO = 128                         # Umbral para detectar la tinta dentro de las celdas
FRACCION_LINEAS_H = 0.6                # Una fila es "linea horizontal" si supera este % del maximo de pixels
FRACCION_LINEAS_V = 0.5                # Una columna es "linea vertical" si supera este % del maximo de pixels
MARGEN_CELDA = 2                       # Pixels que se descartan dentro de cada celda (evita restos de lineas)
AREA_MINIMA = 2                        # Area minima de una componente (el guion "-" tiene area 5)
CONTAR_ESPACIOS_EN_NOMBRE = False      # True: los espacios cuentan para el maximo de 12 caracteres de "Nombre y apellido"
CAMPOS = ["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
MOSTRAR_FIGURAS = True



#  Detección de celdas 

def encontrar_tramos(v):
    # Recibe un vector booleano y devuelve (inicio, fin) de cada tramo consecutivo de True.
    # Sirve porque las lineas pueden tener mas de un pixel de ancho.
    ix = np.flatnonzero(v)
    if len(ix) == 0:
        return []
    cortes = np.flatnonzero(np.diff(ix) > 1)
    inicios = np.r_[ix[0], ix[cortes + 1]]
    fines = np.r_[ix[cortes], ix[-1]]
    return [(int(i), int(f)) for i, f in zip(inicios, fines)]


def detectar_celdas(img):
    # Detecta las lineas de la tabla y devuelve, para cada registro, las 6 celdas de datos
    # (Legajo, Nombre y apellido, Parcial 1, 2, 3 y Condicion Final) como (y0, y1, x0, x1).
    img_th = img < TH_LINEAS
    img_rows = np.sum(img_th, 1)
    img_cols = np.sum(img_th, 0)
    lineas_h = encontrar_tramos(img_rows > FRACCION_LINEAS_H * img_rows.max())
    lineas_v = encontrar_tramos(img_cols > FRACCION_LINEAS_V * img_cols.max())

    # Lineas horizontales esperadas: borde superior + fin del encabezado + 1 por registro
    # Lineas verticales esperadas: 7 columnas --> 8 lineas
    if len(lineas_h) != CANTIDAD_REGISTROS + 2 or len(lineas_v) != 8:
        raise ValueError(f"Tabla no detectada: {len(lineas_h)} lineas horizontales y {len(lineas_v)} verticales")

    registros = []
    for r in range(CANTIDAD_REGISTROS):
        y0 = lineas_h[r + 1][1] + 1         # Justo debajo de la linea superior de la fila
        y1 = lineas_h[r + 2][0]             # Justo arriba de la linea inferior de la fila
        celdas = []
        for c in range(1, 7):               # La columna 0 es "Nro." y no se valida
            x0 = lineas_v[c][1] + 1
            x1 = lineas_v[c + 1][0]
            celdas.append((y0, y1, x0, x1))
        registros.append(celdas)
    return registros, lineas_h, lineas_v


#  Extracción de caracteres 

def unir_componentes(stats):
    # Une componentes que se superponen horizontalmente (por ejemplo, la tilde de la "Ñ",
    # que es una componente conectada distinta de la letra).
    cajas = []
    for x, y, w, h, a in stats:
        if cajas:
            xa, ya, wa, ha, aa = cajas[-1]
            solape = min(xa + wa, x + w) - max(xa, x)
            if solape >= 0.5 * min(wa, w):
                x_nuevo, y_nuevo = min(xa, x), min(ya, y)
                x_fin, y_fin = max(xa + wa, x + w), max(ya + ha, y + h)
                cajas[-1] = [x_nuevo, y_nuevo, x_fin - x_nuevo, y_fin - y_nuevo, aa + a]
                continue
        cajas.append([int(x), int(y), int(w), int(h), int(a)])
    return cajas


def extraer_caracteres(img, celda):
    # Devuelve las cajas (x, y, w, h, area) de los caracteres de una celda, ordenadas de izquierda
    # a derecha, y la imagen binaria de la celda (1 = tinta).
    y0, y1, x0, x1 = celda
    m = MARGEN_CELDA
    celda_th = np.uint8(img[y0:y1, x0:x1] < TH_TEXTO)[m:-m, m:-m]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_th, connectivity=8, ltype=cv2.CV_32S)
    stats = stats[1:]                                   # Descartamos el fondo (etiqueta 0)
    stats = stats[stats[:, -1] >= AREA_MINIMA]          # Descartamos ruido de area muy pequeña
    stats = stats[np.argsort(stats[:, 0])]              # Ordenamos de izquierda a derecha
    cajas = unir_componentes(stats)
    return cajas, celda_th


def estimar_paso(cajas_por_campo):
    # La fuente es de ancho fijo: la distancia entre los centros de dos caracteres consecutivos
    # es siempre el mismo "paso", y un espacio agrega un paso mas. Lo estimamos con la mediana.
    distancias = []
    for cajas in cajas_por_campo:
        if len(cajas) >= 2:
            centros = [c[0] + c[2] / 2 for c in cajas]
            distancias += list(np.diff(centros))
    return float(np.median(distancias)) if len(distancias) > 0 else 10.0


def analizar_campo(cajas, paso):
    # Devuelve (cantidad de caracteres, cantidad de palabras, largo contando los espacios)
    if len(cajas) == 0:
        return 0, 0, 0
    centros = [c[0] + c[2] / 2 for c in cajas]
    palabras = 1
    largo_con_espacios = 1
    for i in range(1, len(centros)):
        saltos = max(1, int(round((centros[i] - centros[i - 1]) / paso)))   # 1 = consecutivos | 2 = hay un espacio
        largo_con_espacios += saltos
        if saltos >= 2:
            palabras += 1
    return len(cajas), palabras, largo_con_espacios


#  Validación de campos 

def validar_campo(indice_campo, cajas, paso):
    n_caracteres, n_palabras, largo_con_espacios = analizar_campo(cajas, paso)
    if indice_campo == 0:       # Legajo: 8 caracteres en total, una unica palabra
        return n_caracteres == 8 and n_palabras == 1
    if indice_campo == 1:       # Nombre y apellido: minimo 2 palabras, maximo 12 caracteres
        largo = largo_con_espacios if CONTAR_ESPACIOS_EN_NOMBRE else n_caracteres
        return n_palabras >= 2 and largo <= 12
    if indice_campo in (2, 3, 4):   # Parciales: 1 o 2 caracteres consecutivos
        return n_caracteres in (1, 2) and n_palabras == 1
    if indice_campo == 5:       # Condicion Final: un unico caracter
        return n_caracteres == 1
    return False


def clasificar_condicion(celda_th, caja):
    # Distingue "A", "L" y "R" a partir de la forma de la letra (sin usar plantillas externas):
    #   - "L" no tiene agujeros.
    #   - "A" y "R" tienen un agujero. La "R" tiene el palo vertical a la izquierda (mucha tinta
    #     en sus primeras columnas), mientras que la "A" tiene solo el pie de la pata.
    x, y, w, h, area = caja
    letra = celda_th[y:y + h, x:x + w]
    contornos, jerarquia = cv2.findContours(letra, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    agujeros = sum(1 for j in jerarquia[0] if j[3] != -1)        # Contornos que tienen "madre/padre"
    tinta_izquierda = letra[:, :2].mean()                        # Fraccion de tinta en las 2 primeras columnas
    if agujeros == 0:
        return "L"
    if agujeros == 1 and tinta_izquierda > 0.45:
        return "R"
    if agujeros == 1:
        return "A"
    return "?"


#  Salidas 

def mostrar_por_terminal(resultados):
    for i, res in enumerate(resultados):
        print(f"> Registro {i + 1}:")
        for nombre, ok in zip(CAMPOS, res):
            print(f"> {nombre}: {'OK' if ok else 'MAL'}")
        print(">")


def guardar_csv(resultados, ruta_csv):
    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["ID", "Legajo", "Nombre y Apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"])
        for i, res in enumerate(resultados):
            escritor.writerow([i + 1] + ["OK" if ok else "MAL" for ok in res])


def generar_imagen_salida(img, celdas_nombre, no_aprobados):
    # no_aprobados: lista de (numero de registro, condicion "L" o "R")
    # Cada fila: numero de registro | crop del campo "Nombre y apellido" | indicador de color
    # Colores en BGR: naranja = debe recuperar (R) | rojo = libre (L)
    escala = 2
    color_ind = {"R": (0, 165, 255), "L": (0, 0, 255)}
    texto_ind = {"R": "RECUPERA", "L": "LIBRE"}
    ancho_numero, ancho_indicador, alto_titulo = 60, 170, 50

    if len(no_aprobados) == 0:
        salida = np.full((120, 700, 3), 255, np.uint8)
        cv2.putText(salida, "Sin alumnos que no hayan aprobado", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return salida

    crops = []
    for nro, cond in no_aprobados:
        y0, y1, x0, x1 = celdas_nombre[nro - 1]
        crop = img[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
        crops.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))

    ancho_crop = max(c.shape[1] for c in crops)
    ancho_total = ancho_numero + ancho_crop + ancho_indicador
    filas = []
    for (nro, cond), crop in zip(no_aprobados, crops):
        alto = crop.shape[0] + 8
        fila = np.full((alto, ancho_total, 3), 255, np.uint8)
        cv2.putText(fila, f"{nro:02d}", (12, alto // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        fila[4:4 + crop.shape[0], ancho_numero:ancho_numero + crop.shape[1]] = crop
        x_ind = ancho_numero + ancho_crop
        fila[:, x_ind:] = color_ind[cond]
        cv2.putText(fila, texto_ind[cond], (x_ind + 12, alto // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(fila, (0, alto - 1), (ancho_total, alto - 1), (180, 180, 180), 1)
        filas.append(fila)

    titulo = np.full((alto_titulo, ancho_total, 3), 255, np.uint8)
    cv2.putText(titulo, "Alumnos que no aprobaron", (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return np.vstack([titulo] + filas)



#  Procesamiento de una planilla 
 
def validar_planilla(id_planilla):
    ruta = f"{CARPETA_IMAGENES}/grade_sheet_{id_planilla}.png"
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")

    #  Detecto celdas y extraigo los caracteres de cada una 
    registros, lineas_h, lineas_v = detectar_celdas(img)
    datos = []
    for celdas in registros:
        datos.append([extraer_caracteres(img, celda) for celda in celdas])   # (cajas, celda_th) por campo

    todas_las_cajas = [cajas for registro in datos for cajas, _ in registro]
    paso = estimar_paso(todas_las_cajas)

    #  Valido campos 
    resultados = []
    for registro in datos:
        resultados.append([validar_campo(i, cajas, paso) for i, (cajas, _) in enumerate(registro)])

    #  Alumnos que no aprobaron (solo registros cargados correctamente) 
    no_aprobados = []
    for i, (registro, res) in enumerate(zip(datos, resultados)):
        if all(res):
            cajas, celda_th = registro[5]
            cond = clasificar_condicion(celda_th, cajas[0])
            if cond in ("L", "R"):
                no_aprobados.append((i + 1, cond))
    celdas_nombre = [celdas[1] for celdas in registros]
    img_salida = generar_imagen_salida(img, celdas_nombre, no_aprobados)

    #  Guardo salidas 
    os.makedirs(CARPETA_SALIDAS, exist_ok=True)
    cv2.imwrite(f"{CARPETA_SALIDAS}/no_aprobados_grade_sheet_{id_planilla}.png", img_salida)
    guardar_csv(resultados, f"{CARPETA_SALIDAS}/resultados_grade_sheet_{id_planilla}.csv")

    return img, registros, resultados, no_aprobados, img_salida, paso


#  Aplicación cíclica sobre las planillas 

if __name__ == "__main__":
    for id_planilla in IDS_PLANILLAS:
        print("=" * 60)
        print(f"PLANILLA {id_planilla}  (grade_sheet_{id_planilla}.png)")
        print("=" * 60)
        img, registros, resultados, no_aprobados, img_salida, paso = validar_planilla(id_planilla)
        mostrar_por_terminal(resultados)

        n_correctos = sum(1 for res in resultados if all(res))
        print(f"Registros cargados correctamente: {n_correctos} de {CANTIDAD_REGISTROS}")
        print(f"Alumnos que no aprobaron (registros correctos): {no_aprobados}")
        print()

        if MOSTRAR_FIGURAS:
            imshow(cv2.cvtColor(img_salida, cv2.COLOR_BGR2RGB), color_img=True, colorbar=False,
                   title=f"Planilla {id_planilla} - Alumnos que no aprobaron")

    if MOSTRAR_FIGURAS:
        plt.show()
# Trabajo Práctico N° 1 - Procesamiento de Imágenes I (2026, 2° semestre)

Tecnicatura Universitaria en Inteligencia Artificial - Facultad de Ciencias Exactas, Ingeniería y Agrimensura - Universidad Nacional de Rosario.

Resolución en Python de los dos problemas del trabajo práctico:

| Problema | Script | Descripción |
|---|---|---|
| 1 | `Problema1.py` | Ecualización local de histograma y análisis de la imagen con detalles ocultos |
| 2 | `Problema2.py` | Validación automática de planillas de calificaciones |

El informe con la descripción de los problemas, las técnicas utilizadas, capturas de los pasos intermedios y las conclusiones está en `informe_TP1.pdf`.

## Contenido del repositorio

```
.
├── Problema1.py
├── Problema2.py
├── informe_TP1.pdf
├── README.md
└── TUIA_PDI_TP1_2026_C2.pdf   (enunciado)
```

Como pide la cátedra, el repositorio contiene solo archivos `.py`, `.pdf` y `.md`. Las imágenes de entrada (las que provee el enunciado) no se incluyen y hay que copiarlas antes de ejecutar (ver "Preparación"). La carpeta `salidas/` la crean los scripts al ejecutarse.

## Requisitos y versiones utilizadas

| Elemento | Versión |
|---|---|
| Python | 3.12.3 |
| numpy | 2.4.4 |
| opencv-python | 4.13.0 |
| matplotlib | 3.10.8 |

También se verificó con Python 3.13, numpy 2.5, opencv-python 5.0 y matplotlib 3.11, con resultados idénticos.

El módulo `csv` y el módulo `os` forman parte de la biblioteca estándar de Python.

Instalación sugerida en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS (en fish: source .venv/bin/activate.fish)
# .venv\Scripts\activate           # Windows
pip install numpy==2.4.4 opencv-python==4.13.0 matplotlib==3.10.8
```

## Preparación

Copiar las imágenes de entrada del enunciado en la misma carpeta que los scripts o dentro de una subcarpeta llamada `imagenes`. Las rutas se arman respecto de la carpeta del script, así que se pueden ejecutar desde cualquier directorio.

| Script | Archivos necesarios |
|---|---|
| `Problema1.py` | `Imagen_con_detalles_escondidos.tif` (también se acepta el nombre del enunciado, `Imagen_con_objetos_ocultos.tiff`) |
| `Problema2.py` | `grade_sheet_1.png`, `grade_sheet_2.png`, `grade_sheet_3.png` y `grade_sheet_4.png` |

## Ejecución

Desde la carpeta del repositorio:

```bash
python Problema1.py
python Problema2.py
```

### Problema1.py

Aplica la ecualización local de histograma a la imagen con detalles ocultos, con ventanas de 3x3, 7x7, 15x15, 25x25, 51x51 y 101x101, y la compara con la ecualización global. La función principal es `ecualizacion_local_histograma(img, ventana)`, que recibe la imagen (2D, uint8) y el tamaño de la ventana (un entero para una ventana cuadrada o una tupla `(M, N)` de enteros positivos). Todo el script tarda unos segundos (la ventana de 101x101 es la más lenta).

Salidas en la carpeta `salidas/` (se crea sola):

- `histograma_original.png` (imagen original e histograma)
- `global.png`
- `local_3x3.png`, `local_7x7.png`, `local_15x15.png`, `local_25x25.png`, `local_51x51.png`, `local_101x101.png`
- `comparacion_ventanas.png`

### Problema2.py

Recorre de forma cíclica las cuatro planillas. Para cada una:

1. Muestra por terminal el resultado (OK o MAL) de cada campo de cada registro.
2. Genera una imagen con los alumnos que no aprobaron (Condición Final "L" o "R"), considerando solo los registros cargados correctamente. Naranja indica que debe recuperar ("R") y rojo que está libre ("L").
3. Genera un archivo CSV con los resultados de la validación.

Al terminar, junta las imágenes de las cuatro planillas en una **única imagen de salida** (`no_aprobados_todas_las_planillas.png`), separada por planilla.

**Decisión sobre la "única imagen de salida".** El enunciado pide que el algoritmo tome como entrada la imagen de *un* formulario y que el CSV tenga un ID que corresponda al orden de los registros de *la* planilla, así que la salida natural es una imagen y un CSV por planilla. Como el punto b también puede leerse como una sola imagen para todo el conjunto, se generan las dos: una por planilla y una global.

Salidas en la carpeta `salidas/`:

- `no_aprobados_grade_sheet_<id>.png` (una por planilla)
- `no_aprobados_todas_las_planillas.png` (imagen única con las cuatro planillas)
- `resultados_grade_sheet_<id>.csv` (uno por planilla)
- `pasos_lineas_grade_sheet_1.png`, `pasos_proyecciones_grade_sheet_1.png`, `pasos_caracteres_grade_sheet_1.png`, `pasos_condicion_grade_sheet_1.png` y `pasos_casos_particulares.png` (pasos intermedios; son las figuras 3 a 7 del informe)

Además, muestra en pantalla las imágenes de salida y los pasos intermedios. El script termina al cerrar las ventanas de las figuras.

## Parámetros configurables (al inicio de `Problema2.py`)

| Constante | Descripción |
|---|---|
| `CONTAR_ESPACIOS_EN_NOMBRE` | Si es `True`, los espacios cuentan para el máximo de 12 caracteres del campo "Nombre y apellido". Por defecto es `False` |
| `MOSTRAR_FIGURAS` | Muestra las imágenes de salida (por planilla y la global) |
| `MOSTRAR_PASOS_INTERMEDIOS` | Muestra y guarda los pasos intermedios (figuras del informe) |
| `IDS_PLANILLAS` | Identificadores de las planillas a procesar |

Si se ejecuta en un entorno sin pantalla, se puede definir `MPLBACKEND=Agg` antes de ejecutar el script (las figuras se guardan igual en `salidas/`).

## Limitaciones conocidas

- Se espera una tabla de 20 registros y 7 columnas con líneas negras, a una resolución igual o mayor que la de las planillas de ejemplo. Si la tabla no se detecta, el programa se detiene con un error en lugar de dar resultados incorrectos.
- Los espacios se detectan con el paso de la fuente de ancho fijo de Legajo y Nombre. En las notas, que usan una fuente proporcional, un espacio mide la mitad de un dígito, así que un espacio entre dos dígitos (por ejemplo "1 0") en general no se detecta. Las planillas de ejemplo no tienen este caso.
- La Condición Final se clasifica por la forma de la letra. Solo se reconocen "L", "R" y "A"; cualquier otro carácter queda sin clasificar y ese registro no aparece en la imagen de no aprobados. Con fuentes de trazo muy fino a tamaño chico (por ejemplo Courier New a 11–12 px) una "L" o una "R" puede no reconocerse; en ese caso el registro queda afuera de la imagen, nunca entra con una condición equivocada.

# Trabajo Práctico N° 1 - Procesamiento de Imágenes I (2026, 2° semestre)

Tecnicatura Universitaria en Inteligencia Artificial - Facultad de Ciencias Exactas, Ingeniería y Agrimensura - Universidad Nacional de Rosario.

Resolución en Python de los dos problemas del trabajo práctico:

| Problema | Script | Descripción |
|---|---|---|
| 1 | `problema1.py` | Ecualización local de histograma y análisis de la imagen con detalles ocultos |
| 2 | `problema2.py` | Validación automática de planillas de calificaciones |

El informe con la descripción de los problemas, las técnicas utilizadas, capturas de los pasos intermedios y las conclusiones está en `informe_TP1.pdf`.

## Contenido del repositorio

```
.
├── problema1.py
├── problema2.py
├── informe_TP1.pdf
└── README.md
```

Las imágenes de entrada (no se incluyen en el repositorio) y las salidas que generan los scripts no forman parte de la entrega.

## Requisitos y versiones utilizadas

| Elemento | Versión |
|---|---|
| Python | 3.12.3 |
| numpy | 2.4.4 |
| opencv-python | 4.13.0 |
| matplotlib | 3.10.8 |

El módulo `csv` y el módulo `os` forman parte de la biblioteca estándar de Python.

Instalación sugerida en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS (en fish: source .venv/bin/activate.fish)
# .venv\Scripts\activate           # Windows
pip install numpy==2.4.4 opencv-python==4.13.0 matplotlib==3.10.8
```

## Preparación

Copiar las imágenes de entrada en la misma carpeta que los scripts, o dentro de una subcarpeta llamada `imagenes`:

| Script | Archivos necesarios |
|---|---|
| `problema1.py` | `Imagen_con_detalles_escondidos.tif` (también se acepta el nombre `Imagen_con_objetos_ocultos.tiff`) |
| `problema2.py` | `grade_sheet_1.png`, `grade_sheet_2.png`, `grade_sheet_3.png` y `grade_sheet_4.png` |

## Ejecución

Desde la carpeta del repositorio:

```bash
python problema1.py
python problema2.py
```

### problema1.py

Aplica la ecualización local de histograma a la imagen con detalles ocultos, con ventanas de 3x3, 7x7, 15x15, 25x25, 51x51 y 101x101, y la compara con la ecualización global. La función principal es `ecualizacion_local_histograma(img, ventana)`, que recibe la imagen y el tamaño de la ventana (un entero para una ventana cuadrada o una tupla `(M, N)`). La ventana de 101x101 es la más lenta: el script puede tardar algunos minutos en terminar.

Salidas en la carpeta `salidas/` (se crea sola):

- `global.png`
- `local_3x3.png`, `local_7x7.png`, `local_15x15.png`, `local_25x25.png`, `local_51x51.png`, `local_101x101.png`
- `comparacion_ventanas.png`

### problema2.py

Recorre de forma cíclica las cuatro planillas. Para cada una:

1. Muestra por terminal el resultado (OK o MAL) de cada campo de cada registro.
2. Genera una imagen con los alumnos que no aprobaron (Condición Final "L" o "R"), considerando solo los registros cargados correctamente. Naranja indica que debe recuperar ("R") y rojo que está libre ("L").
3. Genera un archivo CSV con los resultados de la validación.

Salidas en la carpeta `salidas/`:

- `no_aprobados_grade_sheet_<id>.png`
- `resultados_grade_sheet_<id>.csv`

Además, muestra en pantalla los pasos intermedios de la primera planilla (líneas detectadas, proyecciones y caracteres detectados). La ventana de figuras se cierra al terminar de mirarlas.

## Parámetros configurables (al inicio de `problema2.py`)

| Constante | Descripción |
|---|---|
| `CONTAR_ESPACIOS_EN_NOMBRE` | Si es `True`, los espacios cuentan para el máximo de 12 caracteres del campo "Nombre y apellido". Por defecto es `False` |
| `MOSTRAR_FIGURAS` | Muestra la imagen de salida de cada planilla |
| `MOSTRAR_PASOS_INTERMEDIOS` | Muestra los pasos intermedios de la primera planilla |
| `IDS_PLANILLAS` | Identificadores de las planillas a procesar |

Si se ejecuta en un entorno sin pantalla, se puede definir `MPLBACKEND=Agg` antes de ejecutar el script.

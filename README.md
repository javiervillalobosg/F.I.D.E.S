# F.I.D.E.S

Framework de Investigación y Detección de Evidencia Sintética. Prototipo de investigación que analiza imágenes posiblemente editadas con IA generativa, registra su cadena de custodia y genera un informe PDF con la probabilidad estimada, sus límites y los parámetros usados.

Proyecto de título, Ingeniería Civil Informática, Universidad Andrés Bello.

## Estado

Es un piloto: 24 pares de imágenes (fotografía real y edición de imagen completa con una sola herramienta), 22 rasgos de bajo nivel y una regresión logística calibrada con escalado de Platt. Los resultados son indicios, no prueba, y no tienen validez pericial por sí solos.

Resultados de la validación cruzada agrupada por fotografía (AUC, modelo entrenado con todas las variantes):

| Original | JPEG 90 | JPEG 70 | JPEG 50 | Escala 50 % |
|---|---|---|---|---|
| 0,81 | 0,81 | 0,80 | 0,76 | 0,64 |

Limitaciones: muestra pequeña, un solo generador, sin edición localizada, sin evaluación sobre la cadena real de WhatsApp ni intentos de evasión, y calibración sin mejora demostrable con esta cantidad de datos. No analiza capturas de mensajería ni audio.

## Contenido

| Archivo | Función |
|---|---|
| `fides_dataset.py` | Empareja, normaliza y etiqueta el conjunto; genera las variantes degradadas |
| `fides_eval.py` | Extrae los rasgos y evalúa con validación cruzada, bootstrap y calibración |
| `fides_ablacion.py` | Ablación por familia de rasgos y tiempos por imagen |
| `fides_mvp.py` | Ingesta con custodia, análisis e informe PDF |
| `test_fides_mvp.py` | Pruebas automatizadas |
| `resultados/` | Rasgos y resultados del piloto |

Las fotografías del conjunto no se incluyen en el repositorio.

## Instalación

```
pip install -r requirements.txt
```

## Uso

Preparación del conjunto (carpetas `real/` y `edited/`, con la edición nombrada como la original más `_edit`):

```
python fides_dataset.py prepare --raw . --out data --tool "ChatGPT" --exclude IMG_8482
python fides_dataset.py degrade --out data
```

Evaluación y ablación:

```
python fides_eval.py --data data --out resultados
python fides_ablacion.py --data data --out resultados
```

Prototipo:

```
python fides_mvp.py train
python fides_mvp.py analyze foto.jpg
python fides_mvp.py verify
```

`analyze` registra el archivo con su SHA-256, guarda una copia de solo lectura, anota cada operación como evento con hash encadenado en SQLite y escribe el informe en `informes/`. `verify` recalcula la cadena y las copias, y falla si algo fue alterado.

Pruebas:

```
python -m pytest -q
```

## Docker

```
docker build -t fides .
docker run --rm -v "%cd%":/data fides analyze foto.jpg
```

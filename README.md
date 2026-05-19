# Sistema Inteligente de Clasificación e Inventariado de Productos de Despensa

Sistema que clasifica automáticamente productos de despensa a partir de imágenes,
genera códigos únicos de inventario y registra los productos en una base de datos
PostgreSQL alojada en Supabase.

---

## Descripción del proyecto

El sistema recibe una imagen de un producto (aceite, arroz, pasta, etc.), la analiza
con un modelo de inteligencia artificial y determina a cuál de las siete categorías de
despensa pertenece. Con esa información genera un código de inventario estructurado
(`INV-CAF-202605-0001`) y permite registrar el producto en la base de datos.

**Caso de uso principal:** digitalizar y organizar el inventario de una despensa o
bodega clasificando productos mediante fotos, sin necesidad de leer etiquetas ni
códigos de barras manualmente.

El usuario puede:
- Clasificar una imagen individual y guardar el producto.
- Hacer cargue masivo de imágenes (múltiples archivos o un ZIP) y guardar todo el
  lote de una sola vez.
- Consultar y exportar el inventario registrado.

---

## Arquitectura y tecnologías

### Stack completo

| Capa | Tecnología |
|------|-----------|
| Interfaz de usuario | Streamlit |
| API REST | Flask + Flask-CORS |
| Inferencia del modelo | ONNX Runtime |
| Base de datos | PostgreSQL (Supabase) |
| Procesamiento de imágenes | Pillow + NumPy |
| Exportación de reportes | openpyxl (Excel) + reportlab (PDF) |
| Contenerización | Docker + Docker Compose |
| Entrenamiento del modelo | TensorFlow / Keras + MobileNetV2 |
| Exportación a ONNX | tf2onnx |
| Preparación del dataset | Hugging Face Datasets |

### Por qué se migró de TensorFlow a ONNX Runtime

El modelo se **entrena** con TensorFlow/Keras, pero se **sirve** con ONNX Runtime.
Esta separación trae varias ventajas:

- **Arranque más rápido:** `onnxruntime` importa en milisegundos; TensorFlow puede
  tardar varios segundos en inicializar.
- **Menor huella de memoria:** el runtime de inferencia es mucho más liviano que la
  librería completa de TensorFlow.
- **Sin GPU requerida en producción:** ONNX Runtime funciona bien en CPU, que es el
  entorno de despliegue habitual de la API.
- **Portabilidad:** el archivo `.onnx` es estándar y puede correrse en otros runtimes
  o lenguajes sin depender de TensorFlow.

TensorFlow sigue siendo una dependencia del proyecto para entrenamiento y conversión,
pero **no se necesita en producción** (solo `onnxruntime`).

### Diagrama de componentes

```
                        ┌─────────────────────────────┐
                        │        Usuario              │
                        └──────────┬──────────────────┘
                                   │ navegador
                        ┌──────────▼──────────────────┐
                        │   Streamlit  :8501          │
                        │   streamlit_app/main.py     │
                        └──────────┬──────────────────┘
                                   │ HTTP (requests)
                        ┌──────────▼──────────────────┐
                        │   Flask API  :5000          │
                        │   app/api/app.py            │
                        └────────┬────────────────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               │                                    │
   ┌───────────▼────────────┐        ┌──────────────▼──────────────┐
   │    ONNX Runtime        │        │   PostgreSQL (Supabase)      │
   │  product_classifier    │        │   tabla: products           │
   │       .onnx            │        │   psycopg2 + python-dotenv  │
   └────────────────────────┘        └─────────────────────────────┘
```

---

## Estructura del proyecto

```text
programacion-avanzada-trabajo-final/
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                  ← API Flask (endpoints)
│   │   └── prediction_utils.py     ← Inferencia con ONNX Runtime
│   ├── database/
│   │   └── inventory.db            ← Base de datos SQLite local (legado)
│   └── services/
│       ├── __init__.py
│       ├── code_generator.py       ← Genera códigos INV-XXX-YYYYMM-0000
│       └── inventory_service.py    ← CRUD contra PostgreSQL (psycopg2)
│
├── datasets/
│   └── processed/
│       ├── labels.json             ← Mapa índice → categoría
│       ├── dataset_report.json     ← Reporte de preparación (JSON)
│       └── dataset_report.md       ← Reporte de preparación (Markdown)
│
├── scripts/
│   ├── prepare_hf_dataset.py       ← Descarga y prepara el dataset desde HF
│   ├── train_model.py              ← Entrena MobileNetV2 con Transfer Learning
│   ├── export_to_onnx.py           ← Convierte .keras → .onnx
│   ├── migrate_to_postgres.py      ← Migra datos de SQLite a PostgreSQL
│   ├── predict_image.py            ← Predice una imagen desde consola
│   ├── predict_and_register.py     ← Predice y registra en BD desde consola
│   └── list_inventory.py           ← Lista el inventario desde consola
│
├── streamlit_app/
│   └── main.py                     ← Interfaz Streamlit (4 tabs)
│
├── trained_models/
│   ├── product_classifier.onnx     ← Modelo listo para inferencia (generado)
│   ├── product_classifier.keras    ← Modelo Keras (generado, no se sube)
│   ├── training_report.json        ← Métricas de entrenamiento
│   ├── accuracy_loss.png           ← Gráfica de entrenamiento
│   └── confusion_matrix.png        ← Matriz de confusión
│
├── .env                            ← Variables de entorno (NO se sube a Git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Variables de entorno

El archivo `.env` debe crearse en la raíz del proyecto con este contenido:

```env
DATABASE_URL=postgresql://usuario:contraseña@host:5432/postgres?sslmode=require
```

> El archivo `.env` está en `.gitignore` y **nunca debe subirse al repositorio**.
> Cada integrante debe crearlo manualmente con la URL real de la base de datos.

---

## Requisitos previos

- Python 3.11
- Git
- Acceso a un proyecto de Supabase (para la base de datos PostgreSQL)
- Visual Studio Code (recomendado) o cualquier editor
- Docker y Docker Compose (solo para el modo contenedores — opcional)

---

## Instalación

### 1. Clonar el repositorio

```powershell
git clone URL_DEL_REPOSITORIO
cd programacion-avanzada-trabajo-final
```

### 2. Crear y activar el entorno virtual

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell rechaza el script de activación:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Con el entorno activo la terminal debe mostrar `(.venv)` al inicio del prompt.

### 3. Instalar dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configurar la base de datos

Crear el archivo `.env` en la raíz del proyecto con la `DATABASE_URL` de Supabase:

```powershell
# Ejemplo (reemplazar con los valores reales):
# DATABASE_URL=postgresql://postgres:TuContraseña@db.xxxx.supabase.co:5432/postgres?sslmode=require
```

Luego crear la tabla en el **SQL Editor de Supabase**:

```sql
CREATE TABLE IF NOT EXISTS products (
    id               SERIAL PRIMARY KEY,
    code             TEXT UNIQUE NOT NULL,
    name             TEXT NOT NULL,
    category         TEXT NOT NULL,
    quantity         INTEGER NOT NULL DEFAULT 1,
    image_path       TEXT NOT NULL,
    confidence       DOUBLE PRECISION NOT NULL,
    created_at       TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'activo'
);
```

---

## Preparar el dataset

El dataset de imágenes **no se sube al repositorio**. Cada integrante debe
generarlo ejecutando:

```powershell
python scripts/prepare_hf_dataset.py --clean-output
```

Este comando descarga el dataset
[`valentinafevu/productos-supermercado`](https://huggingface.co/datasets/valentinafevu/productos-supermercado)
desde Hugging Face en modo streaming y organiza las imágenes en la estructura
necesaria para el entrenamiento.

Al finalizar se crean las carpetas:

```text
datasets/processed/train/
datasets/processed/validation/
datasets/processed/test/
datasets/processed/otros/
```

Y los archivos de reporte:

```text
datasets/processed/labels.json
datasets/processed/dataset_report.json
datasets/processed/dataset_report.md
```

**Resultado esperado (aproximado):**

```text
Total imágenes procesadas : 2487
Train                     : 1740
Validation                : 373
Test                       : 374
```

| Categoría | Imágenes |
|-----------|----------|
| arroz_y_granos | 528 |
| cafe_chocolate | 693 |
| azucar_sal | 396 |
| salsas_y_condimentos | 322 |
| enlatados | 217 |
| pastas | 204 |
| aceites | 127 |

---

## Entrenar el modelo

```powershell
python scripts/train_model.py
```

El script usa MobileNetV2 con Transfer Learning (15 épocas, EarlyStopping,
ReduceLROnPlateau, class weights). Al terminar genera:

```text
trained_models/product_classifier.keras
trained_models/training_report.json
trained_models/accuracy_loss.png
trained_models/confusion_matrix.png
```

**Resultado obtenido en la versión actual:**

```text
Test accuracy : 0.7299  (≈ 73 %)
Test loss     : 0.8321
```

---

## Exportar el modelo a ONNX

Después de entrenar, convertir el modelo al formato ONNX para que la API pueda
usarlo sin cargar TensorFlow:

```powershell
python scripts/export_to_onnx.py
```

Genera:

```text
trained_models/product_classifier.onnx  (≈ 9.8 MB)
```

> Este paso solo es necesario la primera vez o cada vez que se reentrene el modelo.

---

## Migrar datos de SQLite a PostgreSQL (opcional)

Si ya existen registros en la base de datos local (`app/database/inventory.db`),
se pueden migrar a PostgreSQL con:

```powershell
python scripts/migrate_to_postgres.py
```

El script lee todos los registros de SQLite e inserta los que no existan en PostgreSQL
usando `ON CONFLICT DO NOTHING`.

---

## Ejecutar el sistema

Puedes ejecutarlo con Docker Compose en una sola terminal o manualmente con dos
terminales abiertas de forma simultánea.

### Opcion recomendada - Docker Compose

Con Docker solo necesitas una terminal. Asegurate de tener el archivo `.env` en la
raiz del proyecto con `DATABASE_URL` configurada y de que existan:

```text
trained_models/product_classifier.onnx
datasets/processed/labels.json
```

Levanta API Flask y Streamlit juntos:

```powershell
docker compose up --build
```

Servicios disponibles:

- API Flask: `http://localhost:5000`
- Streamlit: `http://localhost:8501`

Para detener todo:

```powershell
docker compose down
```

### Opcion manual - dos terminales

### Terminal 1 — API Flask

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.api.app
```

La API queda disponible en `http://127.0.0.1:5000`.

### Terminal 2 — Interfaz Streamlit

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app/main.py
```

La interfaz se abre automáticamente en `http://localhost:8501`.

---

## Uso del sistema

La interfaz Streamlit tiene cuatro pestañas:

### Pestaña 1 — Clasificar producto

1. Seleccionar método de entrada: **Subir imagen** o **Usar cámara**.
2. Cargar o capturar la imagen del producto.
3. Presionar **Clasificar producto**.
4. Si el modelo no puede clasificar la imagen con suficiente confianza (< 70 %), se
   muestra una advertencia y se solicita una nueva imagen.
5. Revisar categoría predicha, confianza y código generado.
6. Ajustar nombre del producto, cantidad y categoría si es necesario.
7. Presionar **Guardar en inventario**.

### Pestaña 2 — Cargue masivo

1. Elegir el modo: **Subir imágenes** (múltiples archivos) o **Subir ZIP**.
2. Cargar los archivos (formatos válidos: `jpg`, `jpeg`, `png`, `webp`).
3. Presionar **Clasificar todo** — el sistema procesa hasta 100 imágenes.
4. Revisar la tabla de resultados con categoría, confianza y estado por archivo.
   Las imágenes con confianza < 70 % se marcan como "No clasificable" y se omiten
   al guardar.
5. Descargar los resultados en Excel con **Exportar resultados a Excel**.
6. Guardar todos los productos válidos con **Guardar todo en inventario**.

### Pestaña 3 — Inventario

Muestra todos los productos registrados. Permite exportar el inventario completo
en formato **Excel** o **PDF** y tiene botón **Actualizar inventario**.

### Pestaña 4 — Acerca del modelo

Información del modelo entrenado: arquitectura, categorías, archivos generados
y estado del reporte de entrenamiento.

---

## Endpoints de la API

### `GET /health`

Verifica que la API esté activa.

```powershell
curl.exe http://127.0.0.1:5000/health
```

```json
{ "status": "ok", "message": "API funcionando correctamente" }
```

---

### `POST /predict`

Clasifica una imagen individual y devuelve la predicción junto con un código
de inventario generado.

**Form-data:** campo `image` con el archivo de imagen.

```powershell
curl.exe -X POST -F "image=@ruta/imagen.jpg" http://127.0.0.1:5000/predict
```

```json
{
  "prediction": {
    "predicted_category": "cafe_chocolate",
    "confidence": 0.9976,
    "confidence_percent": 99.76,
    "top_predictions": [
      { "category": "cafe_chocolate", "confidence": 0.9976, "confidence_percent": 99.76 },
      { "category": "salsas_y_condimentos", "confidence": 0.0016, "confidence_percent": 0.16 },
      { "category": "azucar_sal", "confidence": 0.0004, "confidence_percent": 0.04 }
    ]
  },
  "generated_code": "INV-CAF-202605-0001"
}
```

---

### `POST /predict/batch`

Clasifica múltiples imágenes en una sola llamada.
Acepta **uno** de los dos campos (no ambos a la vez):

- `files`: uno o más archivos de imagen.
- `zip`: un archivo `.zip` con imágenes (se procesan subdirectorios de forma recursiva).

**Límite:** 100 imágenes por request.

```powershell
# Con archivos sueltos
curl.exe -X POST `
  -F "files=@imagen1.jpg" `
  -F "files=@imagen2.jpg" `
  http://127.0.0.1:5000/predict/batch

# Con ZIP
curl.exe -X POST -F "zip=@productos.zip" http://127.0.0.1:5000/predict/batch
```

```json
{
  "total_received": 2,
  "total_processed": 2,
  "total_skipped": 0,
  "results": [
    {
      "filename": "imagen1.jpg",
      "predicted_category": "cafe_chocolate",
      "confidence": 0.9976,
      "confidence_percent": 99.76,
      "top_predictions": [...],
      "generated_code": "INV-CAF-202605-0004"
    },
    {
      "filename": "imagen2.jpg",
      "predicted_category": "aceites",
      "confidence": 0.9243,
      "confidence_percent": 92.43,
      "top_predictions": [...],
      "generated_code": "INV-ACE-202605-0005"
    }
  ]
}
```

Si una imagen no puede procesarse, el resultado incluye `"error": "descripción"` en
lugar de los campos de predicción, y el proceso continúa con las demás.

---

### `POST /products`

Registra un producto en inventario.

```powershell
$body = @{
  code       = "INV-CAF-202605-0001"
  name       = "Nescafe Tradicion 170g"
  category   = "cafe_chocolate"
  quantity   = 2
  image_path = "imagen.jpg"
  confidence = 0.9976
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/products" `
  -Method POST -ContentType "application/json" -Body $body
```

---

### `GET /products`

Lista todos los productos registrados.

```powershell
curl.exe http://127.0.0.1:5000/products
```

---

### `GET /products/<code>`

Busca un producto por su código único.

```powershell
curl.exe http://127.0.0.1:5000/products/INV-CAF-202605-0001
```

---

## Scripts disponibles

| Script | Descripción |
|--------|-------------|
| `prepare_hf_dataset.py` | Descarga y organiza el dataset desde Hugging Face |
| `train_model.py` | Entrena el clasificador MobileNetV2 |
| `export_to_onnx.py` | Convierte `product_classifier.keras` → `product_classifier.onnx` |
| `migrate_to_postgres.py` | Migra registros de SQLite a PostgreSQL |
| `predict_image.py` | Clasifica una imagen desde la consola |
| `predict_and_register.py` | Clasifica y registra en BD desde la consola |
| `list_inventory.py` | Lista el inventario desde la consola |

---

## Categorías y códigos de inventario

El modelo clasifica productos en siete categorías:

| Categoría técnica | Nombre visible | Prefijo de código |
|-------------------|---------------|-------------------|
| `arroz_y_granos` | Arroz y granos | `ARG` |
| `pastas` | Pastas | `PAS` |
| `aceites` | Aceites | `ACE` |
| `salsas_y_condimentos` | Salsas y condimentos | `SAL` |
| `cafe_chocolate` | Café y chocolate | `CAF` |
| `enlatados` | Enlatados | `ENL` |
| `azucar_sal` | Azúcar y sal | `AZS` |

**Formato del código de inventario:**

```text
INV-{PREFIJO}-{YYYYMM}-{CONSECUTIVO}

Ejemplo: INV-CAF-202605-0001
```

**Regla de confianza:**

| Confianza | Interpretación |
|-----------|---------------|
| ≥ 70 % | Predicción confiable — se acepta para guardar |
| < 70 % | Confianza insuficiente — la imagen no se clasifica; debe reemplazarse |

---

## Archivos excluidos del repositorio

Los siguientes archivos **no se suben a GitHub** y deben generarse localmente:

```text
.venv/                              ← entorno virtual
.env                                ← credenciales de BD
datasets/processed/train/           ← imágenes de entrenamiento
datasets/processed/validation/      ← imágenes de validación
datasets/processed/test/            ← imágenes de prueba
datasets/processed/otros/           ← imágenes sin clasificar
trained_models/product_classifier.keras  ← modelo Keras (grande)
```

Los archivos `.onnx`, reportes `.json` y gráficas `.png` **sí se suben**.

---

## Problemas comunes

### La API no arranca / error de importación

Ejecutar siempre desde la raíz del proyecto usando el módulo:

```powershell
python -m app.api.app
```

No usar `python app/api/app.py` directamente.

### El modelo ONNX no existe

Si falta `trained_models/product_classifier.onnx`, primero entrenar el modelo y
luego exportarlo:

```powershell
python scripts/train_model.py
python scripts/export_to_onnx.py
```

### La base de datos no conecta

1. Verificar que el archivo `.env` existe en la raíz del proyecto.
2. Verificar que `DATABASE_URL` tiene `?sslmode=require` al final (requerido por Supabase).
3. Verificar que la tabla `products` fue creada en Supabase.

### Error al activar el entorno virtual

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### El entorno usa Python incorrecto

Verificar la versión:

```powershell
python --version
```

Si no muestra `Python 3.11.x`, recrear el entorno:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### El modelo clasifica mal algunas imágenes

El modelo tiene una precisión aproximada del 73 %. Para mejores resultados:

- Usar imágenes con buena iluminación.
- Centrar el producto en el encuadre.
- Evitar fondos muy cargados.
- Corregir la categoría manualmente en Streamlit si es necesario.

---

## Resumen rápido para nuevos integrantes

```powershell
# 1. Clonar y entrar al proyecto
git clone URL_DEL_REPOSITORIO
cd programacion-avanzada-trabajo-final

# 2. Entorno virtual
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Configurar .env con DATABASE_URL de Supabase
# (crear el archivo manualmente)

# 4. Preparar dataset
python scripts/prepare_hf_dataset.py --clean-output

# 5. Entrenar y exportar modelo
python scripts/train_model.py
python scripts/export_to_onnx.py

# 6. Ejecutar el sistema (dos terminales)
python -m app.api.app          # Terminal 1
streamlit run streamlit_app/main.py  # Terminal 2
```

---

## Autores

Proyecto desarrollado para el curso de **Programación Avanzada**.

```text
Nombre Integrante 1
Nombre Integrante 2
Nombre Integrante 3
```

# Sistema Inteligente de Clasificación e Inventariado de Productos de Despensa

## Descripción

Este proyecto es un sistema inteligente de inventario capaz de clasificar productos de despensa mediante imágenes, generar automáticamente un código único para cada producto y registrarlo en una base de datos SQLite.

El sistema utiliza un modelo de inteligencia artificial entrenado con TensorFlow/Keras y MobileNetV2. Además, cuenta con una API desarrollada en Flask y una interfaz visual desarrollada en Streamlit.

El usuario puede subir una imagen o tomar una foto con la cámara, obtener la categoría predicha del producto, confirmar los datos y guardar el producto en el inventario.

---

## Tecnologías utilizadas

- Python 3.11
- TensorFlow / Keras
- MobileNetV2
- Flask
- Flask-CORS
- Streamlit
- SQLite
- Pillow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Requests
- Hugging Face Datasets

---

## Categorías del modelo

El modelo clasifica productos de despensa en las siguientes categorías:

```text
arroz_y_granos
pastas
aceites
salsas_y_condimentos
cafe_chocolate
enlatados
azucar_sal
```

---

## Estructura del proyecto

```text
programacion avanzada- proyecto final/
│
├── app/
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── prediction_utils.py
│   │
│   ├── database/
│   │   └── inventory.db
│   │
│   └── services/
│       ├── __init__.py
│       ├── code_generator.py
│       └── inventory_service.py
│
├── datasets/
│   └── processed/
│       ├── labels.json
│       ├── dataset_report.json
│       └── dataset_report.md
│
├── scripts/
│   ├── prepare_hf_dataset.py
│   ├── train_model.py
│   ├── predict_image.py
│   ├── predict_and_register.py
│   └── list_inventory.py
│
├── streamlit_app/
│   └── main.py
│
├── trained_models/
│   ├── training_report.json
│   ├── accuracy_loss.png
│   └── confusion_matrix.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Archivos que no se suben al repositorio

Por tamaño y buenas prácticas, algunos archivos no se suben a GitHub:

```text
.venv/
datasets/processed/train/
datasets/processed/validation/
datasets/processed/test/
datasets/processed/otros/
trained_models/product_classifier.keras
```

Cada integrante debe generar esos archivos en su propio computador ejecutando los comandos indicados en este documento.

---

## Requisitos previos

Antes de ejecutar el proyecto, cada integrante debe tener instalado:

- Python 3.11
- Git
- Visual Studio Code

---

## 1. Instalar Python 3.11

Este proyecto debe ejecutarse con **Python 3.11**.

No se recomienda usar Python 3.14 porque TensorFlow no tiene soporte compatible para esa versión.

Para verificar las versiones instaladas, ejecutar en PowerShell:

```powershell
py -0
```

Debe aparecer algo parecido a:

```text
-3.11-64
```

Si no aparece Python 3.11, se puede instalar desde terminal usando `winget`:

```powershell
winget install Python.Python.3.11
```

Durante la instalación, aceptar los términos si la terminal lo solicita.

Después de instalar, cerrar y abrir nuevamente Visual Studio Code.

Verificar otra vez:

```powershell
py -0
```

---

## 2. Clonar el repositorio

Cada integrante debe clonar el proyecto desde GitHub:

```powershell
git clone URL_DEL_REPOSITORIO
```

Entrar a la carpeta del proyecto:

```powershell
cd NOMBRE_DEL_PROYECTO
```

Ejemplo:

```powershell
cd "programacion avanzada- proyecto final"
```

> Nota: reemplazar `URL_DEL_REPOSITORIO` por la URL real del repositorio y `NOMBRE_DEL_PROYECTO` por el nombre real de la carpeta.

---

## 3. Crear entorno virtual

Desde la raíz del proyecto, crear el entorno virtual con Python 3.11:

```powershell
py -3.11 -m venv .venv
```

Esto crea una carpeta llamada:

```text
.venv/
```

Esa carpeta no se sube a GitHub.

---

## 4. Activar entorno virtual

En PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si se activó correctamente, la terminal debe mostrar algo así:

```text
(.venv) PS C:\ruta\del\proyecto>
```

---

## 5. Si PowerShell no permite activar el entorno

Si aparece un error de permisos o ejecución de scripts, ejecutar:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Cuando pregunte, responder:

```text
Y
```

Luego volver a activar el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 6. Verificar versión de Python dentro del entorno

Con el entorno activo:

```powershell
python --version
```

Debe salir algo como:

```text
Python 3.11.x
```

Si sale Python 3.14, el entorno fue creado con la versión incorrecta. En ese caso, borrar `.venv/` y crearlo otra vez usando:

```powershell
py -3.11 -m venv .venv
```

---

## 7. Instalar dependencias

Con el entorno virtual activo:

```powershell
python -m pip install --upgrade pip
```

Luego instalar todas las librerías del proyecto:

```powershell
python -m pip install -r requirements.txt
```

Si por alguna razón falta alguna dependencia, se pueden instalar manualmente:

```powershell
python -m pip install tensorflow matplotlib numpy scikit-learn datasets pillow pandas tqdm flask flask-cors streamlit requests
```

---

## 8. Verificar instalación de TensorFlow

Ejecutar:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Si muestra una versión de TensorFlow, la instalación está correcta.

Si aparece error indicando que TensorFlow no existe o no se encuentra una versión compatible, revisar que el entorno esté usando Python 3.11.

---

## 9. Preparar el dataset

El dataset completo no se sube al repositorio porque contiene muchas imágenes.

Cada integrante debe generarlo ejecutando el siguiente comando:

```powershell
python scripts/prepare_hf_dataset.py --image-column image --name-column name --category-column subcategory --no-streaming --clean-output
```

Este comando descarga el dataset desde Hugging Face y lo organiza en la estructura necesaria para entrenamiento.

Dataset utilizado:

```text
valentinafevu/productos-supermercado
```

Al finalizar, se deben crear estas carpetas:

```text
datasets/processed/train/
datasets/processed/validation/
datasets/processed/test/
datasets/processed/otros/
```

Y estos archivos:

```text
datasets/processed/labels.json
datasets/processed/dataset_report.json
datasets/processed/dataset_report.md
```

---

## 10. Resultado esperado del dataset

El procesamiento debe generar aproximadamente:

```text
Total imágenes procesadas: 2487
Total imágenes descartadas: 19
```

Distribución aproximada:

```text
arroz_y_granos: 528
pastas: 204
aceites: 127
salsas_y_condimentos: 322
cafe_chocolate: 693
enlatados: 217
azucar_sal: 396
```

División aproximada:

```text
Train: 1740
Validation: 373
Test: 374
```

Estos valores pueden variar ligeramente si el dataset cambia.

---

## 11. Entrenar el modelo

Después de preparar el dataset, entrenar el modelo:

```powershell
python scripts/train_model.py
```

Este script realiza:

- Carga de imágenes desde `train`, `validation` y `test`.
- Uso de MobileNetV2 con Transfer Learning.
- Data augmentation.
- Class weights.
- Entrenamiento del modelo.
- Evaluación con test.
- Generación de matriz de confusión.
- Generación de reporte de clasificación.
- Guardado del modelo.

Al terminar, se debe crear:

```text
trained_models/product_classifier.keras
trained_models/training_report.json
trained_models/accuracy_loss.png
trained_models/confusion_matrix.png
```

El archivo más importante es:

```text
trained_models/product_classifier.keras
```

Ese archivo contiene el modelo entrenado.

---

## 12. Resultado esperado del entrenamiento

En la versión desarrollada, el modelo obtuvo aproximadamente:

```text
Test accuracy: 0.7299
Test loss: 0.8321
```

Esto equivale a una precisión aproximada del:

```text
72.99%
```

El resultado puede variar ligeramente dependiendo del equipo y la ejecución.

---

## 13. Probar una predicción individual

Después de entrenar el modelo, se puede probar una imagen individual:

```powershell
python scripts/predict_image.py --image "ruta/a/imagen.jpg"
```

Ejemplo:

```powershell
python scripts/predict_image.py --image "datasets/processed/test/cafe_chocolate/imagen.jpg"
```

Salida esperada:

```text
Categoría predicha: cafe_chocolate
Confianza: 99.26%

Top predicciones:
1. cafe_chocolate - 99.26%
2. salsas_y_condimentos - 0.43%
3. pastas - 0.21%
```

---

## 14. Predecir y registrar producto localmente

También se puede predecir una imagen y registrarla directamente en SQLite:

```powershell
python scripts/predict_and_register.py --image "ruta/a/imagen.jpg" --name "Nombre del producto" --quantity 1
```

Ejemplo:

```powershell
python scripts/predict_and_register.py --image "datasets/processed/test/cafe_chocolate/imagen.jpg" --name "Nescafe Tradicion 170g" --quantity 5
```

Esto genera un código como:

```text
INV-CAF-202605-0001
```

Y registra el producto en:

```text
app/database/inventory.db
```

---

## 15. Listar inventario desde terminal

Para ver los productos registrados:

```powershell
python scripts/list_inventory.py
```

Salida esperada:

```text
Inventario registrado

ID: 1
Código: INV-CAF-202605-0001
Nombre: Nescafe Tradicion 170g
Categoría: cafe_chocolate
Cantidad: 5
Confianza: 99.26%
Estado: activo
```

---

## 16. Ejecutar la API Flask

Para usar la API, ejecutar:

```powershell
python -m app.api.app
```

Debe aparecer:

```text
Running on http://127.0.0.1:5000
```

No cerrar esta terminal mientras se use la aplicación.

---

## 17. Probar API en navegador

Abrir:

```text
http://127.0.0.1:5000/health
```

Debe mostrar:

```json
{
  "message": "API funcionando correctamente",
  "status": "ok"
}
```

Para ver productos registrados:

```text
http://127.0.0.1:5000/products
```

---

## 18. Ejecutar la interfaz Streamlit

Abrir una segunda terminal.

Activar el entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ejecutar Streamlit:

```powershell
streamlit run streamlit_app/main.py
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```

---

## 19. Uso correcto del sistema completo

Para usar el sistema completo se necesitan dos terminales abiertas.

### Terminal 1: API Flask

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.api.app
```

### Terminal 2: Streamlit

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app/main.py
```

---

## 20. Flujo de uso en Streamlit

En la interfaz:

1. Entrar a la pestaña **Clasificar producto**.
2. Seleccionar método de entrada:
   - Subir imagen.
   - Usar cámara.
3. Cargar o capturar una imagen.
4. Presionar **Clasificar producto**.
5. Revisar:
   - Categoría predicha.
   - Confianza.
   - Código generado.
   - Top predicciones.
6. Escribir o corregir:
   - Nombre del producto.
   - Cantidad.
   - Categoría.
7. Presionar **Guardar en inventario**.
8. Ir a la pestaña **Inventario** para consultar los registros.

---

## 21. Endpoints principales de la API

### GET `/health`

Verifica que la API esté activa.

```text
http://127.0.0.1:5000/health
```

### GET `/products`

Lista los productos registrados.

```text
http://127.0.0.1:5000/products
```

### POST `/predict`

Recibe una imagen y devuelve:

- Categoría predicha.
- Confianza.
- Top 3 predicciones.
- Código generado.

Ejemplo con PowerShell:

```powershell
curl.exe -X POST `
  -F "image=@ruta/a/imagen.jpg" `
  http://127.0.0.1:5000/predict
```

### POST `/products`

Registra un producto en inventario.

Ejemplo usando PowerShell:

```powershell
$body = @{
  code = "INV-CAF-202605-0002"
  name = "Nucita Nuggets Halloween 240g"
  category = "cafe_chocolate"
  quantity = 2
  image_path = "datasets/processed/test/cafe_chocolate/imagen.jpg"
  confidence = 0.8812
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/products" -Method POST -ContentType "application/json" -Body $body
```

---

## 22. Generación de códigos

Los códigos siguen esta estructura:

```text
INV-CAT-YYYYMM-0001
```

Ejemplo:

```text
INV-CAF-202605-0001
```

Prefijos usados:

```text
ARG = arroz_y_granos
PAS = pastas
ACE = aceites
SAL = salsas_y_condimentos
CAF = cafe_chocolate
ENL = enlatados
AZS = azucar_sal
```

---

## 23. Base de datos

La base de datos usada es SQLite.

Archivo:

```text
app/database/inventory.db
```

Tabla principal:

```text
products
```

Campos:

```text
id
code
name
category
quantity
image_path
confidence
created_at
status
```

---

## 24. Regla de confianza

El sistema interpreta la confianza así:

```text
Confianza >= 80%
Predicción confiable.

Confianza entre 50% y 79%
Predicción con confianza media. Se recomienda revisar antes de guardar.

Confianza < 50%
Predicción con baja confianza. Se recomienda corregir manualmente.
```

---

## 25. Problemas comunes

### TensorFlow no se instala

Probablemente se está usando Python 3.14.

Solución:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### No se reconoce el comando pip

Usar:

```powershell
python -m pip install -r requirements.txt
```

en lugar de:

```powershell
pip install -r requirements.txt
```

### La API Flask no aparece activa en Streamlit

Verificar que en una terminal esté corriendo:

```powershell
python -m app.api.app
```

Luego recargar Streamlit.

### Error: `app is not a package`

Ejecutar Flask así:

```powershell
python -m app.api.app
```

No usar:

```powershell
python app/api/app.py
```

Además verificar que existan:

```text
app/__init__.py
app/api/__init__.py
app/services/__init__.py
```

### Error al activar entorno virtual

Ejecutar:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Luego activar:

```powershell
.\.venv\Scripts\Activate.ps1
```

### El modelo se equivoca en algunas imágenes

Esto es normal. El modelo obtuvo una precisión aproximada del 72.99%.

Se recomienda:

- Usar imágenes claras.
- Evitar fondos con muchos objetos.
- Usar buena iluminación.
- Centrar el producto.
- Corregir la categoría manualmente en Streamlit si es necesario.

---

## 26. Archivos importantes

### Código principal

```text
app/api/app.py
app/api/prediction_utils.py
app/services/code_generator.py
app/services/inventory_service.py
streamlit_app/main.py
```

### Scripts de apoyo

```text
scripts/prepare_hf_dataset.py
scripts/train_model.py
scripts/predict_image.py
scripts/predict_and_register.py
scripts/list_inventory.py
```

### Archivos generados

```text
datasets/processed/
trained_models/
app/database/inventory.db
```

---

## 27. Resumen rápido para integrantes del equipo

Después de clonar el proyecto, cada integrante debe ejecutar:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/prepare_hf_dataset.py --image-column image --name-column name --category-column subcategory --no-streaming --clean-output
python scripts/train_model.py
python -m app.api.app
```

En otra terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app/main.py
```

---

## Estado final del proyecto

El proyecto cuenta con:

- Dataset procesado.
- Modelo entrenado.
- Predicción individual.
- Generación de códigos.
- Base de datos SQLite.
- API Flask.
- Interfaz Streamlit.
- Carga de imagen.
- Captura por cámara.
- Registro de productos.
- Consulta de inventario.

El sistema está funcional como una primera versión completa para clasificación e inventariado de productos de despensa.

---

## Mejoras futuras

Se pueden implementar las siguientes mejoras:

- Mejorar el dataset.
- Agregar más imágenes propias.
- Balancear mejor las clases.
- Aplicar fine-tuning al modelo.
- Exportar inventario a Excel.
- Agregar edición y eliminación de productos.
- Generar códigos QR.
- Leer códigos de barras.
- Agregar autenticación.
- Usar PostgreSQL.
- Desplegar en la nube.

---

## Autores

Proyecto desarrollado por:

```text
Nombre Integrante 1
Nombre Integrante 2
Nombre Integrante 3
```

Curso:

```text
Programación Avanzada
```

---

## Conclusión

Este proyecto integra inteligencia artificial, procesamiento de imágenes, API REST, base de datos e interfaz gráfica para construir un sistema inteligente de inventario.

El sistema permite clasificar productos de despensa mediante imágenes, generar códigos únicos y registrar productos dentro de un inventario empresarial de forma automatizada.

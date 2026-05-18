# Reporte de preparacion del dataset

- Dataset usado: `valentinafevu/productos-supermercado`
- Fecha de procesamiento: `2026-05-14T22:45:23.238576+00:00`
- Metodo de division: `sklearn_estratificado`

## Inspeccion inicial
- Total de registros: 7449
- Splits originales: `{'train': 7449}`
- Columnas encontradas: `['image', 'name', 'supermarket_category', 'main_category', 'subcategory', 'full_image_path', 'ai_generated_description']`
- Columna de imagen: `image`
- Columna de nombre/producto: `name`
- Columna de categoria/etiqueta: `subcategory`

## Categorias finales
- arroz_y_granos: 528
- pastas: 204
- aceites: 127
- salsas_y_condimentos: 322
- cafe_chocolate: 693
- enlatados: 217
- azucar_sal: 396

## Division final
- train: 1740
- validation: 373
- test: 374

## Criterios de clasificacion
Primero se uso la categoria original del dataset cuando podia mapearse a una de las cinco categorias objetivo. Cuando no habia una categoria clara, se infirio la categoria usando palabras clave normalizadas del nombre del producto.

- arroz_y_granos: arroz, frijol, frijoles, lenteja, lentejas, garbanzo, garbanzos, grano, granos, maiz, maíz, quinua, quinoa, cereal, avena
- pastas: pasta, pastas, spaghetti, espagueti, espaguetis, macarron, macarrones, fideo, fideos, lasagna, lasaña
- aceites: aceite, aceites, oliva, girasol, canola, vegetal
- salsas_y_condimentos: salsa, salsas, mayonesa, mostaza, tomate, ketchup, condimento, condimentos, caldo, consome, consomé, vinagre, adobo, sazonador, sal de ajo
- cafe_chocolate: cafe, café, chocolate, cocoa, chocolisto, milo, instantaneo, instantáneo, bebida achocolatada
- enlatados: atun, atún, sardina, sardinas, enlatado, enlatados, lata, maiz tierno, maíz tierno, arveja, arvejas, conserva, conservas
- azucar_sal: azucar, azúcar, sal, panela, endulzante, stevia, edulcorante

## Productos no clasificados
- Total enviados a otros: 4962
- Imagenes guardadas en otros: 4943

Ejemplos:
- split=train, index=673, nombre=Pancake Aunt Jemima Solo Leche Doypack x 600g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=674, nombre=Harina de trigo Haz De Oros tradicional x2500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=677, nombre=Harina de trigo Haz De Oros tradicional x500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=678, nombre=Harina de trigo La Nieve x500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=679, nombre=Pancake Aunt Jemima Original Doypack x 600g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=680, nombre=Harina PAN integral semillas nutritivas x500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=683, nombre=Harina Why Not almendras libre gluten x250g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=685, nombre=Premezcla Alcaguete waffles pandeyuca sin gluten x400g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=687, nombre=Mezcla Haz De Oros pandebono x300g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=689, nombre=Pancake Aunt Jemima Solo Leche Doypack x 600g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=690, nombre=Harina de trigo Haz De Oros tradicional x2500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=693, nombre=Harina de trigo Haz De Oros tradicional x500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=694, nombre=Harina de trigo La Nieve x500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=695, nombre=Pancake Aunt Jemima Original Doypack x 600g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=696, nombre=Mezcla Pan para arepas de choclo x850g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=697, nombre=Harina de trigo Haz De Oros tradicional x1000g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=699, nombre=Mezcla lista para pancakes crepes y waffles Haz de oros x 600 g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=700, nombre=Harina de trigo Haz De Oros con polvo para hornear x1000g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=703, nombre=Harina Promasa precocida blanca x1000g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=704, nombre=Harina PAN integral semillas nutritivas x500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=707, nombre=Harina Why Not almendras libre gluten x250g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=709, nombre=Premezcla Alcaguete waffles pandeyuca sin gluten x400g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=711, nombre=Mezcla Haz De Oros pandebono x300g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=714, nombre=Pancake Aunt Jemima Original Doypack x 300g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=715, nombre=Harina de trigo Harina Haz De Oros x1000g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=716, nombre=Harina Promasa precocida amarilla x1000g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=720, nombre=Harina precocida blanca x500g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=721, nombre=Mezcla Haz de Oros Almojábanas x250grs, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=725, nombre=Carve Noel x330g, categoria_original=harinas-y-mezclas-para-preparar
- split=train, index=729, nombre=Mezcla Lista Para Pancakes Crepes y Waffles Light Haz de oros x 300g, categoria_original=harinas-y-mezclas-para-preparar

## Descartes y advertencias
- Total de imagenes descartadas: 19
- otros_imagen_danada_o_error_guardado: 19
- No quedaron categorias vacias.

## Primeros ejemplos inspeccionados
### Ejemplo 1
```json
{
  "image": "Image(path=C:/Users/User/.cache/huggingface/hub/datasets--valentinafevu--productos-supermercado/snapshots/094c86cf43c426d6b56d71616b1edc4c017faefc/train\\0.webp)",
  "name": "Arroz Diana blanco x10kg ",
  "supermarket_category": "supermercado",
  "main_category": "despensa",
  "subcategory": "arroz-y-granos",
  "full_image_path": "dataset/train/0.webp",
  "ai_generated_description": "El producto es azúcar en forma de perlas, presentado en una bolsa con capacidad para 10 kg. La forma de las perlas de azúcar es esférica y uniforme, lo que facilita su uso en diversas aplicaciones culinarias y de repostería. La cantidad de 10 kg indica que es un paquete grande, adecuado para uso int..."
}
```
### Ejemplo 2
```json
{
  "image": "Image(path=C:/Users/User/.cache/huggingface/hub/datasets--valentinafevu--productos-supermercado/snapshots/094c86cf43c426d6b56d71616b1edc4c017faefc/train\\1.webp)",
  "name": "Arroz Diana blanco x5kg ",
  "supermarket_category": "supermercado",
  "main_category": "despensa",
  "subcategory": "arroz-y-granos",
  "full_image_path": "dataset/train/1.webp",
  "ai_generated_description": "El producto es un paquete que contiene arroz blanco. El paquete tiene una forma rectangular y contiene 5 kilogramos de arroz. El arroz está presentado en una imagen grande en el paquete, lo que sugiere que es el contenido principal. No hay información adicional sobre el tipo de arroz o su procesamie..."
}
```
### Ejemplo 3
```json
{
  "image": "Image(path=C:/Users/User/.cache/huggingface/hub/datasets--valentinafevu--productos-supermercado/snapshots/094c86cf43c426d6b56d71616b1edc4c017faefc/train\\2.webp)",
  "name": "Arroz Sonora x10kg ",
  "supermarket_category": "supermercado",
  "main_category": "despensa",
  "subcategory": "arroz-y-granos",
  "full_image_path": "dataset/train/2.webp",
  "ai_generated_description": "El producto es arroz, presentado en una bolsa. La cantidad de arroz en la bolsa es de un kilogramo. La forma del arroz es granular, típica de este tipo de producto. El tamaño de los granos es uniforme, lo que sugiere que es un arroz de buena calidad. La bolsa contiene una cantidad considerable de ar..."
}
```
### Ejemplo 4
```json
{
  "image": "Image(path=C:/Users/User/.cache/huggingface/hub/datasets--valentinafevu--productos-supermercado/snapshots/094c86cf43c426d6b56d71616b1edc4c017faefc/train\\3.webp)",
  "name": "Arroz Diana blanco x3kg ",
  "supermarket_category": "supermercado",
  "main_category": "despensa",
  "subcategory": "arroz-y-granos",
  "full_image_path": "dataset/train/3.webp",
  "ai_generated_description": "El producto es un paquete que contiene arroz blanco. El arroz está presentado en una sola porción dentro del paquete, lo que sugiere que se trata de una cantidad considerable. El tamaño del paquete indica que contiene 3000 gramos de arroz. La forma en que se presenta el arroz, en una gran masa blanc..."
}
```
### Ejemplo 5
```json
{
  "image": "Image(path=C:/Users/User/.cache/huggingface/hub/datasets--valentinafevu--productos-supermercado/snapshots/094c86cf43c426d6b56d71616b1edc4c017faefc/train\\4.webp)",
  "name": "Arroz Diana Premium blanco x4000g ",
  "supermarket_category": "supermercado",
  "main_category": "despensa",
  "subcategory": "arroz-y-granos",
  "full_image_path": "dataset/train/4.webp",
  "ai_generated_description": "El producto es arroz blanco contenido en una bolsa flexible. La cantidad de arroz en la bolsa es de 400 gramos. El arroz se presenta en granos blancos y alargados, característicos del arroz blanco comúnmente consumido. La bolsa permite almacenar y conservar el arroz en su interior."
}
```

# Inteligencia Artificial Explicable para la Detección y Análisis de Hallazgos Mamográficos como apoyo al Diagnóstico de Cáncer de Mama

Codigos desarrollados dentro del transcurso de la tesis para reproducir los experimentos realizados.

## Organización
El repositorio está organizado de manera de poder utilizarlo como una librería `src/correlax`, asi como tambien presentar codigos desarrollados dentro del proceso, y algunos notebooks desarrollados con distinto nivel de documentacion. (carpeta `unmaintained/`). 

## Requerimientos

### Dataset
Para el entrenamiento de los modelos desarollados se utilizó [VinDr-Mammo](https://vindr.ai/datasets/mammo).

### Codigo
Los requerimientos de librerías se estan manejando mediante [`uv`](https://astral.sh/uv).

```bash
# Clonar el repositorio
git clone https://github.com/dmelladoc/CodigoTesis.git

# cambiar la extensión del pyproject.toml en base a la GPU disponible
# cuda (NVidia GPU)
cp pyproject.toml.cuda pyproject.toml 
# xpu (Intel ARC)
cp pyproject.toml.xpu pyproject.toml

# Sincronizar y crear el ambiente con uv
uv sync

# Finalmente ejecutar un script
# Detector de hallazgos
uv run findclf_image.py [carpeta o imagenes] --roi
# entregar mascaras de hallazgos
uv run findclf_image.py [ruta/a/imagen.dcm] --roi --mask --threshold 0.25

# Detección + Explicaciones
uv run eval_correlax.py [carpeta o imagen] --roi
```

## Consideraciones

### Detector de RoI

- por temas de adaptabilidad, por defecto está desabilitado la deteccion de RoI. Para habilitarlo uno selecciona `--roi`. 
Para seleccionar mediante otsu, se agrega el argumento `--detector-type otsu` por defecto, tambien puede utilizar un detector Faster R-CNN `--detector-type fcrnn`

### Pytorch para GPU
Dentro de los requerimientos, utilizamos `pytorch` como framework para ML. 
Por limitaciones respecto a como detectar el hardware con `uv`, se optó por crear archivos `pyproject.toml` en base a las versiones de gpu probadas.

Actualmente tenemos 2 instalaciones configuradas:

- `xpu`: Intel ARC.
- `cuda`: Nvidia GPUs

Para instalar la versión de pytorch para una GPU en especifico, se debe modificar la extensión del archivo `pyproject.toml`.

Esto asegurará que se instale la versión específica de pytorch para la GPU a utilizar.

```bash
# Instalar la versión para NVIDIA
cp pyproject.toml.cuda pyproject.toml

# O para Intel ARC
cp pyproject.toml.xpu pyproject.toml

# Sincronizar paquetes
uv sync
```

Queda pendiente probar con Apple Silicon (`mps`)

### Carpeta `unmaintained`

Esta carpeta contiene mi código preliminar con el cual se realizaron las pruebas iniciales, y algunos experimentos.
Estos notebooks se encuentran en diferentes estados de documentación y algunas funciones fueron modificadas en el proceso de adaptación.
Estos se entregan a modo de evidencia para reproducir los experimentos, pero requieren algunas adaptaciones para hacerlos funcionar.

Entre algunas modificaciones necesarias se incluyen:

- funciones asociadas a `findclf` y `correlax`
- Algunos bloques se detallan en el código, pero estan implementados de manera interna en la librería.
- salidas de notebooks.

De manera proxima se proveerán versiones limpias de los notebooks: 

- [`ClassifierTrain.ipynb`](unmaintained/03-ClassifierTrain.ipynb): Se planifica un script de entrenamiento en base a ruta a dataset, reproducción del proceso de entrenamiento.
- [`ClassifierMetrics.ipynb`](unmaintained/04-ClassifierMetrics.ipynb): Se planifica un script de evaluación en base a ruta a dataset VinDrMammo, para obtener las métricas del clasificador de hallazgos.
- [`PointingGame.ipynb`](unmaintained/06C-PointingGameEval.ipynb): De mismo modo, poder aplicar pointing game en base a anotaciones de dataset VinDrMammo. Junto con extensión para ajuste de numero de ventanas (para acelerar proceso). Script requiere ~10 dias de ejecución en condiciones actuales.
- [`Corrmedia.ipynb`](unmaintained/Corrmedia.ipynb): Poder obtener las metricas a diferentes niveles de muestras, de la precisión numerica de CorRELAX. Adaptado a un script para probar con una o varias imagenes.
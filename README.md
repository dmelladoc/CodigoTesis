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
uv run findclf_image.py ruta/a/imagen.dcm --roi --mask --threshold 0.25

# Detección + Explicaciones
uv run eval_correlax.py
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
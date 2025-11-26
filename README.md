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

# Sincronizar y crear el ambiente con uv
uv sync

# Finalmente ejecutar un script
# Detector de hallazgos
uv run findclf_image.py [carpeta o imagenes] --roi
# entregar mascaras de hallazgos
uv run findflf_image.py ruta/a/imagen.dcm --roi --mask --threshold 0.25

# Detección + Explicaciones
uv run eval_correlax.py
```

Algunos consideraciones:
- por temas de adaptabilidad, por defecto está desabilitado la deteccion de RoI. Para habilitarlo uno selecciona `--roi`. 
Para seleccionar mediante otsu, se agrega el argumento `--detector-type otsu` por defecto, tambien puede utilizar un detector Faster R-CNN `--detector-type fcrnn`

**Nota**: Dentro de los requerimientos, utilizamos `pytorch` como framework para ML. 
Dentro del repositorio está por defecto la versión para aceleradores Intel ARC (Pytorch xpu).
Quedan pendientes la deteccion automática del acelerador disponible (CUDA, AppleSilicon, CPU)

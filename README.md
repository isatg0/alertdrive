# AlertDrive IA — Detección de somnolencia del conductor

Plataforma web que monitorea el rostro del conductor por webcam o video, detecta
signos de fatiga (cierre prolongado de ojos, bostezos, cabeceos) y genera alertas.

**Estado: Fase 1 completa** — estructura del repositorio y extracción de características.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/download_model.py   # descarga models/face_landmarker.task (~3.7 MB)
```

Python 3.10 o superior. MediaPipe 0.10.2x o superior.

## Uso

**1. Validar con tu webcam** (ver EAR, MAR, PERCLOS y pose en vivo):

```bash
python scripts/live_preview.py
python scripts/live_preview.py --source video.mp4
```

**2. Crear el manifiesto del dataset** (una fila por video con sujeto y etiqueta):

```bash
python -m alertdrive.data.manifest --dataset uta-rldd --root RUTA/UTA-RLDD --out manifests/uta.csv
python -m alertdrive.data.manifest --dataset yawdd    --root RUTA/YawDD    --out manifests/yawdd.csv
```

Revisa el resumen que imprime (videos, sujetos, conteo por etiqueta) contra la
documentación del dataset. Para otros datasets (p. ej. NTHU-DDD) se puede escribir
el manifiesto a mano con las columnas `video_path, subject_id, label, dataset,
frame_labels_path`; la última es opcional y apunta a la anotación por frame.

**3. Extraer características por ventana**:

```bash
# Prueba rápida: 3 videos, 60 s cada uno
python -m alertdrive.data.build_features --manifest manifests/uta.csv --out data/features/uta.csv \
    --sample-fps 5 --max-seconds 60 --limit 3

# Corrida completa en paralelo (reanudable si se interrumpe)
python -m alertdrive.data.build_features --manifest manifests/uta.csv --out data/features/uta.csv \
    --sample-fps 5 --workers 4
```

**4. Pruebas**:

```bash
pytest
```

La prueba de integración usa MediaPipe real. Para ejecutarla, pon cualquier foto
frontal de un rostro en `tests/assets/face.jpg`; sin ella se omite.

## Arquitectura

```
Video/Webcam ─► FaceLandmarkDetector ─► FrameFeatureExtractor ─► WindowAggregator ─► vector de características
                (MediaPipe, 478 pts,     (EAR, MAR, pose,          (PERCLOS, parpadeos/min,     │
                 pose, blendshapes)       líneas base, eventos)     bostezos, cabeceos...)     ▼
                                                                                  Clasificadores (Fase 2)
```

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Todos los umbrales en un solo lugar |
| `vision/face_landmarker.py` | Detección de rostro y landmarks (MediaPipe Tasks) |
| `vision/landmarks.py` | Índices de ojos y boca; estructura `FaceObservation` |
| `features/geometry.py` | EAR, MAR y ángulos de Euler |
| `features/temporal.py` | Línea base adaptativa y detector de eventos |
| `features/extractor.py` | Características por frame con estado |
| `features/window.py` | Características por ventana deslizante |
| `pipeline.py` | Video → filas de características |
| `data/manifest.py` | Adaptadores de datasets (UTA-RLDD, YawDD, CSV genérico) |
| `data/build_features.py` | Extracción por lotes, paralela y reanudable |

## Decisiones de diseño

- **Un solo pipeline para entrenar e inferir.** El mismo extractor procesa los
  datasets y la webcam, así el modelo desplegado ve exactamente las
  características con las que fue evaluado.
- **Umbrales relativos al conductor.** El EAR con ojos abiertos varía entre
  personas; en la foto de prueba fue 0.216, casi en el umbral fijo clásico de
  0.20. Por eso un ojo se considera cerrado si su EAR cae por debajo del 70 % del
  percentil 90 de los últimos 30 s. Con la pose se aplica la misma idea: se usa
  la desviación respecto a la mediana del conductor, no el ángulo absoluto.
- **Ventanas en lugar de frames.** Un parpadeo cierra los ojos pero no indica
  sueño. Los indicadores de la literatura (PERCLOS, frecuencia y duración de
  parpadeo, bostezos) son temporales. Por defecto: ventanas de 15 s cada 5 s.
- **Sujeto en cada fila.** Permite dividir train/test por persona en la Fase 2 y
  evitar fuga de datos.
- **API Tasks de MediaPipe.** La API antigua `mp.solutions` ya no existe en las
  versiones recientes. La nueva entrega además la matriz de pose de la cabeza.

## Características por ventana

`face_present_ratio`, `ear_mean/std/min`, `ear_rel_mean/min`, `perclos`,
`blink_rate_per_min`, `blink_dur_mean_s`, `long_closure_count`,
`microsleep_count`, `max_closure_s`, `mar_mean/max`, `mouth_open_fraction`,
`yawn_count`, `pitch_dev_mean/std`, `yaw_dev_std`, `roll_abs_mean`,
`head_off_fraction`, `looking_away_fraction`, `nod_count`,
`bs_eye_blink_mean`, `bs_jaw_open_max`.

## Limitaciones conocidas

- Las etiquetas de YawDD y UTA-RLDD son por video, no por instante: una ventana de
  un video "drowsy" puede contener momentos de alerta. Hay que declararlo en el
  informe.
- NTHU-DDD usa video infrarrojo; el rendimiento de MediaPipe sobre IR debe verificarse.
- Lentes de sol y poca luz degradan la detección de ojos.

## Hoja de ruta

- [x] Fase 1: estructura y extracción de características
- [ ] Fase 2: algoritmo A (reglas) vs. algoritmo B (ML), evaluación por sujeto
- [ ] Fase 3: plataforma web (Streamlit), alertas visuales y sonoras
- [ ] Fase 4: histórico de eventos (SQLite) y dashboard
- [ ] Fase 5: pruebas en condiciones reales e informe

# Fase 2 · Clasificación: algoritmo A (reglas) vs. algoritmo B (ML)

## Instalación

Agrega estas líneas a `requirements.txt` y reinstala:

```
scikit-learn>=1.3
matplotlib>=3.7
joblib>=1.3
```

```bash
pip install -r requirements.txt
pip install -e .
```

## Uso

**1. Probar el flujo sin dataset** (datos sintéticos; sus métricas no sirven para el informe):

```bash
python -m alertdrive.classify.synthetic --out data/features/synthetic.csv
python -m alertdrive.classify.train --features data/features/synthetic.csv --fast
```

**2. Prueba corta con el dataset real** (con el CSV de la corrida `--limit 3` basta):

```bash
python -m alertdrive.classify.train --features data/features/uta.csv --fast --folds 3 --no-final
```

**3. Corrida completa**:

```bash
python -m alertdrive.classify.train --features data/features/uta.csv --models rules rf svm --folds 5 --subsample 3
```

Se pueden pasar varios CSV (`--features uta.csv yawdd.csv`); la partición por
sujeto funciona igual porque los identificadores llevan prefijo del dataset.

Opciones útiles:

| Opción | Efecto |
|---|---|
| `--low-vigilant drop/positive/negative` | Qué hacer con la clase intermedia de UTA-RLDD (por defecto se descarta) |
| `--subsample 3` | Entrena con 1 de cada 3 ventanas por video. Con ventana de 15 s y paso de 5 s, así no se solapan. La prueba siempre usa todas |
| `--fast` | Rejilla de hiperparámetros reducida, para iterar rápido |
| `--no-final` | Sólo evalúa; no entrena ni guarda los modelos finales |

## Salidas

- `reports/fase2/resumen.md`: tablas listas para el informe (por ventana, por video y ΔF1 con IC 95 %).
- `reports/fase2/roc.png`, `confusion.png`, `importancia_rf.png`: figuras para el informe.
- `reports/fase2/metricas.json` y `metricas_por_particion.csv`: todos los números, incluidos los hiperparámetros elegidos en cada partición.
- `models/drowsiness_<modelo>.joblib`: modelos entrenados con todos los datos; `drowsiness_best.joblib` es el de mayor F1.

Conviene agregar `reports/` y `models/*.joblib` al `.gitignore`.

## Los algoritmos

**A · Reglas expertas** (`classify/rules.py`). Escala de puntos con umbrales de la literatura:

| Regla (ventana de 15 s) | Puntos |
|---|---|
| PERCLOS ≥ 0,15 | 2 |
| PERCLOS ≥ 0,30 | +1 |
| Microsueño (cierre ≥ 1 s) | 3 |
| Cierre prolongado (> 0,5 s) | 1 |
| Duración media de parpadeo ≥ 0,3 s | 1 |
| Bostezo | 1 (+1 si hay dos o más) |
| Cabeceo | 2 |
| Cabeza fuera de posición ≥ 30 % del tiempo | 1 |

Los umbrales y puntos son fijos. Sólo el puntaje mínimo para alertar se elige
con los datos de entrenamiento de cada partición (el que maximiza F1).

**B · Random Forest** y **B' · SVM RBF** (`classify/models.py`). Pipeline de
imputación por mediana → (estandarización para SVM) → clasificador, con
`class_weight="balanced"` y búsqueda de hiperparámetros anidada.

## Protocolo de evaluación (para el informe)

1. **Validación cruzada por sujeto** (`StratifiedGroupKFold`, 5 particiones): nadie aparece en entrenamiento y prueba a la vez.
2. **Hiperparámetros anidados** (`GroupKFold` interno de 3): la partición de prueba nunca se usa para elegir nada.
3. **Métricas**: exactitud, exactitud balanceada, precisión, recall, especificidad, F1 y ROC-AUC por partición (media ± desviación), más matriz de confusión y ROC con las predicciones fuera de partición.
4. **Por video**: un video es positivo si al menos la mitad de sus ventanas lo es. Es la vista que coincide con cómo etiquetan UTA-RLDD y YawDD.
5. **Comparación estadística**: IC 95 % de F1(B) − F1(A) con bootstrap de sujetos (2000 remuestreos). Se remuestrean sujetos y no ventanas porque las ventanas de una persona están correlacionadas.
6. **Importancia por permutación** del Random Forest, calculada sobre cada partición de prueba.
7. **Latencia** de inferencia por ventana, para justificar el uso en tiempo real.

## Uso en la plataforma (Fase 3)

```python
from alertdrive.classify.predictor import DrowsinessPredictor

predictor = DrowsinessPredictor.load("models/drowsiness_best.joblib")
result = predictor.predict(aggregator.compute(t_now))   # diccionario de la Fase 1
result.is_drowsy, result.score, result.reasons
```

`reasons` siempre trae las reglas activadas, aunque el modelo sea el Random
Forest, para mostrar al conductor el motivo de la alerta. Si la ventana tuvo
poco rostro visible, `valid` es `False` y no se declara somnolencia.

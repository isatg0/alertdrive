# Fase 5 · Evaluación de la plataforma completa

**Datos:** 24 videos, 12 sujetos (alerta: 12, somnoliento: 12), 4.0 h analizadas por modelo.

**Protocolo:** muestreo a 5 fps, predicción cada segundo sobre la ventana de 15 s, los primeros 15 s de cada video son calibración y no cuentan. Enfriamiento entre alarmas: 8 s.

## Alertas por video

Con la persistencia por defecto de la plataforma (3 s). Un video somnoliento se considera detectado si tiene al menos una alarma; la latencia se mide desde el fin de la calibración.

| Modelo | Videos somnolientos detectados | Latencia mediana (s) | Tiempo en alarma (somnolientos) | Videos alerta sin alarmas | Falsas alarmas por hora | Exactitud balanceada |
|---|---|---|---|---|---|---|
| Random Forest | 100% (12) | 3 | 80% | 0% (12) | 261.8 | 50% |
| Reglas expertas | 100% (12) | 6 | 57% | 17% (12) | 98.5 | 58% |
| SVM (kernel RBF) | 100% (12) | 3 | 89% | 0% (12) | 255.1 | 50% |

## Efecto de la persistencia

Más persistencia reduce las falsas alarmas a costa de alarmar más tarde. Ver `persistencia.png`.

| Modelo | Persistencia (s) | Detectados | Latencia mediana (s) | Falsas alarmas por hora |
|---|---|---|---|---|
| Random Forest | 1 | 100% | 1 | 279.8 |
| Random Forest | 2 | 100% | 2 | 271.9 |
| Random Forest | 3 | 100% | 3 | 261.8 |
| Random Forest | 4 | 100% | 4 | 256.2 |
| Random Forest | 5 | 100% | 5 | 251.7 |
| Random Forest | 7 | 100% | 8 | 241.5 |
| Random Forest | 10 | 100% | 11 | 228.0 |
| Reglas expertas | 1 | 100% | 1 | 109.8 |
| Reglas expertas | 2 | 100% | 5 | 103.0 |
| Reglas expertas | 3 | 100% | 6 | 98.5 |
| Reglas expertas | 4 | 100% | 7 | 91.8 |
| Reglas expertas | 5 | 100% | 8 | 87.8 |
| Reglas expertas | 7 | 100% | 10 | 70.4 |
| Reglas expertas | 10 | 100% | 16 | 55.7 |
| SVM (kernel RBF) | 1 | 100% | 1 | 266.9 |
| SVM (kernel RBF) | 2 | 100% | 2 | 260.7 |
| SVM (kernel RBF) | 3 | 100% | 3 | 255.1 |
| SVM (kernel RBF) | 4 | 100% | 4 | 251.7 |
| SVM (kernel RBF) | 5 | 100% | 5 | 245.5 |
| SVM (kernel RBF) | 7 | 100% | 8 | 224.1 |
| SVM (kernel RBF) | 10 | 100% | 11 | 210.0 |

## Predicción por segundo (sin lógica de alertas)

Sólo segundos con ventana completa y rostro visible. La etiqueta de cada segundo es la del video.

| Modelo | Segundos | Exactitud | Precisión | Recall | Especificidad | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Random Forest | 12910 | 0.649 | 0.612 | 0.831 | 0.463 | 0.705 | 0.664 |
| Reglas expertas | 12910 | 0.708 | 0.767 | 0.605 | 0.812 | 0.676 | 0.726 |
| SVM (kernel RBF) | 12910 | 0.698 | 0.642 | 0.908 | 0.483 | 0.752 | 0.706 |

## Rendimiento en este computador

| Etapa | ms por frame |
|---|---|
| Detección de rostro y landmarks (MediaPipe) | 80.0 |
| Características por frame y ventana | 2.67 |
| Total por frame (media) | 82.7 |
| Total por frame (percentil 95, peor video) | 172.8 |
| Clasificador (una predicción por segundo) | 228.90 |

Frames por segundo alcanzables: **12**. La plataforma necesita 5, así que tiene un margen de 2.4 veces sobre el tiempo real.

## Umbral de puntaje

En lugar de la decisión del modelo, se exige un puntaje mínimo para contar un segundo como fatiga (persistencia 3 s). Para cada modelo se recomienda el umbral con menos falsas alarmas que todavía detecta al menos el 90% de los videos somnolientos. Ver `umbral.png`.

| Modelo | Umbral | Detectados | Latencia mediana (s) | Videos alerta sin alarmas | Falsas alarmas por hora | Antes (criterio del modelo) |
|---|---|---|---|---|---|---|
| Random Forest | **0.91** | 92% | 167 | 50% | 39.4 | 261.8 falsas alarmas por hora |
| Reglas expertas | **0.154** | 100% | 10 | 42% | 50.7 | 98.5 falsas alarmas por hora |
| SVM (kernel RBF) | **0.829** | 92% | 27 | 33% | 127.8 | 255.1 falsas alarmas por hora |

El umbral se eligió con estos mismos sujetos, así que estas cifras son optimistas: preséntalas como análisis de sensibilidad, no como resultado de prueba. Además se calcularon con los modelos de `models_part1`; los reentrenados con más sujetos producen puntajes en una escala parecida pero no idéntica, así que el valor es un punto de partida que se confirma en las pruebas reales.

Para usarlo: en la plataforma, «Umbral de puntaje» en la barra lateral; en consola, `--threshold`.

## Archivos

- `por_video.csv`: métricas de cada video, modelo y persistencia.
- `umbral.csv`, `umbral.png`: todos los umbrales probados y el compromiso detección/falsas alarmas.
- `resumen_plataforma.csv`, `por_segundo.csv`, `rendimiento.csv`: las tablas de este reporte.
- `persistencia.png`: detección, latencia y falsas alarmas según la persistencia.

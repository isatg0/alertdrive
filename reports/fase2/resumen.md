# Fase 2 · Comparación de algoritmos

**Datos:** 4197 ventanas, 36 videos, 18 sujetos | positivas: 2105 (50.2%), negativas: 2092

**Protocolo:** validación cruzada de 5 particiones agrupadas por sujeto (StratifiedGroupKFold), búsqueda de hiperparámetros anidada con GroupKFold, low_vigilant = `drop`, submuestreo de entrenamiento = 3.

## Métricas por ventana (media ± desviación entre particiones)

| Algoritmo | Exactitud | Exactitud balanceada | Precisión | Recall (sensibilidad) | Especificidad | F1 | ROC-AUC | Latencia (ms/ventana) |
|---|---|---|---|---|---|---|---|---|
| A · Reglas expertas | 0.674 ± 0.040 | 0.677 ± 0.039 | 0.743 ± 0.055 | 0.552 ± 0.142 | 0.802 ± 0.083 | 0.622 ± 0.082 | 0.691 ± 0.044 | 0.002 |
| B · Random Forest | 0.684 ± 0.045 | 0.687 ± 0.044 | 0.690 ± 0.099 | 0.741 ± 0.138 | 0.633 ± 0.206 | 0.700 ± 0.034 | 0.785 ± 0.038 | 0.127 |
| B' · SVM (kernel RBF) | 0.698 ± 0.050 | 0.701 ± 0.049 | 0.712 ± 0.089 | 0.718 ± 0.132 | 0.684 ± 0.184 | 0.703 ± 0.047 | 0.752 ± 0.051 | 0.096 |

## Métricas por video (voto de mayoría de sus ventanas)

| Algoritmo | Exactitud | Precisión | Recall (sensibilidad) | Especificidad | F1 | ROC-AUC | Videos |
|---|---|---|---|---|---|---|---|
| A · Reglas expertas | 0.722 | 0.833 | 0.556 | 0.889 | 0.667 | 0.880 | 36 |
| B · Random Forest | 0.833 | 0.773 | 0.944 | 0.722 | 0.850 | 0.870 | 36 |
| B' · SVM (kernel RBF) | 0.806 | 0.789 | 0.833 | 0.778 | 0.811 | 0.821 | 36 |

## Diferencia de F1 frente al algoritmo A (bootstrap por sujeto)

| Algoritmo | ΔF1 | IC 95 % | Sujetos |
|---|---|---|---|
| B · Random Forest | +0.067 | [-0.009, +0.156] | 18 |
| B' · SVM (kernel RBF) | +0.072 | [-0.018, +0.167] | 18 |

Si el intervalo no contiene 0, la diferencia es significativa al 95 %.

**Modelo seleccionado para la plataforma:** B' · SVM (kernel RBF) (mayor F1 medio por sujeto).

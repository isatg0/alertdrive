# Fase 2 · Comparación de algoritmos

**Datos:** 1372 ventanas, 12 videos, 6 sujetos | positivas: 669 (48.8%), negativas: 703

**Protocolo:** validación cruzada de 5 particiones agrupadas por sujeto (StratifiedGroupKFold), búsqueda de hiperparámetros anidada con GroupKFold, low_vigilant = `drop`, submuestreo de entrenamiento = 3.

## Métricas por ventana (media ± desviación entre particiones)

| Algoritmo | Exactitud | Exactitud balanceada | Precisión | Recall (sensibilidad) | Especificidad | F1 | ROC-AUC | Latencia (ms/ventana) |
|---|---|---|---|---|---|---|---|---|
| A · Reglas expertas | 0.627 ± 0.058 | 0.625 ± 0.062 | 0.695 ± 0.133 | 0.464 ± 0.150 | 0.786 ± 0.130 | 0.542 ± 0.099 | 0.632 ± 0.078 | 0.007 |
| B · Random Forest | 0.746 ± 0.102 | 0.746 ± 0.102 | 0.769 ± 0.112 | 0.696 ± 0.183 | 0.797 ± 0.109 | 0.721 ± 0.132 | 0.784 ± 0.126 | 0.472 |
| B' · SVM (kernel RBF) | 0.736 ± 0.107 | 0.738 ± 0.104 | 0.743 ± 0.135 | 0.738 ± 0.170 | 0.738 ± 0.167 | 0.728 ± 0.118 | 0.805 ± 0.097 | 0.052 |

## Métricas por video (voto de mayoría de sus ventanas)

| Algoritmo | Exactitud | Precisión | Recall (sensibilidad) | Especificidad | F1 | ROC-AUC | Videos |
|---|---|---|---|---|---|---|---|
| A · Reglas expertas | 0.583 | 0.667 | 0.333 | 0.833 | 0.444 | 0.806 | 12 |
| B · Random Forest | 0.833 | 1.000 | 0.667 | 1.000 | 0.800 | 0.972 | 12 |
| B' · SVM (kernel RBF) | 0.833 | 0.833 | 0.833 | 0.833 | 0.833 | 0.972 | 12 |

## Diferencia de F1 frente al algoritmo A (bootstrap por sujeto)

| Algoritmo | ΔF1 | IC 95 % | Sujetos |
|---|---|---|---|
| B · Random Forest | +0.179 | [+0.105, +0.241] | 6 |
| B' · SVM (kernel RBF) | +0.183 | [+0.109, +0.266] | 6 |

Si el intervalo no contiene 0, la diferencia es significativa al 95 %.

**Modelo seleccionado para la plataforma:** B' · SVM (kernel RBF) (mayor F1 medio por sujeto).

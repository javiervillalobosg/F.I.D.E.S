# Resultados piloto FIDES (24 pares, 240 muestras)

AUC [IC 95% bootstrap por pares]

| Modelo | orig | jpeg90 | jpeg70 | jpeg50 | scale50 |
|---|---|---|---|---|---|
| Completo, entrena orig (A) | 0.80 [0.73-0.88] | 0.79 [0.72-0.88] | 0.79 [0.71-0.88] | 0.75 [0.68-0.86] | 0.63 [0.56-0.71] |
| Completo, entrena todas (B) | 0.81 [0.72-0.90] | 0.81 [0.72-0.90] | 0.80 [0.71-0.89] | 0.76 [0.69-0.86] | 0.64 [0.57-0.73] |
| Solo razon alta frecuencia | 0.73 [0.64-0.83] | 0.73 [0.63-0.83] | 0.73 [0.64-0.83] | 0.73 [0.64-0.83] | 0.63 [0.56-0.73] |
| Control: solo bits/pixel | 0.58 [0.52-0.65] | 0.59 [0.54-0.67] | 0.59 [0.54-0.66] | 0.58 [0.53-0.65] | 0.58 [0.53-0.65] |

Calibracion, modelo A (ECE con 5 intervalos / Brier)

| Version | orig | jpeg90 | jpeg70 | jpeg50 | scale50 |
|---|---|---|---|---|---|
| Sin calibrar | 0.10 / 0.19 | 0.14 / 0.19 | 0.06 / 0.20 | 0.13 / 0.21 | 0.10 / 0.24 |
| Platt | 0.14 / 0.19 | 0.11 / 0.19 | 0.09 / 0.20 | 0.07 / 0.21 | 0.11 / 0.24 |
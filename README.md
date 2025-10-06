# Treinamento de Redes Neurais com Keras (Dados Tabulares)

Este repositório contém os códigos desenvolvidos para a atividade de **Treinamento de Redes Neurais com Keras**, utilizando datasets tabulares para problemas de **classificação multiclasse** e **regressão**.

---

## Integrantes:
### Caio Henrique - RM 554600
### Carlos Eduardo - RM 555223
### Antônio Lino - RM 554518

## **Exercício 1 – Classificação Multiclasse (Wine Dataset)**

**Dataset:** [Wine Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/wine)

### **Objetivo**
Treinar uma rede neural para classificar vinhos em 3 classes e comparar o desempenho com um modelo do scikit-learn (`LogisticRegression`).

### **Configuração do Modelo Keras**
- 2 camadas ocultas com 32 neurônios cada  
- Função de ativação: ReLU  
- Camada de saída: 3 neurônios, ativação Softmax  
- Função de perda: `categorical_crossentropy`  
- Otimizador: Adam  

### **Resultados**
| Modelo                | Acurácia (teste) |
|-----------------------|----------------|
| Keras                 | 97.22%         |
| LogisticRegression    | 97.22%         |

**Conclusão:** Ambos os modelos tiveram desempenho igual no dataset, com acurácia de 97.22%.

---

## **Exercício 2 – Regressão (California Housing)**

**Dataset:** [California Housing Dataset (Scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

### **Objetivo**
Treinar uma rede neural para prever o valor médio das casas e comparar com um modelo do scikit-learn (`LinearRegression`).

### **Configuração do Modelo Keras**
- 3 camadas ocultas: 64, 32, 16 neurônios  
- Função de ativação: ReLU  
- Camada de saída: 1 neurônio, ativação Linear  
- Função de perda: MSE  
- Otimizador: Adam  

### **Resultados**
| Modelo                | RMSE (teste)   |
|-----------------------|----------------|
| Keras                 | 0.5274         |
| LinearRegression      | 0.7456         |

**Conclusão:** O modelo Keras apresentou melhor desempenho, com RMSE menor que o LinearRegression.

---


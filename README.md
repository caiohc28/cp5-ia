# Projeto de Inteligência Artificial e Visão Computacional

Este repositório contém os códigos desenvolvidos para as atividades de **Treinamento de Redes Neurais com Keras** e **Visão Computacional**.

---

## 👥 Integrantes do Grupo

- **Caio Henrique** - RM 554600
- **Carlos Eduardo** - RM 555223
- **Antônio Lino** - RM 554518

---

# 📊 PARTE 1: Treinamento de Redes Neurais com Keras (Dados Tabulares)

Esta seção contém experimentos com redes neurais aplicadas a datasets tabulares para problemas de **classificação multiclasse** e **regressão**.

---

## **Exercício 1 – Classificação Multiclasse (Wine Dataset)**

**Dataset:** [Wine Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/wine)

### **Objetivo**
Treinar uma rede neural para classificar vinhos em 3 classes e comparar o desempenho com um modelo do scikit-learn (`LogisticRegression`).

### **Configuração do Modelo Keras**
- **Arquitetura:** 2 camadas ocultas com 32 neurônios cada
- **Função de ativação:** ReLU nas camadas ocultas
- **Camada de saída:** 3 neurônios com ativação Softmax
- **Função de perda:** `categorical_crossentropy`
- **Otimizador:** Adam
- **Métricas:** Acurácia

### **Resultados**

| Modelo                | Acurácia (teste) |
|-----------------------|------------------|
| Keras Neural Network  | 97.22%          |
| LogisticRegression    | 97.22%          |

**Conclusão:** Ambos os modelos tiveram desempenho equivalente no dataset, alcançando acurácia de 97.22%. O modelo de rede neural conseguiu aprender os padrões com a mesma eficiência do modelo linear.

---

## **Exercício 2 – Regressão (California Housing)**

**Dataset:** [California Housing Dataset (Scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

### **Objetivo**
Treinar uma rede neural para prever o valor médio das casas na Califórnia e comparar com um modelo do scikit-learn (`LinearRegression`).

### **Configuração do Modelo Keras**
- **Arquitetura:** 3 camadas ocultas com 64, 32 e 16 neurônios
- **Função de ativação:** ReLU nas camadas ocultas
- **Camada de saída:** 1 neurônio com ativação Linear
- **Função de perda:** MSE (Mean Squared Error)
- **Otimizador:** Adam
- **Métricas:** RMSE (Root Mean Squared Error)

### **Resultados**

| Modelo                | RMSE (teste)    |
|-----------------------|-----------------|
| Keras Neural Network  | 0.5274         |
| LinearRegression      | 0.7456         |

**Conclusão:** O modelo Keras apresentou desempenho superior, com RMSE significativamente menor (0.5274) comparado ao LinearRegression (0.7456). A rede neural conseguiu capturar relações não-lineares nos dados, resultando em previsões mais precisas.

---

## 🚀 Como Rodar os Experimentos (Parte 1)

1. **Acesse o notebook no Google Colab:**
   - `Cp5_IA.ipynb`

2. **Execute as células sequencialmente:**
   - Cada célula contém o código para um dos exercícios
   - Clique na célula e pressione `Shift + Enter` para executá-la
   - Os resultados serão exibidos automaticamente

3. **Requisitos:**
   ```python
   pip install tensorflow scikit-learn pandas numpy matplotlib
   ```

---

# 🎯 PARTE 2: Visão Computacional

Esta seção contém o projeto de detecção e classificação de objetos utilizando duas ferramentas diferentes de Visão Computacional.

---

## **Ferramentas Utilizadas**

### **1. YOLOv8 (Ultralytics)**
- **Tipo:** Detecção de objetos em tempo real
- **Função:** Detectar e localizar objetos na imagem
- **Dataset:** COCO (80 classes pré-treinadas)
- **Saída:** Bounding boxes com coordenadas (x, y, width, height)

### **2. Hugging Face Transformers**
- **Tipo:** Classificação de imagens
- **Função:** Identificar com precisão o conteúdo da imagem
- **Dataset:** ImageNet (1000+ classes)
- **Saída:** Labels com scores de confiança

---

## **Objetivo do Projeto**

Desenvolver um sistema de visão computacional que:
1. **Detecta** objetos em uma imagem usando YOLOv8
2. **Classifica** cada objeto detectado usando Hugging Face
3. **Compara** as técnicas e demonstra suas diferenças
4. **Integra** ambas as ferramentas para um sistema completo

---

## **Arquitetura do Sistema**

```
┌─────────────────────────────────────────────┐
│          Imagem de Entrada                  │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌──────────────┐    ┌──────────────────┐
│   YOLOv8     │    │  Hugging Face    │
│  (Detecção)  │    │ (Classificação)  │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       │   Bounding Boxes    │  Labels + Scores
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   INTEGRAÇÃO    │
        │ Detecta + Recorta│
        │  + Classifica    │
        └─────────────────┘
```

---

## **Funcionalidades Implementadas**

### ✅ **Detecção com YOLOv8**
- Detecta até 80 classes diferentes (pessoas, carros, animais, objetos, etc.)
- Retorna coordenadas precisas de cada objeto
- Desenha bounding boxes coloridas
- Exibe confiança de cada detecção

### ✅ **Classificação com Hugging Face**
- Classifica imagem completa em 1000+ categorias
- Top 5 predições com scores
- Gráficos de barras interativos
- Suporte a modelos especializados (alimentos, animais, objetos)

### ✅ **Sistema Integrado**
- Detecta objetos com YOLO
- Recorta cada região detectada
- Classifica cada recorte com HF
- **Resultado:** "Objeto X na posição Y com classificação Z"

---

## **Análise Comparativa das Ferramentas**

| Critério            | YOLOv8                      | Hugging Face              |
|---------------------|----------------------------|---------------------------|
| **Função Principal**| Detecção + Localização     | Classificação            |
| **O que retorna**   | Coordenadas (x,y,w,h)      | Label + confiança        |
| **Número de Classes**| 80 (COCO)                 | 1000+ (ImageNet)         |
| **Velocidade**      | 30-60 FPS (tempo real)     | ~5-10 imagens/segundo    |
| **Uso ideal**       | "Onde está?"               | "O que é?"               |
| **Múltiplos objetos**| ✅ Sim                    | ❌ Classifica imagem toda|
| **Precisão**        | Boa para localização       | Excelente para ID        |

---

## **Resultados Obtidos**

### **Exemplo de Processamento:**

**Imagem Input:** Foto de uma bola de basquete

**Saída YOLOv8:**
```
✅ Detectou 1 objeto
   • sports ball: 67.85% de confiança
   • Coordenadas: (312, 195, 145, 140)
```

**Saída Hugging Face:**
```
🏆 Top 5 Classificações:
   🥇 basketball           ████████████████████████████████████████ 99.7%
   🥈 volleyball           ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0%
   🥉 soccer ball          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0%
      tennis ball         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0%
      baseball            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0%
```

**Sistema Integrado:**
```
🎯 Resultado Final:
   Objeto detectado: sports ball (YOLO)
   Classificação precisa: basketball (Hugging Face)
   Localização: posição (312, 195)
   Confiança combinada: 67.85% × 99.7% = 67.65%
   
💡 Análise: O YOLOv8 detectou genericamente como "bola esportiva",
   enquanto o Hugging Face identificou especificamente como "basquete"
   com altíssima confiança (99.7%), demonstrando a complementaridade
   das duas ferramentas.
```

---

## **Arquivos Gerados**

O sistema gera automaticamente 3 imagens de resultado:

1. **`resultado_yolo.jpg`** - Imagem com detecções do YOLOv8
2. **`comparacao_final.jpg`** - Comparação lado a lado (YOLO vs HF)
3. **`classificacao_detalhada.jpg`** - Gráfico de barras com Top 5

---

## 🚀 Como Rodar o Projeto (Parte 2)

### **1. Instalação de Dependências**
```python
!pip install ultralytics transformers torch pillow matplotlib
```

### **2. Upload da Imagem**
- Faça upload de uma imagem de teste no Google Colab
- Nomeie como `test_image.jpg`
- Ou modifique a variável `IMAGE_PATH` no código

### **3. Execução**
```python
# Opção 1: Execute o arquivo Python
python projeto_visao_computacional.py

# Opção 2: Execute no Jupyter/Colab
# Copie o código e execute célula por célula
```

### **4. Modelos Alternativos**

Você pode testar diferentes modelos do Hugging Face:

```python
# Classificação geral (padrão)
classifier = pipeline("image-classification", 
                     model="google/vit-base-patch16-224")

# Especializado em alimentos
classifier = pipeline("image-classification", 
                     model="nateraw/food")

# Especializado em objetos do cotidiano
classifier = pipeline("image-classification", 
                     model="microsoft/resnet-50")
```

---

### **Vídeo explicativo**

```
Link: https://www.youtube.com/watch?v=urXTno4Dsq8
```

## **Como executar**

Pré-requisitos

Conta Google (para usar Google Colab)
Navegador web atualizado
Imagem de teste (formato JPG, PNG ou JPEG)


Passo 1: Acessar o Google Colab

Abra o navegador e acesse: https://colab.research.google.com/
Faça login com sua conta Google
Clique em "Novo notebook" ou "File → New notebook"


Passo 2: Instalar Dependências
Na primeira célula do notebook, cole e execute:
python# Instalação de bibliotecas necessárias
!pip install ultralytics transformers torch pillow matplotlib

# Verificar instalação
print("✅ Bibliotecas instaladas com sucesso!")
⏱️ Tempo estimado: 2-3 minutos

Passo 3: Upload da Imagem de Teste

No menu lateral esquerdo, clique no ícone de pasta 📁
Clique no ícone de upload (seta para cima) ⬆️
Selecione sua imagem de teste
Aguarde o upload completar

---

## **Comparação: Quando Usar Cada Ferramenta**

### **Use YOLOv8 quando precisar:**
- ✅ Localizar objetos espacialmente
- ✅ Detectar múltiplos objetos simultaneamente
- ✅ Trabalhar em tempo real (vídeos, câmeras)
- ✅ Obter coordenadas precisas

### **Use Hugging Face quando precisar:**
- ✅ Classificação detalhada e específica
- ✅ Identificar objetos em categorias granulares
- ✅ Trabalhar com domínios especializados
- ✅ Alta precisão em categor

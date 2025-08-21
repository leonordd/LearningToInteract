# LearningToInteract
Final Repository of Dissertation

0fullbody_record.py --> BASE Version learning_to_interact/2input/fullbody/fullbody_video_recording82_1.py

BASE Version learning_to_interact/comunication/v7/keypoints_optimized6/keypoints_optimized6.pde

learning_to_interact/toy/0input_pred.py é o anterior mediapipe_holistic.py

1combine_files.py --> BASE Version learning_to_interact/00_pipeline/3combinar_ficheiros_csv2.py

2model_training.py --> BASE Version learning_to_interact/3integration/classification/version23/version22_1to4.py

Para instalar as dependências de python:
```bash
pip3 install -r requirements.txt
```

Para instalar as dependências de processing manualmente

```plaintext
COMO INSTALAR:
1. Abrir Processing IDE
2. Ir a Sketch > Import Library > Add Library...
3. Procurar e instalar as seguintes bibliotecas:

BIBLIOTECAS NECESSÁRIAS:
------------------------
1. VIDEO EXPORT
   Nome: Video Export
   Autor: Abe Pazos (hamoid)
   Descrição: Para exportar vídeos (com.hamoid.*)
   
   OU pesquisar por: "hamoid" na biblioteca

BIBLIOTECAS JÁ INCLUÍDAS (NÃO INSTALAR):
----------------------------------------
As seguintes são built-in do Processing/Java:
- javax.swing.* (interface gráfica)
- java.io.* (input/output de ficheiros)
- java.net.* (networking)
- java.util.concurrent.* (threading)
- java.text.* (formatação de texto/datas)

VERIFICAÇÃO:
-----------
Após instalação, verifica se pode usar:
import com.hamoid.*;

NOTA:
-----
Se não encontrar "Video Export", pode tentar:
1. Descarregar diretamente do GitHub: https://github.com/hamoid/video_export_processing
2. Colocar na pasta libraries do seu sketchbook do Processing
```







# Treino de Modelo de Classificação com PyTorch

**Problema de Classificação**
Este projeto consiste no treino de um modelo de multiclassificação que utiliza PyTorch, a partir de um dataset de inputs extraído de um ficheiro `.csv`. 

> ⚠️ **Nota:** Esta versão foi extraída do Google Colab (version2.ipyn) e **não inclui** a integração entre as partes do Módulo III. Inclui apenas o treino do modelo.

---

## 📌 Referências Utilizadas

1. [Build your first ML model in Python (YouTube)](https://www.youtube.com/watch?v=29ZQ3TDGgRQ) | [Google Colab](https://colab.research.google.com/drive/1KDqZvbLXXc75TchFzWmuT43Qq4488HgN)
2. [Train Test Split with Python Machine Learning (Scikit-Learn)](https://www.youtube.com/watch?v=SjOfbbfI2qY)
3. [How to Train a Machine Learning Model with Python](https://www.youtube.com/watch?v=T1nSZWAksNA)
4. [Machine Learning Tutorial Python - 7](https://www.youtube.com/watch?v=fwY9Qv96DJY)
5. [Pytorch Multiclass Classification with ROC and AUC](https://www.youtube.com/watch?v=EoqXQTT74vY) | [GitHub](https://github.com/jeffheaton/app_deep_learning)

---

## 📂 Estrutura do código de treino

```plaintext
📁 data/
   └── dataset19/
   |   └── combinado19.csv      # Dataset combinado
📁 output/                      # ficheiros do modelo treinado

Explicação de ficheiros:
📄 version20.py                  # Script de treino do modelo, correr este ficheiro 1º. Treina o modelo apenas uma vez com uma arquitetura especifica definida. Com normalização de dados
📄 integrated_system12.py        # script de predições em tempo real
📄 README.md                     # Este ficheiro
```

---

## ⚙️ Tecnologias Usadas
- Python 3.8+
- PyTorch
- pandas / NumPy
- scikit-learn
- Matplotlib / seaborn

---

## 🧠 Lógica do Código

### 1. **Carregar de Dados**
   - Lê os dados de `combinado.csv`
   - Remove colunas irrelevantes (`zp`, `zlh`, `zrh`, etc.)
   - Separar o `X` e o `y`. Define `X` como as features e `y` como a variável `Valor`

   - feature_columns: colunas no ficheiro combinado.csv sem o `MillisSinceEpoch`,`LocalMillisProcessing`,`Valor`,`zp`,`zlh`, `zrh`, `fm`, ou seja, sem as coordenadas z e a face mesh, apenas a pose e o hand recognition
   - output: `Valor` (valores de 0 a 20), como representado na imagem seguida
        - 0: Fundo Branco (Processing) | Mesh ausente (python)
        - 1: Forma redonda (Processing) | Mesh a fazer um movimento qualquer (Python)
        - 2 a 6: Forma redonda que aumenta e diminui (Processing) | Movimento específico da mão (Python)
        ![Figura exemplificativa do output](data/example.png)

### 2. **Preparar os dados**
   - Converter os dados em tensores PyTorch
   - Separar os dados em treino (80%) e teste/validação (20%)

### 3. **Modelo PyTorch**
   - Implementação do modelo `FlexibleModel` com arquitetura da rede "personalizável" - `Neural Network`
   - Função de Optimizer: `SGD`
   - Função de loss: `CrossEntropyLoss`

### 4. **Treino**
   - Treino do modelo por 200 epochs (Going from raw logits -> prediction probabilities -> prediction labels)
   - Avaliação da performance no conjunto de teste

### 5. **Avaliação**
   - Métricas de accuracy
   - Matriz de confusão normalizada
   - Visualização com `plot_predictions` e `plot_decision_boundary`

### 6. **Guardar Modelo**
   - Guarda o modelo no ficheiro `trained_model_1to6.pth` com a informação: input size, output size, arquitetura, etc.)

---

## 🏁 Como Executar

### 1. Clonar o repositório:

```bash
git clone https://github.com/ #(ainda sem repo)
cd version20
```

### 2. Instalar as dependências:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn

#ou (dependendo da versão de python)
pip3 install torch pandas numpy scikit-learn matplotlib seaborn
```

### 3. Corre o script principal

```bash
python version20.py

#ou (dependendo da versão de python)
python3 version20.py
```

---

## 💾 Guardar e Carregar Modelo

```python
# Guardar
torch.save(model_info, 'data/trained_model_1to6.pth')

# Carregar
checkpoint = torch.load('data/trained_model_1to6.pth', map_location='cpu', weights_only=True)
```

---

## 🧪 Print de valores

No final do código, existem partes que verificam os dados:
- Número de classes
- Arquitetura final do modelo
- Dimensões dos logits
- etc.

---
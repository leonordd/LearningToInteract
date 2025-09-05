# LearningToInteract
Este repositório serve como meio de disseminação do projeto de dissertação intitulado "Aprender a Interagir".

## Resumo do projeto

O domínio de investigação em Interação Humano-Computador (IHC) é vasto, no entanto, o conhecimento científico sobre o uso de técnicas de Inteligência Artificial (IA) para criar novas formas de interação com os sistemas está ainda pouco desenvolvido. Estes sistemas não se adaptam ao ambiente e não alteram as suas qualidades ao longo do tempo. Esta investigação propõe estudar a aplicabilidade de técnicas de aprendizagem computacional na interação com os sistemas. Pretende-se desenvolver um artefacto visual dinâmico e interativo que aprenda a agir consoante os dados de entrada escolhidos e fornecidos pelo utilizador, fomentando a exploração lúdica. Para tal, procedeu-se à revisão do estado da arte relativo a interfaces tecnológicas e Arte, a par da IA. Tendo em mente a fase de investigação, procedeu-se à conceptualização e implementação do artefacto. O artefacto foi desenvolvido em dois segmentos: artefacto não adaptativo – com interatividade com rato e teclado ou gestos predefinidos – e o artefacto adaptativo – com inputs gestuais escolhidos pelo utilizador. Neste último segmento, aplicaram-se e treinaram-se modelos de IA para a personalização da interação. A componente visual compreendeu o design e desenvolvimento das animações e de outros elementos necessários à representação do artefacto. A implementação seguiu uma abordagem iterativa, que envolveu experiências e testes diferentes do sistema computacional. Posteriormente, realizaram-se testes de avaliação com utilizadores, o que resultou na discussão sobre os mesmos. Os resultados obtidos evidenciaram potencialidades e limitações do projeto, que servem de base para uma reflexão sobre os contributos e possíveis caminhos de investigação futura.

# Implementação do Brinquedo

## Instalação das dependências necessárias
### Python
Para instalar as dependências de python:
```bash
pip3 install -r requirements.txt
```
### Processing
Para instalar as dependências de processing manualmente

COMO INSTALAR:
1. Abrir Processing IDE
2. Ir a Sketch > Import Library > Add Library...
3. Procurar e instalar as seguintes bibliotecas:

```plaintext
1. VIDEO EXPORT
   Nome: Video Export
   Autor: Abe Pazos (hamoid)
   Descrição: Para exportar vídeos (com.hamoid.*)
   
   OU pesquisar por: "hamoid" na biblioteca
```

BIBLIOTECAS JÁ INCLUÍDAS (NÃO INSTALAR)
As seguintes são built-in do Processing/Java:
- javax.swing.* (interface gráfica)
- java.io.* (input/output de ficheiros)
- java.net.* (networking)
- java.util.concurrent.* (threading)
- java.text.* (formatação de texto/datas)

VERIFICAÇÃO – Após instalação, verifica se pode usar: import com.hamoid.*;

> ⚠️ **Nota:** Se não encontrar "Video Export", pode tentar: 1. Descarregar diretamente do GitHub: https://github.com/hamoid/video_export_processing; 2. Colocar na pasta libraries do seu sketchbook do Processing




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
   └── 📁 0base/
   |   └── 📄 dados_teclas1.csv    # Conjunto de dados de saída
   |   └── 📄 video1.mp4           # Vídeo do brinquedo
   └── 📁 dataset43/
       └── 📁 output/              # ficheiros do modelo treinado
       |     └── 🌄 confusion_matrix.png   
       |     └── 🌄 training_curves.png   
       |     └── training_model.pth   
       |     └── training_statistics.json   
       └── 📄 combinado.csv    # Conjunto de dados combinado
       └── 📄 v1.csv           # Conjunto de dados de entrada
       └── 📄 v1.mp4           # Vídeo de captura de gestos

📁 toy2/          # programa de Processing
📁 training/      # treino do modelo

📄 README.md                   # Este ficheiro
```

---

## ⚙️ Tecnologias Usadas
- Python 3.8+
- PyTorch
- pandas / NumPy
- scikit-learn
- Matplotlib / seaborn

---

## 🧠 Lógica do Código (versão _coordinates only_)

### 1) **Carregar os dados**
- Lê `data/<dataset_folder>/combinado.csv`.
- Exclui **todas** as features de visibilidade e colunas não-feature:  
  `MillisSinceEpoch`, `TempoVideo`, `AnimacaoAtual`, `ValorMapeado`, `Valor`, `FrameNumber`, `Face`, `Pose`, `RightHand`, `LeftHand`.
- Mantém **apenas as coordenadas** (pose + mãos em x/y/z se existirem como coordenadas, mas **sem** visibilidade).
- `X` = features de coordenadas; `y` = `AnimacaoAtual`.

> ⚠️ Alteração face à versão anterior: já **não** se usa `Valor (0–20)`.  
> Agora o alvo é `AnimacaoAtual` (tipicamente **4 classes: 0–3**).

### 2) **Preparação dos dados**
- Substitui valores default `500.0` por `0.0`.
- Conversão para tensores PyTorch.
- _Split_: treino/validação/teste estratificado.  
  - 1º split: `train_test_split(..., test_size=0.3, stratify=y)`  
  - 2º split (val/test) aplicado conforme o script.

### 3) **Modelo PyTorch**
- `WeightedFlexibleModel(input_size, output_size, hidden_layers=[256,128,64])`
- Ativação **ReLU** + **Dropout 0.2** nas camadas escondidas.
- **Loss**: `CrossEntropyLoss`  
- **Optimizer**: `Adam(lr=0.0016, weight_decay=1e-4)`  
- **Scheduler**: `ReduceLROnPlateau(patience=10, factor=0.5)`

### 4) **Treino**
- **Default de épocas**: **10** (alterado do antigo 500).  
- Acompanha `loss`, `accuracy`, `weighted_accuracy`, `grad_norm`, `lr`.  
- **Early stopping** por _weighted accuracy_ de validação (`patience=100` no código atual).
- Guarda o melhor estado do modelo (com base na _weighted accuracy_).

### 5) **Avaliação**
- Métricas detalhadas (accuracy, precision/recall/F1 macro e por classe, Cohen’s Kappa, MCC).
- **Matriz de confusão (normalizada)**.
- Curvas de treino (loss/accuracy/lr/grad_norm).

### 6) **Guardar resultados**
- Modelo: `data/<dataset_folder>/output/trained_model.pth`
- Estatísticas (JSON): `data/<dataset_folder>/output/training_statistics.json`
- Gráficos:  
  `training_curves.png` e `confusion_matrix.png` em `data/<dataset_folder>/output/`

---

## 🏁 Como Executar

### 1) Clonar (quando tiver repositório)
```bash
git clone https://github.com/ # (ainda sem repo)
cd learning_to_interact/3integration/classification/version23
```

### 2) Instalar dependências
Se ainda não as tiver
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn joblib
# ou
pip3 install torch pandas numpy scikit-learn matplotlib seaborn joblib
```

### 3) Definir dataset
No topo do script ajustar:
```python
dataset_folder = "dataset48"   # muda conforme a pasta activa em data/
csv_file = "combinado.csv"
```

### 4) Correr o treino
```bash
python 2model_training.py
# ou
python3 2model_training.py
```

---

## 💾 Guardar / Carregar Modelo

```python
# Guardar (já feito no fim do script com info completa do treino)
torch.save(model_info, 'data/<dataset_folder>/output/trained_model.pth')

# Carregar (CPU-safe)
checkpoint = torch.load(
    'data/<dataset_folder>/output/trained_model.pth',
    map_location='cpu',
    weights_only=True
)
```

---

## 🧪 Impressões úteis (no fim do script)
- Nº de classes e distribuição.
- Arquitetura final e nº de parâmetros.
- Dimensões dos tensores/logits.
- Sumário das métricas (test/train), tempo por época e total.


## 🔁 Mapa de ficheiros (versão atual)

**Aquisição / Base** `0fullbody_rec.py`

**Pré-processamento** `1combine_files.py` 

**Modelo / Treino (versão coordinates-only)** `2model_training.py` 

- **SEM features de visibilidade** (Face, Pose, RightHand, LeftHand)  
- **Treino apenas com coordenadas**  
- **Épocas por defeito: 10**

**Legacy / Referência** `learning_to_interact/toy2/0input_pred.py` 



## 📌 Notas rápidas
- “Coordinates only” = exclui todas as colunas de **visibilidade** e as colunas administrativas não-feature.  
- Alvo é `AnimacaoAtual` (normalmente 4 classes: `0–3`).  
- Valores default `500.0` são convertidos para `0.0`.  
- Os artefactos (modelo/JSON/gráficos) ficam em: `data/<dataset_folder>/output/`.

# Inspeção de Qualidade com OpenCV e Redes Neurais Convolucionais

**Curso:** Machine Learning e Visão Computacional [T1]  
**Atividade:** Mini-Projeto Avaliativo - Módulo 2 - Semana 07  
**Estudante:**  Miguel Magro
**Vídeo de apresentação:** PREENCHER COM O LINK DO GOOGLE DRIVE  
**Repositório GitHub:** [programado10/projeto-inspecao-qualidade](https://github.com/programado10/projeto-inspecao-qualidade)

## Objetivo do projeto

Desenvolver um pipeline em Python para apoiar a inspeção visual de peças de fundição metálica, combinando processamento clássico de imagens e classificação com aprendizado profundo.

A análise com OpenCV permite observar bordas, texturas e irregularidades nas peças. Em seguida, uma Rede Neural Convolucional (CNN) aprende a classificar as imagens como **Defeituosa** ou **OK**. O projeto também apresenta os gráficos de treinamento, exemplos de previsões e uma matriz de confusão para analisar os acertos e erros do modelo.

**Arquivo principal:** [inspecao_qualidade_revisado.ipynb](inspecao_qualidade_revisado.ipynb). As explicações e os gráficos da execução registrada estão incorporados ao notebook.

## 1. Conhecendo o dataset

Foi utilizado o conjunto **Casting Product Image Data for Quality Inspection**, disponibilizado nas orientações do curso: [acessar o dataset no Google Drive](https://drive.google.com/file/d/1NZOjCHDRrpn7PmbFKVqegUP5arfdXHKK/view?usp=sharing).

O arquivo utilizado neste projeto é `casting_512x512.zip`, com imagens JPEG de 512 × 512 pixels. A quantidade abaixo corresponde a esse arquivo, e não a outras versões do dataset.

| Classe | Significado | Quantidade |
| --- | --- | ---: |
| `def_front` | Peças defeituosas | 781 |
| `ok_front` | Peças sem defeitos visíveis | 519 |
| **Total** | | **1.300** |

### Amostras da análise exploratória

| Imagem | Classe | Motivo da escolha |
| --- | --- | --- |
| `cast_ok_0_9658.jpeg` | `ok_front` | Referência de uma peça OK |
| `cast_def_0_122.jpeg` | `def_front` | Peça com defeito evidente |
| `cast_def_0_9643.jpeg` | `def_front` | Peça com defeito mais discreto |

Essas três imagens são utilizadas para comparar as técnicas clássicas. O carregamento para a CNN considera todas as 1.300 imagens, divididas entre treino e validação.

## 2. Configuração do ambiente e execução

### Organização das pastas

Salve este README na mesma pasta do notebook revisado. A organização de referência é:

```text
projeto_inspecao_qualidade/
├── README.md
├── inspecao_qualidade_revisado.ipynb
├── dataset/
│   └── casting_512x512/
│       ├── def_front/
│       └── ok_front/
├── notebook/
│   └── inspecao_qualidade.ipynb      # Versão original, se mantida
├── docs/                           # Materiais de apoio
└── resultados/                     # Criada durante a execução
```

### Bibliotecas utilizadas

| Biblioteca | Finalidade | Versão na execução registrada |
| --- | --- | --- |
| Python | Linguagem do projeto | 3.12.10 |
| TensorFlow | Carregamento de dados e treinamento | 2.21.0 |
| Keras, acessado por `tensorflow.keras` | Camadas, modelo e callbacks | 3.15.1 |
| OpenCV | Processamento clássico das imagens | 5.0.0 |
| NumPy | Matrizes e cálculos das previsões | 2.4.5 |
| Matplotlib | Visualização das imagens e métricas | 3.11.0 |

`pathlib`, `json` e `zipfile` fazem parte da biblioteca padrão do Python. O JupyterLab pode ser utilizado para abrir e executar o notebook.

### Instalação no Windows

Com Python 3.12 instalado, abra o PowerShell na pasta `projeto_inspecao_qualidade` e execute:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "tensorflow>=2.16,<2.22" "opencv-python>=4.9,<6" "matplotlib>=3.8,<4" "numpy>=1.26,<3" jupyterlab ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name inspecao-qualidade --display-name "Python (Inspeção de Qualidade)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Os comandos utilizam diretamente o Python do ambiente virtual, dispensando a ativação por script. As faixas de versões são as indicadas no notebook; a tabela acima registra o ambiente em que os resultados deste README foram obtidos.

### Preparação das imagens

1. Baixe o dataset pelo link informado na seção 1.
2. Extraia a pasta `casting_512x512` para dentro de `dataset`, preservando as subpastas `def_front` e `ok_front` e os nomes das imagens.
3. Alternativamente, mantenha `casting_512x512.zip` ao lado do notebook, dentro de `dataset` ou na pasta `Downloads`. Se não localizar as pastas extraídas, o notebook procura o ZIP nesses locais e extrai as imagens das duas classes.

Se os arquivos estiverem em outro local, ajuste a célula **2.2 Localização do dataset e pasta de resultados**. Escolha uma das opções:

```python
# Dataset já extraído:
PASTA_DATASET = Path(r"C:/meu_projeto/dataset/casting_512x512")
ARQUIVO_ZIP = None
```

```python
# Dataset ainda compactado:
PASTA_DATASET = None
ARQUIVO_ZIP = Path(r"C:/meu_projeto/casting_512x512.zip")
```

A pasta informada em `PASTA_DATASET` deve conter diretamente `def_front` e `ok_front`.

### Execução do notebook

1. No JupyterLab, abra `inspecao_qualidade_revisado.ipynb`.
2. Selecione o kernel **Python (Inspeção de Qualidade)**.
3. Reinicie o kernel e execute todas as células, de cima para baixo.
4. Confira a localização do dataset, as contagens por classe e os painéis OpenCV.
5. Aguarde o treinamento e a geração dos gráficos, das previsões e da matriz de confusão.
6. Salve o notebook para conservar as saídas da execução.

As **40 células de código** foram executadas em sequência na validação da versão revisada, utilizando as imagens reais. O tempo de treinamento depende do computador. Mesmo com a semente fixa, mudanças de ambiente e de execução podem alterar os resultados.

Na organização apresentada, os arquivos são salvos em `resultados/`, na raiz do projeto. A célula de configuração mostra o caminho efetivamente utilizado. Uma nova execução atualiza os arquivos de saída com os mesmos nomes; copie resultados anteriores se desejar conservá-los.

## 3. Análise exploratória com OpenCV

### Imagem original e conversão para Grayscale

A imagem original serve como referência para as comparações. A conversão de BGR para RGB permite a exibição correta com Matplotlib. A conversão para escala de cinza reduz a representação aos valores de intensidade, facilitando a suavização, a limiarização e a detecção de bordas.

### Suavização com Gaussian Blur

Foram comparados kernels de 3 × 3, 5 × 5 e 7 × 7. O pipeline principal utiliza **Gaussian Blur 5 × 5**, que suaviza pequenas variações da textura e mantém visíveis as irregularidades principais das amostras.

## 4. Destaque de características

### Limiarização

São comparados o threshold manual com limite **120**, o método automático de **Otsu** e o threshold binário invertido. Essas técnicas destacam diferenças de intensidade, mas também realçam a geometria normal da peça. Nas amostras observadas, a limiarização não isolou claramente os defeitos.

### Detecção de bordas com Canny

Foram comparados os pares de limiares **30–100**, **50–150** e **100–200**. A configuração **30–100** foi mantida no pipeline para evidenciar detalhes e descontinuidades dos contornos.

### Operações morfológicas

Foi utilizado um kernel de **3 × 3** para aplicar dilatação, erosão após dilatação e Closing. A dilatação engrossa as bordas; a erosão subsequente reduz esse espessamento. Nas mesmas condições, a sequência dilatação seguida de erosão corresponde ao Closing.

O **Closing** foi mantido para conectar pequenas interrupções nas bordas. Os resultados são comparados nas três amostras, incluindo um painel com as etapas do processamento e a sobreposição dos contornos em vermelho sobre as imagens originais.

### Conclusão da análise exploratória

O processamento tornou perceptíveis irregularidades no contorno da peça com defeito evidente e na região direita da peça com defeito mais discreto. A sobreposição facilita essa observação, mas também inclui contornos normais: ela não representa uma máscara de defeitos confirmados.

Essa etapa auxilia a compreensão visual do problema. A CNN recebe as imagens originais redimensionadas, e não as imagens de bordas produzidas pelo OpenCV.

## 5. Carregamento do dataset e Data Augmentation

O carregamento utiliza `keras.utils.image_dataset_from_directory`, com `subset="both"`, para criar os dois conjuntos em uma única divisão. As imagens são redimensionadas para **128 × 128 pixels**, em RGB, e organizadas em lotes de **32**, com semente **123**.

| Conjunto | Defeituosas | OK | Total |
| --- | ---: | ---: | ---: |
| Treino: 80% | 620 | 420 | 1.040 |
| Validação: 20% | 161 | 99 | 260 |
| **Total** | **781** | **519** | **1.300** |

A divisão é aleatória e não possui estratificação automática. O notebook confere a presença das duas classes, a inclusão de todos os arquivos e a ausência de caminhos compartilhados entre os conjuntos. A lista de arquivos é salva em `divisao_dados.json`.

`label_mode="binary"` gera rótulos no formato `(lote, 1)`. A ordem das classes é fixada como **0 = Defeituosa** e **1 = OK**. O uso de `prefetch` com `AUTOTUNE` prepara os próximos lotes enquanto o modelo processa os atuais.

### Transformações aplicadas no treinamento

| Transformação | Configuração |
| --- | --- |
| Espelhamento | Horizontal e vertical |
| Rotação | Fator 0,2: até 72° em cada sentido |
| Zoom | Fator 0,1 |
| Deslocamento | Até 5% na altura e na largura |
| Brilho | Fator 0,10 na escala de pixels de 0 a 255 |

O Data Augmentation produz variações dinâmicas para reduzir a dependência de uma posição ou iluminação específica. As transformações são aplicadas antes da normalização e ficam desativadas na validação e nas previsões.

## 6. Construção da Rede Neural Convolucional

O modelo é do tipo `Sequential`, com a seguinte estrutura:

```text
Entrada RGB: 128 × 128 × 3
→ Data Augmentation
→ Rescaling: divisão dos pixels por 255
→ Conv2D: 32 filtros, kernel 3 × 3, ReLU
→ MaxPooling2D: 2 × 2
→ Conv2D: 64 filtros, kernel 3 × 3, ReLU
→ MaxPooling2D: 2 × 2
→ Conv2D: 128 filtros, kernel 3 × 3, ReLU
→ MaxPooling2D: 2 × 2
→ Flatten
→ Dense: 128 neurônios, ReLU
→ Dropout: 30%
→ Dense: 1 neurônio, Sigmoid
```

As camadas convolucionais aprendem padrões de bordas, texturas e formas. O MaxPooling reduz as dimensões espaciais. O Flatten transforma os mapas de características em um vetor para as camadas densas. O Dropout ajuda a reduzir o risco de overfitting.

A saída Sigmoid corresponde à estimativa de **P(OK)**: valores menores que **0,5** indicam **Defeituosa**; valores a partir de **0,5** indicam **OK**. Essa pontuação não é uma garantia de acerto nem uma probabilidade calibrada.

## 7. Treinamento da CNN

| Parâmetro | Configuração |
| --- | --- |
| Otimizador | Adam |
| Taxa de aprendizado | 0,001 |
| Função de perda | `binary_crossentropy` |
| Métrica acompanhada | `BinaryAccuracy`, identificada como `accuracy` |
| Limite de épocas | 20 |
| EarlyStopping | Monitora `val_loss`, com paciência de 3 épocas |
| Restauração | Pesos da época com menor `val_loss` |
| ModelCheckpoint | Salva o melhor modelo pelo mesmo critério |
| Histórico | Registrado em CSV e JSON |

O modelo salvo é recarregado antes da auditoria e das previsões. Assim, as métricas finais correspondem ao arquivo `modelo_inspecao_qualidade.keras`. O critério de seleção é a menor perda de validação; a maior acurácia pode ocorrer em outra época.

## 8. Auditoria do treinamento

O notebook apresenta dois gráficos: **Accuracy de treino e validação por época** e **Loss de treino e validação por época**. Eles também são salvos em `accuracy_treino_validacao.png` e `loss_treino_validacao.png`.

### Resultados da execução registrada

| Indicador | Resultado |
| --- | ---: |
| Épocas executadas | 14 |
| Época do modelo selecionado | 11 |
| Loss de validação do modelo salvo | 0,4483 |
| Acurácia de validação | 80,38% |
| Imagens corretas na validação | 209 de 260 |
| Recall da classe Defeituosa | 74,53% |
| Precisão da classe Defeituosa | 92,31% |

Esses valores correspondem à execução incorporada ao notebook revisado. Devem ser atualizados se o modelo for treinado novamente e os resultados mudarem.

### Análise de overfitting

As perdas de treino e validação diminuíram no conjunto, com oscilações. Não houve um afastamento crescente e persistente entre as curvas nas últimas épocas, portanto essa execução não apresenta evidência clara de overfitting acentuado.

Após a época 11, a Val Loss não superou o melhor resultado durante três épocas. O EarlyStopping interrompeu o treinamento na época 14 e restaurou os pesos da época 11.

A comparação das curvas considera que Data Augmentation e Dropout atuam somente no treinamento. Por isso, as métricas de treino e validação são calculadas sob condições diferentes. O EarlyStopping ajuda a selecionar o modelo, mas não comprova sua capacidade de generalização.

## 9. Teste de previsões e análise dos erros

O notebook compara as classes reais com as previstas em um lote da validação e exibe até nove exemplos, com a pontuação da classe escolhida. Em seguida, calcula a matriz de confusão usando **todo o conjunto de validação**. Rótulos e previsões são obtidos juntos em cada lote, mantendo sua correspondência.

### Matriz de confusão

As linhas representam a classe real e as colunas representam a classe prevista.

| Classe real / Classe prevista | Defeituosa | OK | Total |
| --- | ---: | ---: | ---: |
| Defeituosa | 120 | 41 | 161 |
| OK | 10 | 89 | 99 |
| **Total** | **130** | **130** | **260** |

O modelo identificou corretamente 120 peças defeituosas e 89 peças OK. Entretanto, **41 peças defeituosas foram classificadas como OK**, e 10 peças OK foram classificadas como defeituosas. O recall de 74,53% da classe Defeituosa mostra a proporção de defeitos detectados entre os 161 exemplos dessa classe.

Os erros mostram que o modelo ainda precisa melhorar antes de uma aplicação industrial. Como a validação também orientou a seleção dos pesos, estas métricas descrevem o desenvolvimento do projeto. A avaliação em novas condições exigiria um conjunto de teste independente, preferencialmente com peças ou lotes de produção diferentes.

## 10. Arquivos gerados

| Arquivo em `resultados/` | Conteúdo |
| --- | --- |
| `defeito_grayscale.png` | Amostra convertida para escala de cinza |
| `defeito_gaussian_blur.png` | Amostra suavizada |
| `comparacao_morfologia.png` | Comparação de Canny, dilatação e Closing |
| `pipeline_opencv.png` | Etapas clássicas nas três amostras |
| `contornos_sobrepostos.png` | Contornos destacados sobre as imagens originais |
| `data_augmentation.png` | Exemplos das transformações de treinamento |
| `accuracy_treino_validacao.png` | Curvas de acurácia |
| `loss_treino_validacao.png` | Curvas de perda |
| `previsoes_validacao.png` | Exemplos de classificações |
| `matriz_confusao_validacao.png` | Erros e acertos por classe |
| `modelo_inspecao_qualidade.keras` | Modelo completo selecionado pela menor Val Loss |
| `configuracao_modelo.json` | Classes, formato de entrada, limiar e configurações |
| `divisao_dados.json` | Arquivos pertencentes a treino e validação |
| `historico_treinamento.csv` e `.json` | Métricas de cada época |
| `metricas_validacao.json` | Métricas finais e matriz de confusão |

Os arquivos são produzidos ao executar o notebook; não estão incorporados a este README. Os gráficos também podem ser consultados diretamente nas saídas salvas do notebook.

## 11. Organização conforme as sprints do curso

| Sprint | Etapa do projeto | Referência neste README |
| --- | --- | --- |
| 1. Configuração e versionamento | Ambiente, dataset e organização com Git | Seções 1, 2 e 12 |
| 2. Análise exploratória clássica | Grayscale e Gaussian Blur | Seção 3 |
| 3. Destaque de características | Limiarização, Canny e morfologia | Seção 4 |
| 4. Ingestão e augmentation | Leitura em lote, divisão dos dados e transformações | Seção 5 |
| 5. Arquitetura CNN e treinamento | Conv2D, MaxPooling2D, Flatten, Dense e Adam | Seções 6 e 7 |
| 6. Auditoria e gravação | Gráficos, análise, documentação e apresentação | Seções 8 a 10 e 13 |

## 12. Versionamento e documentação

O curso exige uso de Git com branches e commits lógicos. O notebook revisado e este README devem integrar o repositório informado no início do documento. A existência de um histórico de commits e de uma publicação no GitHub não foi verificada nesta revisão; essa etapa deve ser conferida antes da entrega.

Para organizar o versionamento, mantenha o notebook e a documentação sob controle do Git, trabalhe em uma branch de desenvolvimento e registre as alterações reais com mensagens que expliquem sua finalidade. Exemplos para futuras mudanças são `fix: corrigir carregamento das imagens` e `docs: atualizar instrucoes de execucao`.

Revise as alterações antes de integrar a branch principal. Evite incluir o ambiente `.venv/`, arquivos temporários, o ZIP do dataset e as imagens de treinamento no repositório.

## 13. Vídeo de apresentação e entrega

**Link do vídeo:** PREENCHER COM O LINK DO GOOGLE DRIVE.

O vídeo deve ter **até 5 minutos** e apresentar:

1. O objetivo do sistema e as classes utilizadas.
2. Uma demonstração do funcionamento no notebook.
3. O que Grayscale, Blur, limiarização, Canny e morfologia revelaram nas amostras.
4. A estrutura da CNN e as transformações do Data Augmentation, incluindo brilho.
5. Os gráficos de Loss e Acurácia, a interpretação de overfitting e os principais erros de classificação.

### Conferência antes da submissão

- Preencher o nome do estudante e o link do vídeo.
- Conferir o histórico de branches e commits exigido pelo curso.
- Salvar o notebook com suas saídas e conferir se este README corresponde à execução apresentada.
- Enviar o README e o vídeo ao Google Drive, com acesso de **leitor para qualquer pessoa com o link**.
- Abrir os links sem estar autenticado para conferir o acesso.
- Submeter os links na tarefa **Módulo 2 - Mini-Projeto Avaliativo**, no AVA, conforme o prazo informado pela instituição.

As orientações do curso permitem consultar IA generativa, desde que o estudante compreenda o código final e consiga explicar e defender suas decisões no vídeo.

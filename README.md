# Diagnóstico Computacional do TEA via Eletroencefalografia (EEG)

**Contexto:** Trabalho de Conclusão de Curso (TCC) — Engenharia da Computação, PUC Minas.

---

## 1. Visão Geral

Este repositório contém a implementação metodológica e experimental para a triagem, explicabilidade neurofisiológica e auxílio ao diagnóstico automatizado do **Transtorno do Espectro Autista (TEA / ASD)** em comparação com indivíduos **Neurotípicos (TD)**, a partir de sinais de Eletroencefalografia (EEG) multi-paradigma.

A pesquisa investiga e compara de forma progressiva:
1. **Machine Learning Clássico com Engenharia de Biomarcadores Relativos** (Potência Espectral Relativa, Razões Logarítmicas de Maturação Cortical, Transformada Z de Fisher e Limpeza Automática de Artefatos via IQR Intra-Sujeito).
2. **Deep Learning Topológico 3D (`BrainNet-CNN`)** com aumento de dados temporal (*Chunking Augmentation* + *Overlap* de 50%), preservando a geometria espacial dos 19 eletrodos do Sistema Internacional 10-20.
3. **Geometria Riemanniana no Espaço Tangente ($\text{SPD}$) + Teoria dos Grafos Corticais + Super-Fusão Multi-Paradigma (*Stacking*)**, treinando algoritmos especialistas para cada modalidade de exame sob validação cruzada rigorosa por sujeito (`GroupKFold` de 5 *folds*, livre de *data leakage*).
4. **Estudo Científico de Ablação e Padrões de Interferência nos 7 Exames Completos (503.256 Épocas em 5 Bandas, incluindo Gama 30–45 Hz):** Remoção sistemática de bandas oscilatórias (*Leave-One-Band-Out*) e regiões corticais (*Leave-One-Region-Out*), identificando onde a banda Gama e canais periféricos fornecem sinal cognitivo real vs. onde introduzem interferência eletromiográfica (EMG), elevando a acurácia global pós-poda para **83,50%** (**82,86% balanceada**).

---

## 2. Base de Dados (SFARI_EEG Multi-Paradigm BIDS — `ds006780`)

Após uma fase piloto inicial com 56 indivíduos (arquivos `.set`), o projeto migrou para a base clínica pública **SFARI_EEG multi-paradigm dataset (BIDS)**, desenvolvida no *Albert Einstein College of Medicine* e disponibilizada no repositório OpenNeuro:
* **Referência / DOI:** `doi:10.18112/openneuro.ds006780.v1.0.0` (Licença CC0).
* **Coorte Clínica Avaliada:** **103 participantes elegíveis** (**63 TEA [ASD]** e **40 Neurotípicos [TD]**, faixa etária de 8 a 14,5 anos).
* **Todos os 7 Paradigmas Extraídos e Processados (503.256 épocas de 2s com 50% de sobreposição em 5 Bandas: Delta, Theta, Alpha, Beta e Gamma):**
  * **`rest` (*Resting State* — `.bdf`):** Estado de repouso espontâneo (103 sujeitos | 36.389 épocas).
  * **`fast` (*FAST Face Processing* — `.bdf`):** Atenção social a faces vs. objetos (102 sujeitos | 88.840 épocas).
  * **`assr` (*Auditory Steady-State Response* — `.bdf`):** Sincronização auditiva rítmica a 27/40 Hz (96 sujeitos | 46.751 épocas).
  * **`avsrt` (*Audiovisual Simple Reaction Time* — `.bdf`):** Tempo de reação simples audiovisual (93 sujeitos | 82.105 épocas).
  * **`beepflash` (*Cross-Sensory Attention* — `.bdf`):** Alternância atencional multissensorial (91 sujeitos | 108.089 épocas).
  * **`ic` (*Illusory Contours* — `.bdf`):** Integração visual de contornos ilusórios de Kanizsa (99 sujeitos | 82.674 épocas).
  * **`motor` (*Mobile EEG on Treadmill* — `.vhdr`/`.eeg`):** Processamento sensório-motor em caminhada na esteira (95 sujeitos | 58.408 épocas).

---

## 3. Síntese dos Resultados Alcançados (Validação 5-Fold por Sujeito)

Todos os resultados abaixo foram avaliados estritamente no nível do paciente (`Subject-level GroupKFold` de 5 *folds*), garantindo que épocas ou blocos (*chunks*) de um mesmo participante **jamais** se misturem entre treino e teste:

| Fase / Arquitetura Avaliada | Paradigmas Utilizados | Acertos Totais | Acurácia Bruta (%) | Acurácia Balanceada (%) | Especificidade TD (%) | Sensibilidade ASD (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fase 1.4:** HGB em Épocas Brutas (PSD Absoluto) | `REST` | 58 / 103 | 56,31% | 51,96% | 32,50% (13/40) | 71,43% (45/63) |
| **Fase 1.5:** SVM RBF + Biomarcadores Relativos | `REST` | 69 / 103 | 66,99% | 66,17% | 62,50% (25/40) | 69,84% (44/63) |
| **Fase 1.6:** Ensemble Híbrido (Fusão Precoce) | `REST + FAST + ASSR` | 70 / 103 | 67,96% | 63,31% | 42,50% (17/40) | 84,13% (53/63) |
| **Fase 2.5:** Deep Learning `BrainNet-CNN` 3D (Limiar 0.50) | `REST` (6 *chunks*/suj) | 74 / 103 | 71,84% | 69,23% | 57,50% (23/40) | 80,95% (51/63) |
| **Fase 3.2:** `BrainNet-CNN` 3D + Calibração de Youden | `REST` (6 *chunks*/suj) | 76 / 103 | 73,79% | 74,92% | 80,00% (32/40) | 69,84% (44/63) |
| **Fase 4 (Campeão Equilíbrio):** Super-Fusão 3 Exames | `REST + Δ_ASSR + Δ_FAST` | 82 / 103 | 79,61% | 80,60% | 85,00% (34/40) | 76,19% (48/63) |
| **Fase 4 (Campeão Triagem):** Super-Fusão 3 Exames | `REST + FAST + ASSR + Δ_ASSR` | 85 / 103 | 82,52% | 79,78% | 67,50% (27/40) | **92,06% (58/63)** |
| **Fase 5 (Fusão Cega 7 Exames):** Sem Poda de Ruído | Todos os 7 Exames | 81 / 103 | 78,64% | 79,35% | 82,50% (33/40) | 76,19% (48/63) |
| **Fase 5 (Recorde Geral Pós-Ablação):** Poda de Interferências | **`REST + FAST + AVSRT + Motor`** | **86 / 103** | **83,50%** | **82,86%** | **80,00% (32/40)** | **85,71% (54/63)** |

---

## 4. Estrutura do Repositório

```text
TCC_EEG/
│
├── config.py                                      # Configurações globais (7 paradigmas, 5 bandas c/ Gama e regiões corticais)
├── 01_extracao.py                                 # Extração inicial BIDS (.bdf) dos 3 primeiros paradigmas em 4 bandas
├── 02_treinamento_ml.py                           # Fase 1: Machine Learning Clássico (SVM RBF e Ensemble Precoce)
├── 03_treinamento_dl.py                           # Fase 2 e 3: Deep Learning Topológico 3D (BrainNet-CNN + Youden)
├── 04_graficos.py                                 # Geração das Matrizes de Confusão em alta resolução (.png)
├── 05_fusao_multi_paradigma.py                    # Fase 4: Especialistas por Paradigma e Super-Fusão (3 exames)
├── 06_extracao_7_paradigmas_5_bandas.py           # Fase 5: Extração paralela dos 7 exames (.bdf e .vhdr) nas 5 bandas
├── 07_estudo_diferencial_ablacao.py               # Fase 5: Ablação de bandas/lobos, análise de interferência e Fusão (83,50%)
├── 08_validacao_auditoria_blindada.py             # Fase 6: Validação auditada sem canais oculares (15ch Riemann) em 10 sorteios
├── Documentacao_Processo.md                       # Relatório científico completo de todas as fases, tabelas e descobertas
├── Plano_TCC.md                                   # Planejamento metodológico inicial do projeto
├── ablacao_bandas_7_exames.png                    # Mapa de calor: relevância vs. interferência das 5 bandas nos 7 exames
├── ablacao_regioes_7_exames.png                   # Mapa de calor: relevância vs. interferência dos 5 lobos nos 7 exames
├── comparativo_7_paradigmas_gama.png              # Comparativo dos 7 exames (4 vs. 5 bandas + taxa de ruído IQR)
├── matriz_confusao_fase5_super_fusao_83_5.png     # Matriz de Confusão do Recorde Geral (83,50% Bruta | 82,86% Balanceada)
├── .gitignore                                     # Exclusão de ambientes virtuais (venv) e matrizes pesadas (.npy/.npz)
└── README.md                                      # Guia do projeto e instruções de reprodução
```

---

## 5. Como Rodar os Testes em Outra Máquina (Passo a Passo)

Qualquer integrante do grupo ou avaliador pode reproduzir todos os experimentos seguindo os passos abaixo:

### Passo 1: Clonar o Repositório e Criar o Ambiente Virtual (`venv`)
Certifique-se de possuir o **Python 3.10 ou 3.11** instalado. No terminal (PowerShell ou CMD), execute:

```powershell
git clone <URL_DO_REPOSITORIO>
cd TCC_EEG

# Criar ambiente virtual
python -m venv venv

# Ativar o ambiente virtual (Windows)
.\venv\Scripts\activate
# (No Linux/macOS use: source venv/bin/activate)
```

### Passo 2: Instalar as Dependências
Com o ambiente virtual `(venv)` ativo, instale os pacotes científicos necessários:

```powershell
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn mne tensorflow matplotlib seaborn
```

### Passo 3: Preparar os Dados (Duas Opções)

Como os arquivos brutos de EEG (`.bdf` e `.vhdr`) e as matrizes de cache (`.npy` / `.npz`) são grandes, eles estão listados no `.gitignore`. Escolha **uma** das duas opções:

* **Opção A — Rápida (Recomendada para o Grupo):**
  Copie o arquivo compacto pré-extraído `cache_7_paradigmas_5_bandas.npz` (~37 MB, para rodar toda a Fase 5 com os 7 exames) e/ou `X_features.npy`, `y_labels.npy`, `metadata.csv` (para rodar as Fases 1 a 4) diretamente para a pasta raiz `TCC_EEG/`. Com isso, **não é necessário** baixar os 75 GB de arquivos brutos do OpenNeuro!

* **Opção B — Extração Completa do Zero (A partir da Base Bruta OpenNeuro):**
  1. Baixe o dataset **ds006780** no OpenNeuro (`https://openneuro.org/datasets/ds006780`).
  2. No arquivo `config.py`, altere a variável `DATASET_DIR` (linha 10) para a pasta onde o dataset foi salvo:
     ```python
     DATASET_DIR = r'D:\caminho\para\ds006780-download'
     ```
  3. Execute o extrator paralelo dos 7 paradigmas em 5 bandas (~12 minutos em 6 núcleos):
     ```powershell
     python 06_extracao_7_paradigmas_5_bandas.py
     ```
     *(Opcional: caso deseje regerar também a matriz bruta `X_features.npy` das Fases 1 a 3, execute `python 01_extracao.py`)*

### Passo 4: Executar os Modelos, Ablações e Gerar os Gráficos

1. **Rodar a Fase 5 Completa (7 Exames + Ablação de Bandas e Lobos + Poda de Interferência — 83,50%):**
   ```powershell
   python 07_estudo_diferencial_ablacao.py
   ```
2. **Rodar a Fase 4 (Especialistas nos 3 Primeiros Exames + Geometria Riemanniana):**
   ```powershell
   python 05_fusao_multi_paradigma.py
   ```
3. **Rodar a Fase 3.2 (Deep Learning Topológico 3D — `BrainNet-CNN` + Youden):**
   ```powershell
   python 03_treinamento_dl.py
   ```
4. **Rodar a Fase 1 (Machine Learning Clássico — SVM RBF e Ensemble Precoce):**
   ```powershell
   python 02_treinamento_ml.py
   ```
5. **Regerar todas as Matrizes de Confusão (`.png`):**
   ```powershell
   python 04_graficos.py
   ```
6. **Rodar a Fase 6 (Validação Auditada sem Canais Oculares `Fp1, Fp2, F7, F8` + Empilhamento Aninhado em 10 Sorteios):**
   ```powershell
   python 08_validacao_auditoria_blindada.py
   ```

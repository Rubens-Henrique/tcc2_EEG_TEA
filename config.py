# config.py
import os

# ==========================================
# CAMINHOS DE DIRETÓRIOS E ARQUIVOS (BIDS)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho onde o dataset ds006780 foi baixado no Disco D:
DATASET_DIR = r'D:\dataset_tcc\ds006780-download'
PARTICIPANTS_FILE = os.path.join(DATASET_DIR, 'participants.tsv')

# Dicionário de Paradigmas para Fusão Multi-Paradigma:
# Chave = identificador curto | Valor = (palavra-chave no arquivo .bdf, coluna no participants.tsv)
PARADIGMS = {
    'rest': ('rest', 'completed_Resting_state'),
    'fast': ('fast', 'completed_FAST'),
    'assr': ('assr', 'completed_ASSR')
}

# Mantidos para retrocompatibilidade
TARGET_TASK_KEYWORD = 'rest'
COMPLETED_COL = 'completed_Resting_state'

# Caminhos para salvar as matrizes extraídas
X_FEATURES_PATH = os.path.join(BASE_DIR, 'X_features.npy')
Y_LABELS_PATH = os.path.join(BASE_DIR, 'y_labels.npy')
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.csv')

# ==========================================
# PARÂMETROS DO EEG E EXTRAÇÃO
# ==========================================
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
}
EPOCH_DURATION = 2.0
EPOCH_OVERLAP = 1.0  # 50% de sobreposição (1 segundo). Dobra o número de épocas extraídas.
SFREQ_RESAMPLE = 256

# Seleção de canais para evitar a Maldição da Dimensionalidade
USAR_SISTEMA_10_20 = True
CANAIS_10_20 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]

# ==========================================
# PARÂMETROS DE MACHINE LEARNING
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==========================================
# FASE 5: TODOS OS 7 PARADIGMAS + 5 BANDAS (INCLUINDO GAMA) + REGIÕES CORTICAIS
# ==========================================
# Chave: (task_tag_no_arquivo, formato_arquivo, coluna_completed_no_tsv)
PARADIGMS_7 = {
    'rest':      ('Restingstate', 'bdf',  'completed_Resting_state'),
    'fast':      ('FAST',         'bdf',  'completed_FAST'),
    'assr':      ('ASSR',         'bdf',  'completed_ASSR'),
    'avsrt':     ('AVSRT',        'bdf',  'completed_AVSRT'),
    'beepflash': ('Beepflash',    'bdf',  'completed_Beep-Flash'),
    'ic':        ('IC',           'bdf',  'completed_Illusory_contours'),
    'motor':     ('motor',        'vhdr', 'completed_Motor'),
}

BANDS_5 = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

# Mapeamento anatômico dos 19 eletrodos do Sistema 10-20 (índices de 0 a 18 em CANAIS_10_20)
REGIOES_CORTICAIS = {
    'Frontal':   [0, 1, 2, 3, 4, 5, 6],   # Fp1, Fp2, F7, F3, Fz, F4, F8
    'Temporal':  [7, 11],                  # T7, T8
    'Central':   [8, 9, 10],               # C3, Cz, C4 (Córtex Sensório-Motor)
    'Parietal':  [12, 13, 14, 15, 16],     # P7, P3, Pz, P4, P8
    'Occipital': [17, 18],                 # O1, O2 (Córtex Visual Primário)
}

CACHE_7P_5B_PATH = os.path.join(BASE_DIR, 'cache_7_paradigmas_5_bandas.npz')

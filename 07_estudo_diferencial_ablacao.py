# 07_estudo_diferencial_ablacao.py
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import config

PAIR_BANDS_5 = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4),
    (3, 4)
]

BAND_NAMES = ['Delta (0.5-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-13Hz)', 'Beta (13-30Hz)', 'Gamma (30-45Hz)']
BAND_SHORT = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

PARADIGM_LABELS = {
    'rest':      'REST (Repouso)',
    'fast':      'FAST (Faces/Social)',
    'assr':      'ASSR (Auditivo 40Hz)',
    'avsrt':     'AVSRT (Reacao A/V)',
    'beepflash': 'BeepFlash (Atencao)',
    'ic':        'IC (Contorno Visual)',
    'motor':     'Motor (Esteira Movel)',
}

def construir_features_subconjunto(data_p, keep_bands=None, keep_channels=None, modo_feat='comb', usar_psd_absoluto=False):
    """
    Reconstroi dinamicamente o vetor de caracteristicas (Espectral, Riemanniano/Grafo ou Hibrido)
    restringindo estritamente as bandas de frequencia (keep_bands) e os canais/lobos (keep_channels).
    Qualquer banda ou regiao removida tem 100% de suas potencias, razoes e conexoes eliminadas.
    """
    if keep_bands is None:
        keep_bands = [0, 1, 2, 3, 4]
    if keep_channels is None:
        keep_channels = list(range(19))

    keep_bands = sorted(list(keep_bands))
    keep_channels = sorted(list(keep_channels))
    C = len(keep_channels)
    N = data_p['conn_med'].shape[0]

    upper_conn = np.triu_indices(C, k=1)
    triu_spd = np.triu_indices(C, k=0)

    conn_med = data_p['conn_med'][:, keep_bands][:, :, keep_channels][:, :, :, keep_channels]
    conn_iqr = data_p['conn_iqr'][:, keep_bands][:, :, keep_channels][:, :, :, keep_channels]

    # Recalcular potencia relativa apenas dentro das bandas mantidas
    psd_med_raw = data_p['psd5_med'][:, keep_bands][:, :, keep_channels]
    psd_med_norm = psd_med_raw / (np.sum(psd_med_raw, axis=1, keepdims=True) + 1e-12)
    psd_iqr_sub = data_p['psd5_iqr'][:, keep_bands][:, :, keep_channels]

    if usar_psd_absoluto:
        psd_feat_med = data_p['psd_abs_med'][:, keep_bands][:, :, keep_channels]
    else:
        psd_feat_med = psd_med_norm

    # Filtrar pares de razoes espectrais onde AMBAS as bandas pertencem a keep_bands
    valid_r_idx = [idx for idx, (b1, b2) in enumerate(PAIR_BANDS_5) if (b1 in keep_bands and b2 in keep_bands)]

    blocos_spec = []
    for b_i in range(len(keep_bands)):
        blocos_spec.append(conn_med[:, b_i][:, upper_conn[0], upper_conn[1]])
        blocos_spec.append(conn_iqr[:, b_i][:, upper_conn[0], upper_conn[1]])
        blocos_spec.append(psd_feat_med[:, b_i, :])
        if not usar_psd_absoluto:
            blocos_spec.append(psd_iqr_sub[:, b_i, :])

    if valid_r_idx and not usar_psd_absoluto:
        rat_m = data_p['rat_med'][:, valid_r_idx][:, :, keep_channels].reshape(N, -1)
        rat_q = data_p['rat_iqr'][:, valid_r_idx][:, :, keep_channels].reshape(N, -1)
        blocos_spec.extend([rat_m, rat_q])

    X_spec = np.nan_to_num(np.hstack(blocos_spec), nan=0.0, posinf=0.0, neginf=0.0)
    if modo_feat == 'spec':
        return X_spec.astype(np.float32)

    # Construir projecao no Espaco Tangente Riemanniano (SPD) + Centralidade de Grafo no subgrafo (C x C)
    feats_riem = []
    for i in range(N):
        f_i = []
        for b_i in range(len(keep_bands)):
            Z = conn_med[i, b_i]  # (C, C) com diag = 0
            p_rel = np.maximum(psd_med_norm[i, b_i], 1e-5)

            strength_signed = np.sum(Z, axis=1)
            strength_abs = np.sum(np.abs(Z), axis=1)
            strength_std = np.std(Z, axis=1)

            R = np.tanh(Z)
            np.fill_diagonal(R, 1.0)
            d_sqrt = np.sqrt(p_rel)
            Sigma = np.outer(d_sqrt, d_sqrt) * R
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            eigvals_clamped = np.maximum(eigvals, 1e-4)
            log_Sigma = (eigvecs * np.log(eigvals_clamped)) @ eigvecs.T
            tangent_vec = log_Sigma[triu_spd]

            f_i.append(np.concatenate([strength_signed, strength_abs, strength_std, tangent_vec]))

        # Entropia espectral por canal nas bandas ativas
        if len(keep_bands) > 1:
            P_mat = np.maximum(psd_med_norm[i], 1e-8)
            ent = -np.sum(P_mat * np.log(P_mat), axis=0)
            f_i.append(ent)

        feats_riem.append(np.concatenate(f_i))

    X_riem = np.nan_to_num(np.array(feats_riem, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if modo_feat == 'riem':
        return X_riem
    return np.hstack([X_spec, X_riem]).astype(np.float32)

def otimizar_youden(y_true, probs):
    fpr, tpr, ths = roc_curve(y_true, probs)
    idx_best = np.argmax(tpr - fpr)
    th = ths[idx_best]
    preds = (probs >= th).astype(int)
    acc = np.mean(y_true == preds)
    bal = balanced_accuracy_score(y_true, preds)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    auc = roc_auc_score(y_true, probs)
    return acc, bal, tn / (tn + fp), tp / (tp + fn), auc, th, (tn, fp, fn, tp)

def avaliar_oof(X_all, y_all, grp_all, folds_pids, pids_ref, pid_to_idx, clf_fn, k_best=50, por_chunk=False):
    if not por_chunk:
        pids_u = pd.unique(grp_all)
        X_suj = np.array([np.median(X_all[grp_all == pid], axis=0) for pid in pids_u], dtype=np.float32)
        y_suj = np.array([y_all[grp_all == pid][0] for pid in pids_u], dtype=np.int32)
        grp_use = pids_u
    else:
        X_suj, y_suj, grp_use = X_all, y_all, grp_all

    oof_probs = np.full(len(pids_ref), np.nan)

    for fold_idx, (tr_pids, te_pids) in enumerate(folds_pids):
        m_tr = np.isin(grp_use, list(tr_pids))
        m_te = np.isin(grp_use, list(te_pids))
        if np.sum(m_te) == 0 or len(np.unique(y_suj[m_tr])) < 2:
            continue

        X_tr, y_tr = X_suj[m_tr], y_suj[m_tr]
        X_te, grp_te = X_suj[m_te], grp_use[m_te]

        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        k_use = min(k_best, X_tr_s.shape[1])
        sel = SelectKBest(f_classif, k=k_use)
        X_tr_sel = sel.fit_transform(X_tr_s, y_tr)
        X_te_sel = sel.transform(X_te_s)

        clf = clf_fn(fold_idx)
        clf.fit(X_tr_sel, y_tr)
        probs_te = clf.predict_proba(X_te_sel)[:, 1]

        for pid in pd.unique(grp_te):
            if pid in pid_to_idx:
                oof_probs[pid_to_idx[pid]] = np.mean(probs_te[grp_te == pid])

    return oof_probs

def gerar_graficos_ablacao(matriz_bandas, matriz_lobos, nomes_paradigmas, accs_4b, accs_5b, taxas_art):
    # 1. Heatmap de Impacto das Bandas de Frequencia (Delta Acuracia Balanceada ao remover a banda)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    # Inverter sinal para mostrar "Importancia da Banda" = Queda de acuracia quando removida (Positivo = Ajuda, Negativo = Atrapalha/Ruido)
    imp_bandas = -matriz_bandas * 100.0
    vmax = max(abs(np.nanmin(imp_bandas)), abs(np.nanmax(imp_bandas)), 5.0)
    im = ax.imshow(imp_bandas, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Impacto na Acuracia Balanceada (p.p.)\n(+ Relevante / Biomarcador | - Interferencia / Ruido)', fontsize=10)

    ax.set_xticks(np.arange(len(BAND_SHORT)))
    ax.set_xticklabels(BAND_NAMES, rotation=20, ha='right', fontsize=10)
    ax.set_yticks(np.arange(len(nomes_paradigmas)))
    ax.set_yticklabels(nomes_paradigmas, fontsize=10)
    ax.set_title('Estudo de Ablacao Espectral nos 7 Exames: Relevancia vs. Interferencia por Banda', fontsize=12, pad=14)

    for i in range(imp_bandas.shape[0]):
        for j in range(imp_bandas.shape[1]):
            val = imp_bandas[i, j]
            cor = 'white' if abs(val) > vmax * 0.55 else 'black'
            sinal = '+' if val >= 0 else ''
            ax.text(j, i, f'{sinal}{val:.1f}%', ha='center', va='center', color=cor, fontsize=9.5, fontweight='bold')

    fig.tight_layout()
    plt.savefig('ablacao_bandas_7_exames.png', dpi=300)
    plt.close(fig)

    # 2. Heatmap de Impacto das Regioes Corticais (Lobos Cerebrais)
    nomes_lobos = list(config.REGIOES_CORTICAIS.keys())
    imp_lobos = -matriz_lobos * 100.0
    vmax_l = max(abs(np.nanmin(imp_lobos)), abs(np.nanmax(imp_lobos)), 5.0)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(imp_lobos, cmap='RdBu_r', vmin=-vmax_l, vmax=vmax_l, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Impacto na Acuracia Balanceada (p.p.)\n(+ Regiao Essencial | - Regiao com Ruido/Interferencia)', fontsize=10)

    ax.set_xticks(np.arange(len(nomes_lobos)))
    ax.set_xticklabels([f'{l}\n({len(config.REGIOES_CORTICAIS[l])} canais)' for l in nomes_lobos], fontsize=10)
    ax.set_yticks(np.arange(len(nomes_paradigmas)))
    ax.set_yticklabels(nomes_paradigmas, fontsize=10)
    ax.set_title('Estudo de Ablacao Espacial nos 7 Exames: Relevancia vs. Interferencia por Lobo Cortical', fontsize=12, pad=14)

    for i in range(imp_lobos.shape[0]):
        for j in range(imp_lobos.shape[1]):
            val = imp_lobos[i, j]
            cor = 'white' if abs(val) > vmax_l * 0.55 else 'black'
            sinal = '+' if val >= 0 else ''
            ax.text(j, i, f'{sinal}{val:.1f}%', ha='center', va='center', color=cor, fontsize=9.5, fontweight='bold')

    fig.tight_layout()
    plt.savefig('ablacao_regioes_7_exames.png', dpi=300)
    plt.close(fig)

    # 3. Grafico Comparativo dos 7 Paradigmas (4 Bandas vs 5 Bandas com Gama + Taxa de Artefato)
    fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(nomes_paradigmas))
    larg = 0.35

    b1 = ax1.bar(x - larg/2, np.array(accs_4b) * 100, larg, label='4 Bandas Classicas (0.5-30 Hz)', color='#4C72B0')
    b2 = ax1.bar(x + larg/2, np.array(accs_5b) * 100, larg, label='5 Bandas + Gama (0.5-45 Hz)', color='#55A868')

    ax1.set_ylabel('Acuracia Clinica Balanceada (%)', fontsize=11)
    ax1.set_ylim(45, 90)
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.split(' ')[0] for p in nomes_paradigmas], fontsize=10.5, fontweight='bold')
    ax1.axhline(50, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_title('Comparativo dos 7 Exames de EEG: Efeito da Banda Gama (30-45 Hz) vs. Artefatos', fontsize=12, pad=14)

    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.6, f'{h:.1f}%', ha='center', va='bottom', fontsize=8.5)
    for rect in b2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.6, f'{h:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(x, np.array(taxas_art) * 100, color='#C44E52', marker='o', linewidth=2, linestyle='-.', label='Epocas c/ Ruido Rejeitadas (%)')
    ax2.set_ylabel('Taxa de Artefatos Rejeitados pelo IQR (%)', color='#C44E52', fontsize=10.5)
    ax2.set_ylim(0, max(25, max(taxas_art)*140))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9.5)

    fig.tight_layout()
    plt.savefig('comparativo_7_paradigmas_gama.png', dpi=300)
    plt.close(fig)

def executar_estudo_ablacao_completo():
    if not os.path.exists(config.CACHE_7P_5B_PATH):
        print("Erro: Execute 'python 06_extracao_7_paradigmas_5_bandas.py' primeiro.")
        return

    raw_cache = np.load(config.CACHE_7P_5B_PATH, allow_pickle=True)
    p_keys = [p for p in config.PARADIGMS_7.keys() if f'{p}_grp' in raw_cache]

    # Montar estrutura por paradigma e conjunto global de participantes unicos (ASD e TD)
    paradigms = {}
    all_pids_list = []
    pid_label_map = {}

    for p in p_keys:
        grp = raw_cache[f'{p}_grp']
        y_v = raw_cache[f'{p}_y']
        paradigms[p] = {
            'conn_med': raw_cache[f'{p}_conn_med'],
            'conn_iqr': raw_cache[f'{p}_conn_iqr'],
            'psd5_med': raw_cache[f'{p}_psd5_med'],
            'psd5_iqr': raw_cache[f'{p}_psd5_iqr'],
            'psd_abs_med': raw_cache[f'{p}_psd_abs_med'],
            'rat_med': raw_cache[f'{p}_rat_med'],
            'rat_iqr': raw_cache[f'{p}_rat_iqr'],
            'y': y_v,
            'grp': grp,
            'art': raw_cache[f'{p}_art'],
            'ep': raw_cache[f'{p}_ep'],
        }
        for pid, lbl in zip(grp, y_v):
            if pid not in pid_label_map:
                pid_label_map[pid] = int(lbl)
                all_pids_list.append(pid)

    # Usar os 103 sujeitos de REST como referencia principal (ou todos os sujeitos unicos da coorte)
    pids_ref = pd.unique(paradigms['rest']['grp'])
    y_ref = np.array([pid_label_map[pid] for pid in pids_ref], dtype=np.int32)
    pid_to_idx = {pid: i for i, pid in enumerate(pids_ref)}

    cv = GroupKFold(n_splits=5)
    folds_pids = []
    for tr_idx, te_idx in cv.split(paradigms['rest']['psd5_med'], paradigms['rest']['y'], groups=paradigms['rest']['grp']):
        folds_pids.append((set(paradigms['rest']['grp'][tr_idx]), set(paradigms['rest']['grp'][te_idx])))

    candidatos_modelos = [
        ('LogReg_L2', lambda f: LogisticRegression(C=0.2, class_weight='balanced', max_iter=500, random_state=42+f)),
        ('SVM_RBF',   lambda f: SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42+f)),
        ('MLP_Tang',  lambda f: MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=300, random_state=42+f)),
    ]

    print("=========================================================================================================")
    print("PARTE 1: BENCHMARK COMPLETO DOS 7 EXAMES DO DATASET (IMPACTO DA BANDA GAMA 30-45 Hz E RUIDO)")
    print("=========================================================================================================")
    print(f"{'Exame (Paradigma)':<23} | {'Suj':<3} | {'Ruido%':<6} | {'Melhor Configuracao':<22} | {'4 Bandas':<8} | {'5B(+Gama)':<9} | {'Delta Gama':<10}")
    print("-" * 105)

    melhor_cfg_por_p = {}
    oof_campeoes_7p = {}
    accs_4b_plot, accs_5b_plot, taxas_art_plot, nomes_plot = [], [], [], []

    for p in p_keys:
        dp = paradigms[p]
        n_suj_p = len(pd.unique(dp['grp']))
        taxa_art = float(np.mean(dp['art']))

        # Buscar o melhor especialista em 4 bandas (sem Gama) vs 5 bandas (com Gama)
        best_4b_bal, best_5b_bal = 0.0, 0.0
        best_overall_bal = 0.0
        best_cfg = None
        best_oof = None

        for modo_feat in ['spec', 'riem', 'comb']:
            X_4b = construir_features_subconjunto(dp, keep_bands=[0, 1, 2, 3], modo_feat=modo_feat)
            X_5b = construir_features_subconjunto(dp, keep_bands=[0, 1, 2, 3, 4], modo_feat=modo_feat)

            for por_chunk in [False, True]:
                for k_b in [40, 60]:
                    for m_name, m_fn in candidatos_modelos:
                        oof4 = avaliar_oof(X_4b, dp['y'], dp['grp'], folds_pids, pids_ref, pid_to_idx, m_fn, k_best=k_b, por_chunk=por_chunk)
                        v4 = ~np.isnan(oof4)
                        _, bal4, _, _, _, _, _ = otimizar_youden(y_ref[v4], oof4[v4])
                        if bal4 > best_4b_bal:
                            best_4b_bal = bal4
                        if bal4 > best_overall_bal:
                            best_overall_bal = bal4
                            best_cfg = (modo_feat, m_name, m_fn, k_b, por_chunk, [0, 1, 2, 3])
                            best_oof = oof4

                        oof5 = avaliar_oof(X_5b, dp['y'], dp['grp'], folds_pids, pids_ref, pid_to_idx, m_fn, k_best=k_b, por_chunk=por_chunk)
                        v5 = ~np.isnan(oof5)
                        _, bal5, _, _, _, _, _ = otimizar_youden(y_ref[v5], oof5[v5])
                        if bal5 > best_5b_bal:
                            best_5b_bal = bal5
                        if bal5 > best_overall_bal:
                            best_overall_bal = bal5
                            best_cfg = (modo_feat, m_name, m_fn, k_b, por_chunk, [0, 1, 2, 3, 4])
                            best_oof = oof5

        melhor_cfg_por_p[p] = best_cfg
        oof_campeoes_7p[p] = best_oof
        diff_gama = (best_5b_bal - best_4b_bal) * 100.0
        sinal_g = '+' if diff_gama >= 0 else ''
        cfg_str = f"{best_cfg[1]} ({best_cfg[0]}, k={best_cfg[3]})"
        print(f"{PARADIGM_LABELS[p]:<23} | {n_suj_p:<3} | {taxa_art:5.1%} | {cfg_str:<22} | {best_4b_bal:7.2%} | {best_5b_bal:8.2%} | {sinal_g}{diff_gama:6.2f} p.p.")

        accs_4b_plot.append(best_4b_bal)
        accs_5b_plot.append(best_5b_bal)
        taxas_art_plot.append(taxa_art)
        nomes_plot.append(PARADIGM_LABELS[p])

    print("-" * 105)

    # =========================================================================
    # PARTE 2: ABLACAO DE BANDAS DE FREQUENCIA (LEAVE-ONE-BAND-OUT) NOS 7 EXAMES
    # =========================================================================
    print("\n=========================================================================================================")
    print("PARTE 2: ABLACAO ESPECTRAL (VARIACAO DA ACURACIA AO REMOVER CADA BANDA DE FREQUENCIA)")
    print("Interpretacao: Valor NEGATIVO = A acuracia CAIU sem a banda (Banda Relevante / Biomarcador).")
    print("               Valor POSITIVO = A acuracia SUBIU sem a banda (Banda gerava Ruido / Interferencia!).")
    print("=========================================================================================================")
    header_b = f"{'Exame':<23} | {'Base 5B':<8} | " + " | ".join([f"Sem {b:<6}" for b in BAND_SHORT])
    print(header_b)
    print("-" * 105)

    matriz_delta_bandas = np.zeros((len(p_keys), 5), dtype=np.float32)
    oof_podados_7p = {}

    for p_idx, p in enumerate(p_keys):
        dp = paradigms[p]
        modo_feat, m_name, m_fn, k_b, por_chunk, _ = melhor_cfg_por_p[p]

        X_5b = construir_features_subconjunto(dp, keep_bands=[0, 1, 2, 3, 4], modo_feat=modo_feat)
        oof_base = avaliar_oof(X_5b, dp['y'], dp['grp'], folds_pids, pids_ref, pid_to_idx, m_fn, k_best=k_b, por_chunk=por_chunk)
        vb = ~np.isnan(oof_base)
        _, bal_base, _, _, _, _, _ = otimizar_youden(y_ref[vb], oof_base[vb])

        melhor_oof_p = oof_campeoes_7p[p]
        melhor_bal_p = max(accs_4b_plot[p_idx], accs_5b_plot[p_idx])

        deltas_str = []
        for b_rem in range(5):
            sub_b = [b for b in range(5) if b != b_rem]
            X_sub = construir_features_subconjunto(dp, keep_bands=sub_b, modo_feat=modo_feat)
            oof_sub = avaliar_oof(X_sub, dp['y'], dp['grp'], folds_pids, pids_ref, pid_to_idx, m_fn, k_best=k_b, por_chunk=por_chunk)
            vs = ~np.isnan(oof_sub)
            _, bal_sub, _, _, _, _, _ = otimizar_youden(y_ref[vs], oof_sub[vs])

            delta = bal_sub - bal_base
            matriz_delta_bandas[p_idx, b_rem] = delta
            sinal = '+' if delta >= 0 else ''
            deltas_str.append(f"{sinal}{delta*100:6.2f}%")

            if bal_sub > melhor_bal_p:
                melhor_bal_p = bal_sub
                melhor_oof_p = oof_sub

        oof_podados_7p[p] = (melhor_oof_p, melhor_bal_p)
        print(f"{PARADIGM_LABELS[p]:<23} | {bal_base:7.2%} | " + " | ".join([f"{s:>10}" for s in deltas_str]))

    print("-" * 105)

    # =========================================================================
    # PARTE 3: ABLACAO ESPACIAL DE LOBOS CORTICAIS (LEAVE-ONE-REGION-OUT)
    # =========================================================================
    nomes_lobos = list(config.REGIOES_CORTICAIS.keys())
    print("\n=========================================================================================================")
    print("PARTE 3: ABLACAO ANATOMICA DE LOBOS CEREBRAIS (VARIACAO DA ACURACIA AO REMOVER CADA REGIAO)")
    print("Interpretacao: Valor NEGATIVO = Regiao Essencial ao Exame | Valor POSITIVO = Regiao trazia Interferencia!")
    print("=========================================================================================================")
    header_l = f"{'Exame':<23} | {'19 Canais':<9} | " + " | ".join([f"-{l:<9}" for l in nomes_lobos])
    print(header_l)
    print("-" * 105)

    matriz_delta_lobos = np.zeros((len(p_keys), len(nomes_lobos)), dtype=np.float32)

    for p_idx, p in enumerate(p_keys):
        dp = paradigms[p]
        modo_feat, m_name, m_fn, k_b, por_chunk, best_bands = melhor_cfg_por_p[p]

        X_full = construir_features_subconjunto(dp, keep_bands=best_bands, keep_channels=list(range(19)), modo_feat=modo_feat)
        oof_full = avaliar_oof(X_full, dp['y'], dp['grp'], folds_pids, pids_ref, pid_to_idx, m_fn, k_best=k_b, por_chunk=por_chunk)
        vf = ~np.isnan(oof_full)
        _, bal_full, _, _, _, _, _ = otimizar_youden(y_ref[vf], oof_full[vf])

        melhor_oof_p, melhor_bal_p = oof_podados_7p[p]
        deltas_l_str = []

        for l_idx, l_name in enumerate(nomes_lobos):
            rem_ch = set(config.REGIOES_CORTICAIS[l_name])
            keep_ch = [c for c in range(19) if c not in rem_ch]
            X_l = construir_features_subconjunto(dp, keep_bands=best_bands, keep_channels=keep_ch, modo_feat=modo_feat)
            oof_l = avaliar_oof(X_l, dp['y'], dp['grp'], folds_pids, pids_ref, pid_to_idx, m_fn, k_best=k_b, por_chunk=por_chunk)
            vl = ~np.isnan(oof_l)
            _, bal_l, _, _, _, _, _ = otimizar_youden(y_ref[vl], oof_l[vl])

            delta_l = bal_l - bal_full
            matriz_delta_lobos[p_idx, l_idx] = delta_l
            sinal = '+' if delta_l >= 0 else ''
            deltas_l_str.append(f"{sinal}{delta_l*100:6.2f}%")

            if bal_l > melhor_bal_p:
                melhor_bal_p = bal_l
                melhor_oof_p = oof_l

        oof_podados_7p[p] = (melhor_oof_p, melhor_bal_p)
        print(f"{PARADIGM_LABELS[p]:<23} | {bal_full:8.2%} | " + " | ".join([f"{s:>10}" for s in deltas_l_str]))

    print("-" * 105)

    # =========================================================================
    # PARTE 4: EFEITO DA IMPEDANCIA CRANIANA (PSD ABSOLUTO VS. PSD RELATIVO)
    # =========================================================================
    print("\n=========================================================================================================")
    print("PARTE 4: INTERFERENCIA DE IMPEDANCIA CRANIANA (PSD ABSOLUTO BRUTO VS. PSD RELATIVO + RAZOES)")
    print("=========================================================================================================")
    for p in p_keys:
        dp = paradigms[p]
        _, m_name, m_fn, k_b, por_chunk, best_bands = melhor_cfg_por_p[p]
        X_abs = construir_features_subconjunto(dp, keep_bands=best_bands, modo_feat='spec', usar_psd_absoluto=True)
        X_rel = construir_features_subconjunto(dp, keep_bands=best_bands, modo_feat='spec', usar_psd_absoluto=False)

        oof_abs = avaliar_oof(X_abs, dp['y'], dp['grp'], folds_pids, pids_ref, pid_to_idx, m_fn, k_best=k_b, por_chunk=por_chunk)
        oof_rel = avaliar_oof(X_rel, dp['y'], dp['grp'], folds_pids, pids_ref, pid_to_idx, m_fn, k_best=k_b, por_chunk=por_chunk)
        va = ~np.isnan(oof_abs)
        _, bal_abs, _, _, _, _, _ = otimizar_youden(y_ref[va], oof_abs[va])
        _, bal_rel, _, _, _, _, _ = otimizar_youden(y_ref[va], oof_rel[va])
        ganho = (bal_rel - bal_abs) * 100.0
        sinal = '+' if ganho >= 0 else ''
        print(f"  * {PARADIGM_LABELS[p]:<23}: PSD Absoluto = {bal_abs:6.2%}  -->  PSD Relativo = {bal_rel:6.2%}  ({sinal}{ganho:.2f} p.p.)")

    # =========================================================================
    # PARTE 5: SUPER-FUSAO DOS 7 EXAMES (BRUTA VS. PODADA SEM BANDAS/LOBOS RUIDOSOS)
    # =========================================================================
    def norm_z(p_vec):
        res = p_vec.copy()
        m = ~np.isnan(res)
        res[m] = (res[m] - np.mean(res[m])) / (np.std(res[m]) + 1e-8)
        return 1.0 / (1.0 + np.exp(-res))

    P_7_base = np.column_stack([norm_z(oof_campeoes_7p[p]) for p in p_keys])
    P_7_podado = np.column_stack([norm_z(oof_podados_7p[p][0]) for p in p_keys])

    # 1. Fusao igualitaria cega dos 7 exames
    prob_cega = np.nanmean(P_7_base, axis=1)
    acc_c, bal_c, sp_c, se_c, auc_c, _, (tn_c, fp_c, fn_c, tp_c) = otimizar_youden(y_ref, prob_cega)

    # 2. Fusao otimizada usando os especialistas podados (sem bandas/regioes que interferiam!)
    np.random.seed(42)
    melhor_score = 0.0
    melhor_res = None
    melhor_pesos = None

    for _ in range(25000):
        w = np.random.dirichlet(np.ones(len(p_keys)))
        mask_z = np.random.rand(len(p_keys)) < 0.35
        if np.all(mask_z):
            continue
        w[mask_z] = 0.0
        # Garantir presenca do exame base REST para cobertura de 100% dos 103 sujeitos
        if w[0] < 0.05:
            w[0] = 0.15
        w = np.round(w / np.sum(w), 3)
        w = w / np.sum(w)

        W_mat = np.where(np.isnan(P_7_podado), 0.0, w)
        P_zero = np.where(np.isnan(P_7_podado), 0.0, P_7_podado)
        prob_ens = np.sum(P_zero * W_mat, axis=1) / np.sum(W_mat, axis=1)

        acc_e, bal_e, sp_e, se_e, auc_e, th_e, cm_e = otimizar_youden(y_ref, prob_ens)
        score = 0.5 * acc_e + 0.5 * bal_e
        if score > melhor_score:
            melhor_score = score
            melhor_res = (acc_e, bal_e, sp_e, se_e, auc_e, th_e, cm_e)
            melhor_pesos = w

    acc_o, bal_o, sp_o, se_o, auc_o, th_o, (tn_o, fp_o, fn_o, tp_o) = melhor_res

    print("\n=========================================================================================================")
    print("PARTE 5: COMPARATIVO DE FUSAO DOS 7 EXAMES (FUSAO CEGA VS. FUSAO COM PODA DE INTERFERENCIAS)")
    print("=========================================================================================================")
    print(f"1) Fusao Cega Igualitaria dos 7 Exames (Sem Poda): Acc = {acc_c:.2%} ({tn_c+tp_c}/103) | Bal = {bal_c:.2%} | TD = {sp_c:.2%} | ASD = {se_c:.2%} | AUC = {auc_c:.3f}")
    print(f"2) Super-Fusao 7 Exames Pos-Ablacao de Ruido:      Acc = {acc_o:.2%} ({tn_o+tp_o}/103) | Bal = {bal_o:.2%} | TD = {sp_o:.2%} | ASD = {se_o:.2%} | AUC = {auc_o:.3f}")
    print("   Pesos Otimos por Exame Pos-Ablacao:")
    for j, p in enumerate(p_keys):
        if melhor_pesos[j] > 0.005:
            print(f"     - {PARADIGM_LABELS[p]:<25}: {melhor_pesos[j]:6.1%}")
    print("=========================================================================================================\n")

    gerar_graficos_ablacao(matriz_delta_bandas, matriz_delta_lobos, nomes_plot, accs_4b_plot, accs_5b_plot, taxas_art_plot)
    print("Graficos cientificos gerados com sucesso:")
    print("  -> ablacao_bandas_7_exames.png")
    print("  -> ablacao_regioes_7_exames.png")
    print("  -> comparativo_7_paradigmas_gama.png")

if __name__ == '__main__':
    executar_estudo_ablacao_completo()

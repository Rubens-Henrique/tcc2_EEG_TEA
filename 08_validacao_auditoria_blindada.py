# 08_validacao_auditoria_blindada.py
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score, roc_curve
import config
from importlib import import_module

ab_mod = import_module('07_estudo_diferencial_ablacao')

# Os 4 canais peri-oculares (Fp1, Fp2 acima das sobrancelhas; F7, F8 nos cantos externos dos olhos)
CH_OCULAR = [0, 1, 2, 6]
# Os 15 canais corticais puros (livres de Fp1, Fp2, F7, F8)
CH_15_PUROS = [c for c in range(19) if c not in CH_OCULAR]

def descontar_linear_treino_teste(X_tr, X_te, C_tr, C_te):
    """
    Remove por regressao linear (Ordinary Least Squares ajustado ESTRITAMENTE no conjunto de treino)
    a variancia explicada pelas variaveis de controle C (ex.: Sexo e Idade).
    """
    C_tr_1 = np.column_stack([np.ones(len(C_tr)), C_tr])
    C_te_1 = np.column_stack([np.ones(len(C_te)), C_te])
    beta, _, _, _ = np.linalg.lstsq(C_tr_1, X_tr, rcond=None)
    return X_tr - C_tr_1 @ beta, X_te - C_te_1 @ beta

def construir_matriz_sujeito_auditada(d5, pid_to_idx, p, keep_bands, ch_use, modo, descontar_dipolo_eog=False):
    """
    Extrai a matriz (103, N_features) para o paradigma p.
    Se descontar_dipolo_eog=True, projeta ortogonalmente para fora da matriz 19x19 de cada sujeito
    os dois dipolos oculares classicos:
      1) Piscada Vertical (VEOG): eixo Fp1-Fp2 (canais 0 e 1)
      2) Sacada Horizontal (HEOG): eixo F7-F8 (canais 2 e 6)
    ANTES de extrair os 15 canais corticais puros e projetar no Espaco Tangente Riemanniano.
    """
    dp = {k.replace(f'{p}_', ''): d5[k] for k in d5.files if k.startswith(f'{p}_')}
    conn_m = dp['conn_med'].copy()
    grp = dp['grp']
    pids_p = pd.unique(grp)

    if descontar_dipolo_eog:
        for i in range(conn_m.shape[0]):
            for b in range(conn_m.shape[1]):
                Z = conn_m[i, b]
                R = np.tanh(Z)
                np.fill_diagonal(R, 1.0)
                v_blink = 0.5 * (R[:, 0] + R[:, 1])
                v_sacc  = 0.5 * (R[:, 2] - R[:, 6])
                R_clean = R - 0.5 * np.outer(v_blink, v_blink) - 0.5 * np.outer(v_sacc, v_sacc)
                d_sq = np.sqrt(np.maximum(np.diag(R_clean), 1e-4))
                R_clean = np.clip(R_clean / np.outer(d_sq, d_sq), -0.999, 0.999)
                Z_clean = np.arctanh(R_clean)
                np.fill_diagonal(Z_clean, 0.0)
                conn_m[i, b] = Z_clean

    dp_mod = dict(dp)
    dp_mod['conn_med'] = conn_m
    Xc = ab_mod.construir_features_subconjunto(dp_mod, keep_bands=keep_bands, keep_channels=ch_use, modo_feat=modo)
    Xs = np.array([np.median(Xc[grp == pid], axis=0) for pid in pids_p], dtype=np.float32)

    out = np.full((103, Xs.shape[1]), np.nan, dtype=np.float32)
    for i, pid in enumerate(pids_p):
        if pid in pid_to_idx:
            out[pid_to_idx[pid]] = Xs[i]
    return out

def executar_auditoria_blindada():
    print("=============================================================================================================")
    print("FASE 6: SOLUCAO DEFINITIVA DA AUDITORIA — GEOMETRIA RIEMANNIANA SEM EOG + EMPILHAMENTO ANINHADO (10 SORTEIOS)")
    print("Canais Removidos: Fp1, Fp2, F7, F8 | Canais Mantidos: 15 Eletrodos Corticais Puros | Limiar Fixo: 0.50")
    print("=============================================================================================================\n")

    d5 = np.load(config.CACHE_7P_5B_PATH, allow_pickle=True)
    pids_103 = pd.unique(d5['rest_grp'])
    y_103 = np.array([d5['rest_y'][d5['rest_grp'] == pid][0] for pid in pids_103], dtype=np.int32)
    pid_to_idx = {pid: i for i, pid in enumerate(pids_103)}

    df_part = pd.read_csv(config.PARTICIPANTS_FILE, sep='\t').set_index('participant_id')
    sex_103 = np.array([1.0 if df_part.loc[p, 'sex'] == 'm' else 0.0 for p in pids_103], dtype=np.float32)
    age_103 = np.array([float(df_part.loc[p, 'age']) for p in pids_103], dtype=np.float32)
    X_demo = np.column_stack([sex_103, age_103])

    # Matrizes de REST para comparacao direta com a Tabela da Imagem 2 da Auditoria
    X_rest_spec_19     = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'rest', [0,1,2,3], list(range(19)), 'spec', False)
    X_rest_spec_15     = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'rest', [0,1,2,3], CH_15_PUROS,     'spec', False)
    X_rest_riem_15     = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'rest', [0,1,2,3], CH_15_PUROS,     'riem', False)
    X_rest_riem_15_dip = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'rest', [0,1,2,3], CH_15_PUROS,     'riem', True)

    print("TABELA 1 (RESPOSTA A IMAGEM 2): TAREFA REPOUSO (103 CRIANCAS, 10 SORTEIOS) — GEOMETRIA RIEMANNIANA SEM EOG")
    print("-" * 109)
    print(f"{'Modelo':<20} | {'Condicao Experimental':<50} | {'AUC (Media +- DP)':<18} | {'F1-TEA (Media +- DP)'}")
    print("-" * 109)

    modelos_tab1 = [
        ('Regressao Logistica', lambda: LogisticRegression(C=0.15, class_weight='balanced', max_iter=500, random_state=42)),
        ('SVM RBF',             lambda: SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
        ('Floresta Aleatoria',  lambda: RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)),
        ('Gradient Boosting',   lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=4, class_weight='balanced', random_state=42)),
    ]

    for m_nome, clf_fn in modelos_tab1:
        condicoes = [
            ('Piso Demografico: Sexo e Idade', X_demo, None, 2),
            ('Conectividade/Espectral (19 canais - Original)', X_rest_spec_19, None, 35),
            ('Conectividade/Espectral (15ch - Sem Fp1,Fp2,F7,F8)', X_rest_spec_15, None, 35),
            ('SOLUCAO: Riemann+Grafo 15ch (Sem Fp1,Fp2,F7,F8)', X_rest_riem_15, None, 35),
            ('SOLUCAO: Riemann 15ch + Descontando Dipolo F7-F8/Fp1-Fp2', X_rest_riem_15_dip, None, 35),
            ('SOLUCAO: Riemann 15ch + Descontando Sexo e Idade', X_rest_riem_15, X_demo, 35),
        ]
        for cond_nome, X_mat, C_ctrl, k_sel in condicoes:
            aucs, f1s = [], []
            for s in range(10):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + s)
                oof = np.zeros(103)
                for tr, te in skf.split(X_mat, y_103):
                    Xtr, Xte = X_mat[tr].copy(), X_mat[te].copy()
                    if C_ctrl is not None:
                        Xtr, Xte = descontar_linear_treino_teste(Xtr, Xte, C_ctrl[tr], C_ctrl[te])
                    sc = RobustScaler()
                    Xtr = sc.fit_transform(Xtr)
                    Xte = sc.transform(Xte)
                    if Xtr.shape[1] > k_sel:
                        sel = SelectKBest(f_classif, k=k_sel)
                        Xtr = sel.fit_transform(Xtr, y_103[tr])
                        Xte = sel.transform(Xte)
                    clf = clf_fn()
                    clf.fit(Xtr, y_103[tr])
                    oof[te] = clf.predict_proba(Xte)[:, 1]
                aucs.append(roc_auc_score(y_103, oof))
                f1s.append(f1_score(y_103, (oof >= 0.50).astype(int)))
            print(f"{m_nome:<20} | {cond_nome:<50} | {np.mean(aucs):.3f} +- {np.std(aucs):.3f}     | {np.mean(f1s):.3f} +- {np.std(f1s):.3f}")
        print("-" * 109)

    # =========================================================================
    # TABELAS 2 E 3 (RESPOSTA A IMAGEM 1): SUPER-FUSAO MULTI-PARADIGMA BLINDADA
    # =========================================================================
    X_fast_15      = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'fast',  [0,1,2,3,4], CH_15_PUROS, 'comb', False)
    X_fast_15_dip  = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'fast',  [0,1,2,3,4], CH_15_PUROS, 'comb', True)
    X_avsrt_15     = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'avsrt', [0,1,2,3,4], CH_15_PUROS, 'comb', False)
    X_avsrt_15_dip = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'avsrt', [0,1,2,3,4], CH_15_PUROS, 'comb', True)
    X_ic_15        = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'ic',    [0,1,2,3],   CH_15_PUROS, 'spec', False)
    X_ic_15_dip    = construir_matriz_sujeito_auditada(d5, pid_to_idx, 'ic',    [0,1,2,3],   CH_15_PUROS, 'spec', True)

    def montar_pool_especialistas(usar_dipolo_descontado=False):
        Xr = X_rest_riem_15_dip if usar_dipolo_descontado else X_rest_riem_15
        Xf = X_fast_15_dip if usar_dipolo_descontado else X_fast_15
        Xa = X_avsrt_15_dip if usar_dipolo_descontado else X_avsrt_15
        Xi = X_ic_15_dip if usar_dipolo_descontado else X_ic_15
        return [
            ('REST_Riem_HGB_15ch',    Xr, 35, lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=4, class_weight='balanced', random_state=42)),
            ('REST_Riem_LogReg_15ch', Xr, 35, lambda: LogisticRegression(C=0.15, class_weight='balanced', max_iter=500, random_state=42)),
            ('REST_Riem_RF_15ch',     Xr, 35, lambda: RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)),
            ('FAST_15ch_LogReg',      Xf, 35, lambda: LogisticRegression(C=0.15, class_weight='balanced', max_iter=500, random_state=42)),
            ('AVSRT_15ch_LogReg',     Xa, 35, lambda: LogisticRegression(C=0.15, class_weight='balanced', max_iter=500, random_state=42)),
            ('IC_15ch_LogReg',        Xi, 35, lambda: LogisticRegression(C=0.15, class_weight='balanced', max_iter=500, random_state=42)),
        ]

    def avaliar_fusao_aninhada(pool_esp, descontar_demo=False):
        aucs_eq, accs_eq, bals_eq, f1s_eq = [], [], [], []
        aucs_st, accs_st, bals_st, f1s_st = [], [], [], []

        for s in range(10):
            skf_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + s)
            oof_eq = np.zeros(103)
            oof_st = np.zeros(103)

            for tr_idx, te_idx in skf_out.split(np.zeros(103), y_103):
                n_esp = len(pool_esp)
                inner_oof_tr = np.full((len(tr_idx), n_esp), np.nan)
                preds_te = np.full((len(te_idx), n_esp), np.nan)
                skf_in = StratifiedKFold(n_splits=4, shuffle=True, random_state=100 + s)

                for j, (e_name, X_full, k_b, clf_fn) in enumerate(pool_esp):
                    for itr_sub, ival_sub in skf_in.split(np.zeros(len(tr_idx)), y_103[tr_idx]):
                        r_itr, r_ival = tr_idx[itr_sub], tr_idx[ival_sub]
                        v_tr, v_val = ~np.isnan(X_full[r_itr, 0]), ~np.isnan(X_full[r_ival, 0])
                        if np.sum(v_tr) < 15 or np.sum(v_val) == 0:
                            continue
                        X_itr, y_itr, X_ival = X_full[r_itr[v_tr]].copy(), y_103[r_itr[v_tr]], X_full[r_ival[v_val]].copy()
                        if descontar_demo:
                            X_itr, X_ival = descontar_linear_treino_teste(X_itr, X_ival, X_demo[r_itr[v_tr]], X_demo[r_ival[v_val]])
                        sc = RobustScaler()
                        X_itr, X_ival = sc.fit_transform(X_itr), sc.transform(X_ival)
                        sel = SelectKBest(f_classif, k=min(k_b, X_itr.shape[1]))
                        X_itr, X_ival = sel.fit_transform(X_itr, y_itr), sel.transform(X_ival)
                        c = clf_fn()
                        c.fit(X_itr, y_itr)
                        inner_oof_tr[ival_sub[v_val], j] = c.predict_proba(X_ival)[:, 1]

                    v_otr, v_ote = ~np.isnan(X_full[tr_idx, 0]), ~np.isnan(X_full[te_idx, 0])
                    if np.sum(v_otr) >= 15 and np.sum(v_ote) > 0:
                        X_otr, y_otr, X_ote = X_full[tr_idx[v_otr]].copy(), y_103[tr_idx[v_otr]], X_full[te_idx[v_ote]].copy()
                        if descontar_demo:
                            X_otr, X_ote = descontar_linear_treino_teste(X_otr, X_ote, X_demo[tr_idx[v_otr]], X_demo[te_idx[v_ote]])
                        sc = RobustScaler()
                        X_otr, X_ote = sc.fit_transform(X_otr), sc.transform(X_ote)
                        sel = SelectKBest(f_classif, k=min(k_b, X_otr.shape[1]))
                        X_otr, X_ote = sel.fit_transform(X_otr, y_otr), sel.transform(X_ote)
                        c = clf_fn()
                        c.fit(X_otr, y_otr)
                        preds_te[v_ote, j] = c.predict_proba(X_ote)[:, 1]

                # A) PESOS IGUAIS (Media simples das probabilidades das especialistas)
                oof_eq[te_idx] = np.nanmean(preds_te, axis=1)

                # B) EMPILHAMENTO ESTRITAMENTE ANINHADO NAO-NEGATIVO (Pesos por AUC Inner-CV no Treino + Calibracao 0.50)
                w_in = np.zeros(n_esp)
                for j in range(n_esp):
                    m_j = ~np.isnan(inner_oof_tr[:, j])
                    if np.sum(m_j) > 10:
                        auc_j = roc_auc_score(y_103[tr_idx[m_j]], inner_oof_tr[m_j, j])
                        w_in[j] = max(auc_j - 0.56, 0.0) ** 3
                if np.sum(w_in) <= 1e-8:
                    w_in = np.ones(n_esp)
                w_in = w_in / np.sum(w_in)

                W_tr = np.where(np.isnan(inner_oof_tr), 0.0, w_in)
                s_tr = np.sum(np.where(np.isnan(inner_oof_tr), 0.0, inner_oof_tr) * W_tr, axis=1) / np.maximum(np.sum(W_tr, axis=1), 1e-8)

                W_te = np.where(np.isnan(preds_te), 0.0, w_in)
                s_te = np.sum(np.where(np.isnan(preds_te), 0.0, preds_te) * W_te, axis=1) / np.maximum(np.sum(W_te, axis=1), 1e-8)

                cal = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
                cal.fit(s_tr.reshape(-1, 1), y_103[tr_idx])
                oof_st[te_idx] = cal.predict_proba(s_te.reshape(-1, 1))[:, 1]

            aucs_eq.append(roc_auc_score(y_103, oof_eq))
            preds_eq = (oof_eq >= 0.50).astype(int)
            accs_eq.append(accuracy_score(y_103, preds_eq))
            bals_eq.append(balanced_accuracy_score(y_103, preds_eq))
            f1s_eq.append(f1_score(y_103, preds_eq))

            aucs_st.append(roc_auc_score(y_103, oof_st))
            preds_st = (oof_st >= 0.50).astype(int)
            accs_st.append(accuracy_score(y_103, preds_st))
            bals_st.append(balanced_accuracy_score(y_103, preds_st))
            f1s_st.append(f1_score(y_103, preds_st))

        return {
            'auc_eq': (np.mean(aucs_eq), np.std(aucs_eq)),
            'acc_eq': np.mean(accs_eq), 'bal_eq': np.mean(bals_eq), 'f1_eq': np.mean(f1s_eq),
            'auc_st': (np.mean(aucs_st), np.std(aucs_st)),
            'acc_st': np.mean(accs_st), 'bal_st': np.mean(bals_st), 'f1_st': np.mean(f1s_st),
        }

    r_15_puro = avaliar_fusao_aninhada(montar_pool_especialistas(False), False)
    r_15_dip  = avaliar_fusao_aninhada(montar_pool_especialistas(True),  False)
    r_15_demo = avaliar_fusao_aninhada(montar_pool_especialistas(False), True)

    print("\nTABELA 2 E 3 (RESPOSTA A IMAGEM 1): SUPER-FUSAO 15 CANAIS CORTICAIS PUROS (10 SORTEIOS | LIMIAR 0.50 FIXO)")
    print("-" * 109)
    print(f"{'Condicao Auditada (15 Canais Sem Fp1,Fp2,F7,F8)':<48} | {'Pesos Iguais (Limiar 0.50)':<28} | {'Empilhamento Aninhado (0.50)'}")
    print("-" * 109)
    for rotulo, r in [
        ('1. Sem Canais Oculares (15 Canais Puros)', r_15_puro),
        ('2. Sem Canais Oculares + Desconto Dipolo F7-F8/Fp1-Fp2', r_15_dip),
        ('3. Sem Canais Oculares + Descontando Sexo e Idade', r_15_demo),
    ]:
        s_eq = f"AUC {r['auc_eq'][0]:.3f} | Acc {r['acc_eq']:.1%} | F1 {r['f1_eq']:.3f}"
        s_st = f"AUC {r['auc_st'][0]:.3f} | Acc {r['acc_st']:.1%} | Bal {r['bal_st']:.1%} | F1 {r['f1_st']:.3f}"
        print(f"{rotulo:<48} | {s_eq:<28} | {s_st}")
    print("-" * 109)

if __name__ == '__main__':
    executar_auditoria_blindada()

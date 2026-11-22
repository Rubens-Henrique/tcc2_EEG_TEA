# 06_extracao_7_paradigmas_5_bandas.py
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfiltfilt, welch
import config

PAIR_BANDS_5 = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4),
    (3, 4)
]

def localizar_arquivos_sujeito(pid, task_tag, fmt):
    eeg_dir = os.path.join(config.DATASET_DIR, pid, 'eeg')
    if not os.path.exists(eeg_dir):
        return []

    ext = '_eeg.bdf' if fmt == 'bdf' else '_eeg.vhdr'
    alvo = f'task-{task_tag.lower()}_'
    encontrados = []
    for fname in sorted(os.listdir(eeg_dir)):
        fl = fname.lower()
        if fl.endswith(ext) and alvo in fl:
            encontrados.append(os.path.join(eeg_dir, fname))

    # Para o paradigma Motor (~10 runs por sujeito, 25 GB totais), selecionamos ate 2 runs longos
    # de caminhada (~5-10 min por sujeito) para manter paridade estatistica com os demais paradigmas
    if fmt == 'vhdr' and len(encontrados) > 2:
        longos = []
        for fp in encontrados:
            try:
                r_tmp = mne.io.read_raw_brainvision(fp, preload=False, verbose=False)
                if r_tmp.times[-1] >= 90.0:
                    longos.append(fp)
                if len(longos) == 2:
                    break
            except Exception:
                continue
        if longos:
            return longos
        return encontrados[:3]

    return encontrados

def ler_e_preprocessar_arquivo(fpath, fmt, canais_alvo):
    if fmt == 'bdf':
        raw = mne.io.read_raw_bdf(fpath, include=canais_alvo, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_brainvision(fpath, preload=False, verbose=False)
        presentes = [c for c in canais_alvo if c in raw.ch_names]
        if len(presentes) < len(canais_alvo):
            return None
        raw.pick(presentes)
        raw.load_data(verbose=False)

    if len(raw.ch_names) < len(canais_alvo):
        return None
    raw.reorder_channels(canais_alvo)

    sfreq = float(raw.info['sfreq'])
    d = raw.get_data().astype(np.float64)
    if d.shape[1] < int(sfreq * 4):
        return None

    # Referencia Media Comum (CAR) nos 19 canais do Sistema 10-20
    d = d - np.mean(d, axis=0, keepdims=True)

    # Filtro passa-faixa global 0.5 - 45.0 Hz
    sos_broad = butter(4, [0.5, 45.0], btype='bandpass', fs=sfreq, output='sos')
    d = sosfiltfilt(sos_broad, d, axis=1)

    # Reamostragem limpa para 256 Hz
    if abs(sfreq - 512.0) < 1e-2:
        d = d[:, ::2]
    elif abs(sfreq - 256.0) >= 1e-2:
        d = mne.filter.resample(d, down=sfreq / 256.0, npad='auto', verbose=False)

    return d.astype(np.float32)

def extrair_epocas_5_bandas(d_list):
    """
    Recebe lista de arrays (19, n_samples) a 256 Hz de um sujeito, fatia epocas de 2s com 50% overlap,
    e extrai para as 5 bandas (Delta, Theta, Alpha, Beta, Gamma):
    - conns_z: (N_ep, 5, 19, 19) com arctanh(r)
    - psds:    (N_ep, 5, 19) com potencia espectral de Welch
    """
    win = int(config.EPOCH_DURATION * config.SFREQ_RESAMPLE)          # 512 amostras (2.0 s)
    step = int((config.EPOCH_DURATION - config.EPOCH_OVERLAP) * config.SFREQ_RESAMPLE)  # 256 amostras (1.0 s)
    band_limits = list(config.BANDS_5.values())

    ep_conns_all = []
    ep_psds_all = []

    for d in d_list:
        if d.shape[1] < win:
            continue
        n_ep = (d.shape[1] - win) // step + 1
        idx = np.arange(win)[None, :] + step * np.arange(n_ep)[:, None]
        ep_broad = d[:, idx].transpose(1, 0, 2)  # (n_ep, 19, 512)

        freqs, Pxx = welch(ep_broad, fs=256.0, nperseg=256, axis=2)

        conns_this = np.zeros((n_ep, 5, 19, 19), dtype=np.float32)
        psds_this = np.zeros((n_ep, 5, 19), dtype=np.float32)

        for b_idx, (fmin, fmax) in enumerate(band_limits):
            sos_b = butter(4, [fmin, fmax], btype='bandpass', fs=256.0, output='sos')
            d_b = sosfiltfilt(sos_b, d, axis=1).astype(np.float32)
            ep_b = d_b[:, idx].transpose(1, 0, 2)  # (n_ep, 19, 512)

            ep_mean = np.mean(ep_b, axis=2, keepdims=True)
            ep_std = np.std(ep_b, axis=2, keepdims=True) + 1e-12
            ep_norm = (ep_b - ep_mean) / ep_std

            corr = np.matmul(ep_norm, ep_norm.transpose(0, 2, 1)) / float(win)
            corr = np.clip(corr, -0.999, 0.999)
            z_mat = np.arctanh(corr)
            # Zerar diagonal principal
            z_mat[:, np.arange(19), np.arange(19)] = 0.0
            conns_this[:, b_idx] = np.nan_to_num(z_mat, nan=0.0, posinf=0.0, neginf=0.0)

            f_mask = (freqs >= fmin) & (freqs <= fmax)
            psd_b = np.mean(Pxx[:, :, f_mask], axis=2)
            psds_this[:, b_idx] = np.maximum(np.nan_to_num(psd_b, nan=1e-12), 1e-12)

        ep_conns_all.append(conns_this)
        ep_psds_all.append(psds_this)

    if not ep_conns_all:
        return None, None

    return np.concatenate(ep_conns_all, axis=0), np.concatenate(ep_psds_all, axis=0)

def processar_um_sujeito(args):
    pid, label, task_tag, fmt, n_chunks = args
    fpaths = localizar_arquivos_sujeito(pid, task_tag, fmt)
    if not fpaths:
        return None

    d_list = []
    for fp in fpaths:
        try:
            d = ler_e_preprocessar_arquivo(fp, fmt, config.CANAIS_10_20)
            if d is not None:
                d_list.append(d)
        except Exception:
            continue

    if not d_list:
        return None

    conns_ep, psds_ep = extrair_epocas_5_bandas(d_list)
    if conns_ep is None or len(conns_ep) < n_chunks * 5:
        return None

    tam_chunk = len(conns_ep) // n_chunks
    c_conn_med, c_conn_iqr = [], []
    c_psd5_med, c_psd5_iqr = [], []
    c_psd_abs_med = []
    c_rat_med, c_rat_iqr = [], []
    taxas_art = []

    for c in range(n_chunks):
        ini = c * tam_chunk
        fim = (c + 1) * tam_chunk if c < n_chunks - 1 else len(conns_ep)

        conn_blk = conns_ep[ini:fim]  # (M, 5, 19, 19)
        psd_blk = psds_ep[ini:fim]    # (M, 5, 19)

        # Energia global por epoca (todas as 5 bandas e 19 canais)
        energia = np.mean(np.sum(psd_blk, axis=1), axis=1)
        q25, q75 = np.percentile(energia, [25, 75])
        iqr = q75 - q25
        mask_limpa = (energia >= (q25 - 1.5 * iqr)) & (energia <= (q75 + 1.5 * iqr))
        taxa_rejeicao = 1.0 - (np.sum(mask_limpa) / float(len(energia)))
        taxas_art.append(taxa_rejeicao)

        if np.sum(mask_limpa) < 3:
            mask_limpa = np.ones(len(energia), dtype=bool)

        conn_clean = conn_blk[mask_limpa]
        psd_clean = psd_blk[mask_limpa]

        # Potencia relativa nas 5 bandas (soma = 1 em cada eletrodo e epoca)
        psd_sum5 = np.sum(psd_clean, axis=1, keepdims=True) + 1e-12
        psd_rel5 = psd_clean / psd_sum5

        # Todas as 10 razoes log10 entre pares de bandas (5 escolhe 2 = 10)
        ratios_ep = np.zeros((len(psd_clean), 10, 19), dtype=np.float32)
        for r_idx, (b1, b2) in enumerate(PAIR_BANDS_5):
            ratios_ep[:, r_idx, :] = np.log10(psd_clean[:, b1, :] / psd_clean[:, b2, :])
        ratios_ep = np.nan_to_num(ratios_ep, nan=0.0, posinf=0.0, neginf=0.0)

        c_conn_med.append(np.median(conn_clean, axis=0))
        c_conn_iqr.append(np.percentile(conn_clean, 75, axis=0) - np.percentile(conn_clean, 25, axis=0))

        c_psd5_med.append(np.median(psd_rel5, axis=0))
        c_psd5_iqr.append(np.percentile(psd_rel5, 75, axis=0) - np.percentile(psd_rel5, 25, axis=0))
        c_psd_abs_med.append(np.median(np.log10(psd_clean), axis=0))

        c_rat_med.append(np.median(ratios_ep, axis=0))
        c_rat_iqr.append(np.percentile(ratios_ep, 75, axis=0) - np.percentile(ratios_ep, 25, axis=0))

    return {
        'pid': pid,
        'label': label,
        'n_epocas': len(conns_ep),
        'taxa_art': float(np.mean(taxas_art)),
        'conn_med': np.array(c_conn_med, dtype=np.float32),
        'conn_iqr': np.array(c_conn_iqr, dtype=np.float32),
        'psd5_med': np.array(c_psd5_med, dtype=np.float32),
        'psd5_iqr': np.array(c_psd5_iqr, dtype=np.float32),
        'psd_abs_med': np.array(c_psd_abs_med, dtype=np.float32),
        'rat_med': np.array(c_rat_med, dtype=np.float32),
        'rat_iqr': np.array(c_rat_iqr, dtype=np.float32),
    }

def executar_extracao_completa_7p_5b(n_chunks=6, max_workers=4):
    print("=======================================================================================")
    print("FASE 5: EXTRACAO PARALELA DOS 7 PARADIGMAS BIDS EM 5 BANDAS (DELTA, THETA, ALPHA, BETA, GAMMA)")
    print("=======================================================================================")

    df_part = pd.read_csv(config.PARTICIPANTS_FILE, sep='\t')
    df_asd_td = df_part[df_part['group'].isin(['ASD', 'TD'])].copy().reset_index(drop=True)

    cache_dict = {}
    if os.path.exists(config.CACHE_7P_5B_PATH):
        try:
            old = np.load(config.CACHE_7P_5B_PATH, allow_pickle=True)
            for k in old.files:
                cache_dict[k] = old[k]
            print(f"[CACHE] Arquivo existente detectado com chaves ja processadas.")
        except Exception:
            cache_dict = {}

    t_global = time.time()

    for p_key, (task_tag, fmt, comp_col) in config.PARADIGMS_7.items():
        if f'{p_key}_grp' in cache_dict:
            n_suj_cached = len(pd.unique(cache_dict[f'{p_key}_grp']))
            print(f"---> [{p_key.upper()}] Ja extraido no cache ({n_suj_cached} sujeitos). Pulando...")
            continue

        df_elig = df_asd_td[df_asd_td[comp_col] == 'yes'].copy().reset_index(drop=True)
        tarefas = [
            (row['participant_id'], 1 if row['group'] == 'ASD' else 0, task_tag, fmt, n_chunks)
            for _, row in df_elig.iterrows()
        ]

        print(f"\n---> Extraindo [{p_key.upper()}] ({len(tarefas)} sujeitos elegiveis, formato .{fmt})...")
        t0 = time.time()
        resultados_p = []

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(processar_um_sujeito, t): t[0] for t in tarefas}
            concluidos = 0
            for fut in as_completed(futs):
                concluidos += 1
                res = fut.result()
                if res is not None:
                    resultados_p.append(res)
                if concluidos % 20 == 0 or concluidos == len(tarefas):
                    print(f"     [{p_key.upper()}] Progresso: {concluidos}/{len(tarefas)} sujeitos ({time.time()-t0:.1f}s)")

        # Ordenar de forma deterministica pela ordem original de df_elig
        ordem_pids = {pid: idx for idx, pid in enumerate(df_elig['participant_id'].values)}
        resultados_p.sort(key=lambda r: ordem_pids.get(r['pid'], 9999))

        conn_med = np.concatenate([r['conn_med'] for r in resultados_p], axis=0)
        conn_iqr = np.concatenate([r['conn_iqr'] for r in resultados_p], axis=0)
        psd5_med = np.concatenate([r['psd5_med'] for r in resultados_p], axis=0)
        psd5_iqr = np.concatenate([r['psd5_iqr'] for r in resultados_p], axis=0)
        psd_abs_med = np.concatenate([r['psd_abs_med'] for r in resultados_p], axis=0)
        rat_med = np.concatenate([r['rat_med'] for r in resultados_p], axis=0)
        rat_iqr = np.concatenate([r['rat_iqr'] for r in resultados_p], axis=0)
        y_vec = np.concatenate([np.full(n_chunks, r['label'], dtype=np.int32) for r in resultados_p])
        grp_vec = np.concatenate([np.full(n_chunks, r['pid'], dtype=object) for r in resultados_p])
        art_suj = np.array([r['taxa_art'] for r in resultados_p], dtype=np.float32)
        ep_suj = np.array([r['n_epocas'] for r in resultados_p], dtype=np.int32)

        cache_dict[f'{p_key}_conn_med'] = conn_med
        cache_dict[f'{p_key}_conn_iqr'] = conn_iqr
        cache_dict[f'{p_key}_psd5_med'] = psd5_med
        cache_dict[f'{p_key}_psd5_iqr'] = psd5_iqr
        cache_dict[f'{p_key}_psd_abs_med'] = psd_abs_med
        cache_dict[f'{p_key}_rat_med'] = rat_med
        cache_dict[f'{p_key}_rat_iqr'] = rat_iqr
        cache_dict[f'{p_key}_y'] = y_vec
        cache_dict[f'{p_key}_grp'] = grp_vec
        cache_dict[f'{p_key}_art'] = art_suj
        cache_dict[f'{p_key}_ep'] = ep_suj

        np.savez_compressed(config.CACHE_7P_5B_PATH, **cache_dict)
        print(f"     [OK] {p_key.upper()}: {len(resultados_p)} sujeitos validos | {np.sum(ep_suj)} epocas | "
              f"Artefatos rejeitados: {np.mean(art_suj):.1%} | Tempo: {time.time()-t0:.1f}s")

    print("\n=======================================================================================")
    print(f"EXTRACAO DOS 7 PARADIGMAS EM 5 BANDAS CONCLUIDA EM {(time.time()-t_global)/60.0:.2f} MINUTOS!")
    print(f"Arquivo salvo em: {config.CACHE_7P_5B_PATH}")
    print("=======================================================================================")

if __name__ == '__main__':
    executar_extracao_completa_7p_5b(n_chunks=6, max_workers=4)

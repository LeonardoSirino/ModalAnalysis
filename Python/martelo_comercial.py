import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILES_DIR = r'G:\OneDrive - Lactec\00 PROJETOS\COSERN\Ensaios\Ponteiras novas\Haste'
f = np.linspace(0, 1000, num=501)


def get_files(files_dir):
    folders = os.listdir(files_dir)
    files = []
    for folder in folders:
        folder_path = os.path.join(files_dir, folder)
        file_path = os.path.join(folder_path, 'text_data.txt')

        file = open(file_path, 'r')
        files.append(file)

    return files


def calc_bandwidth(hammer_fft, f):
    hammer_level = np.mean(hammer_fft[30:40])
    thr = hammer_level * 0.707

    offset = 40

    zero_crossings = np.where(np.diff(np.sign(hammer_fft[offset:] - thr)))[0]
    try:
        f_band = f[zero_crossings[0] + offset]
    except:
        f_band = 1000

    return f_band


def extract_FRFs(file, channel):
    data = file.read()
    blocks = data.split('fs:\n')
    all_FRF = []
    legend = []
    for block in blocks[1:]:
        [fs, data] = block.split('\ndata:\n')
        fs = float(fs.replace(',', '.'))

        data = data.replace(',', '.')
        data = [line.split('\t') for line in data.split('\n')[:-2]]
        data = np.array(data)
        data = data.astype(np.float)

        hammer = data[:, 0]
        acel = data[:, channel+1]

        hammer_fft = abs(np.fft.fft(hammer))
        acel_fft = abs(np.fft.fft(acel))

        hl = int(len(hammer_fft) / 2)
        f_fft = np.fft.fftfreq(len(hammer), d=1/fs)
        f_fft = f_fft[:hl]

        hammer_fft = hammer_fft[:hl]
        acel_fft = acel_fft[:hl]

        acel_fft = np.interp(f, f_fft, acel_fft)
        hammer_fft = np.interp(f, f_fft, hammer_fft)

        # Critérios de aceitação do impacto
        f_band = calc_bandwidth(hammer_fft, f)
        std_hf = np.std(hammer_fft[50:])

        FRF = acel_fft / hammer_fft

        # plt.plot(hammer_fft)
        legend.append(str(round(f_band, 1)) + ' Hz - ' + str(round(std_hf, 3)))

        if f_band > 50 and std_hf < 2:
            all_FRF.append(FRF)

    # plt.ylim((0, 40))
    # plt.legend(legend)
    # plt.show()

    median_FRF = np.median(all_FRF, axis=0)

    x = []
    y = []

    filtered = []

    for frf in all_FRF:
        error = np.max(abs(frf - median_FRF)) / np.max(median_FRF)
        print(error)
        if error < 10:
            x = np.concatenate((x, f[10:]))
            y = np.concatenate((y, frf[10:]))

            filtered.append(frf)

    filtered = np.array(filtered)
    mean_frf = np.mean(filtered, axis=0)

    sns.lineplot(x, y)
    # plt.plot(x, y)


if __name__ == '__main__':
    files = get_files(FILES_DIR)
    legend = []
    df = pd.DataFrame()
    df['freq'] = f
    for file in files:
        mean_frf = extract_FRFs(file, 2)
        legend.append(file.name.split('\\')[-2])

        df[file.name.split('\\')[-2]] = mean_frf

    df.to_csv('test.csv')

    plt.legend(legend)
    plt.title('Comparação de FRFs')
    plt.ylabel('FRF')
    plt.xlabel('Frequência [Hz]')
    plt.show()

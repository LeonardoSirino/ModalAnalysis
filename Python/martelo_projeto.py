import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILES_DIR = r'G:\OneDrive - Lactec\00 PROJETOS\COSERN\Ensaios\Ponteiras novas\Temp'


def get_files(files_dir):
    folders = os.listdir(files_dir)
    files = []
    for folder in folders:
        folder_path = os.path.join(files_dir, folder)
        file_path = os.path.join(folder_path, 'text_data.txt')

        file = open(file_path, 'r')
        files.append(file)

    return files


def extract_FRFs(file, channel):
    data = file.read()
    blocks = data.split('fs:\n')
    f = np.linspace(0, 1000, num=501)
    all_FRF = []
    for block in blocks[1:]:
        [fs, data] = block.split('\ndata:\n')
        fs = float(fs.replace(',', '.'))

        data = data.replace(',', '.')
        data = [line.split('\t') for line in data.split('\n')[:-2]]
        data = np.array(data)
        data = data.astype(np.float)

        acel = data[:, channel]

        acel_fft = abs(np.fft.fft(acel))

        hl = int(len(acel_fft) / 2)
        f_fft = np.fft.fftfreq(len(acel), d=1/fs)
        f_fft = f_fft[:hl]

        acel_fft = acel_fft[:hl]

        acel_fft = np.interp(f, f_fft, acel_fft)

        all_FRF.append(acel_fft)

    median_FRF = np.median(all_FRF, axis=0)

    x = []
    y = []

    legend = []

    for frf in all_FRF:
        error = np.max(abs(frf - median_FRF)) / np.max(median_FRF)
        # plt.plot(f[10:], frf[10:])
        # legend.append(str(round(error, 3)))
        if error < 0.5:
            x = np.concatenate((x, f[10:]))
            y = np.concatenate((y, frf[10:]))

    # plt.plot(f[10:], median_FRF[10:])
    # legend.append('Median')

    # plt.legend(legend)
    # plt.show()
    # plt.clf()

    sns.lineplot(x, y)
    # plt.plot(x, y)


if __name__ == '__main__':
    files = get_files(FILES_DIR)
    legend = []
    for file in files:
        extract_FRFs(file, 2)
        legend.append(file.name.split('\\')[-2])

    plt.legend(legend)
    plt.title('Comparação de FRFs')
    plt.ylabel('FRF')
    plt.xlabel('Frequência [Hz]')
    plt.show()

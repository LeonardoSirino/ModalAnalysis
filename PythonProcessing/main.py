import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

test_dir = r'L:\Documentos\Programas em LabVIEW\Programas funcionais\ModalAnalysis\PythonProcessing\sample_data\QE12R'

plt.rcParams.update({'font.size': 18})

for test_name in os.listdir(test_dir):
    try:
        file_path = os.path.join(test_dir, test_name)
        print('Iniciando ' + test_name)

        plt.clf()

        file = open(file_path, 'r')
        all_tests = []
        for test in file:
            test_data = json.loads(test)
            all_tests.append(test_data)
        file.close()

        hammer_data = []
        freq_data = []
        all_inerts = None
        MIN_FREQ = 10
        MAX_FREQ = 90
        WINDOW_LENGTH = 5

        for test in all_tests:
            dt = test['dt']
            hammer = test['hammer']
            df = 1 / (len(hammer) * dt)

            hammer_spec = abs(np.fft.fft(hammer))

            spec_length = int(MAX_FREQ / df)
            freqs = np.arange(0, spec_length * df, df)
            hammer_spec = hammer_spec[0:spec_length]

            inerts = []
            min_index = int(MIN_FREQ // df)

            for channel in test['acels']:
                spec = np.fft.fft(channel)[:spec_length]
                spec = abs(spec)
                inert = spec / hammer_spec
                inert = np.convolve(inert, np.ones((WINDOW_LENGTH,))/WINDOW_LENGTH, mode='same')
                inert = inert[min_index:]
                inert /= np.max(inert)
                inerts.append(inert.tolist())

            inerts = np.array(inerts)
            if all_inerts is None:
                all_inerts = inerts
            else:
                all_inerts = np.hstack((all_inerts, inerts))

            freqs = freqs[min_index:]
            freq_data += freqs.tolist()

        freq_data = [round(x, 0) for x in freq_data]
        lines, columns = np.shape(all_inerts)

        legend = ['Canal ' + str(i + 1) for i in range(lines)]
        f, ax = plt.subplots(figsize=(14, 7))

        for line in range(lines):
            sns.lineplot(freq_data, all_inerts[line, :])

        plt.xlim((10, 90))
        plt.ylim((-0.1, 1.1))
        plt.legend(legend, loc=1)
        plt.ylabel('Intertância normalizada')
        plt.xlabel('Frequência [Hz]')
        plt.savefig(test_name + '.png')
    except:
        pass

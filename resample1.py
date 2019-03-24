#coding:utf-8

# re-sample wav to 16Khz sampling 

import os
import argparse
import librosa
import numpy as np
from scipy.io.wavfile import write as wavwrite

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  librosa 0.6.0
#  scipy 1.0.0

def resample(path_in, path_out=None, sampling_rate=16000):
    # load and resampling
    y, sr = librosa.load(path_in, sr=sampling_rate)
    # save  add '-16k' as suffix if path_out is None
    if path_out is None:
        root, ext = os.path.splitext(path_in)
        path_out= root + '-16k' + ext
    
    wavwrite( path_out, sr , ( y * 2 ** 15).astype(np.int16))
    print ('save wav file', path_out)

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='resample wav to 16khz')
    parser.add_argument('--wav_file', '-w', default='a_1.wav', help='python3 resample1.py -w wav-file-name(mono,16bit)')
    args = parser.parse_args()
    
    resample(args.wav_file)

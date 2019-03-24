#coding:utf-8

# A trial transform from peak and drop frequency to tube length and reflection coefficient of two tube or three tube
# by grid search and scipy's optimize.fmin, downhill simplex algorithm.

import sys
import os
import argparse
import numpy as np
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from get_fp5 import *
from tube_peak1 import *
from pre_compute1 import *


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.0.0


def show_figure1(tube, spec0, fout_index, fout_index2, df0, fmin0, LA0, path0=None):
    # comparison frequency response of tube with wav
    index_f_min= int( np.ceil(tube.f_min / df0) )
    index_f_max= int( np.trunc(tube.f_max / df0) + 1) 
    f0= np.arange( index_f_min * df0, index_f_max * df0, df0 )
    resp0= spec0[index_f_min : index_f_max]
    
    NUM_TUBE= tube.NUM_TUBE
    fout_index_i = np.array(fout_index[0:tube.NUM_TUBE], dtype=np.int) - index_f_min
    fout2_index_i= np.array(fout_index2[0:tube.NUM_TUBE], dtype=np.int) - index_f_min
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.title('frequency response: blue tube, green wav: min cost ' + str( round(fmin0,1)) )
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    # tube spectrum
    ax1.semilogy(tube.f, tube.response, 'b', ms=2)
    ax1.semilogy(tube.f[tube.peaks_list] , tube.response[tube.peaks_list], 'ro', ms=3)
    ax1.semilogy(tube.f[tube.drop_peaks_list] , tube.response[tube.drop_peaks_list], 'co', ms=3)
    # wav lpc spectrum
    ax1.semilogy( f0 , resp0, 'g', ms=1)
    ax1.semilogy(f0[fout_index_i] , resp0[fout_index_i], 'ro', ms=3)
    ax1.semilogy(f0[fout2_index_i] , resp0[fout2_index_i], 'co', ms=3)
    
    plt.grid()
    plt.tight_layout()
    
    
    ax2 = fig.add_subplot(212)
    if len(LA0) == 4 or len(LA0) == 3:  # L1,L2,r1 X=[L1,L2,A1,A2] or X=[L1,L2,r1] two tube model
        L1= LA0[0]
        L2= LA0[1]
        
        if len(LA0) == 4:
            A1= LA0[2]
            A2= LA0[3]
        else:
            A1, A2 = get_A1A2( LA0[2] )
        
        ax2.add_patch( patches.Rectangle((0, -0.5* A1), L1, A1, hatch='/', fill=False))
        ax2.add_patch( patches.Rectangle((L1, -0.5* A2), L2, A2, hatch='/', fill=False))
        ax2.set_xlim([0, 30])
        ax2.set_ylim([-20, 20])
    
    elif len(LA0) == 6 or len(LA0) == 5:  # X=[L1,L2,L3,A1,A2,A3] or X=[L1,L2,L3,r1,r2]   when three tube model
        L1= LA0[0]
        L2= LA0[1]
        L3= LA0[2]
        
        if len(LA0) == 6:
            A1= LA0[3]
            A2= LA0[4]
            A3= LA0[5]
        else:
            A1, A2, A3 = get_A1A2A3( LA0[3], LA0[4] )
        
        ax2.add_patch( patches.Rectangle((0, -0.5* A1), L1, A1, hatch='/', fill=False))
        ax2.add_patch( patches.Rectangle((L1, -0.5* A2), L2, A2, hatch='/', fill=False))
        ax2.add_patch( patches.Rectangle((L1+L2, -0.5* A3), L3, A3, hatch='/', fill=False))
        ax2.set_xlim([0, 30])
        ax2.set_ylim([-20, 20])

    ax2.set_title('cross-section area')
    plt.xlabel('Length [cm]')
    plt.ylabel('Cross-section area [ratio]')
    plt.grid()
    plt.tight_layout()
    
    if path0 is not None:
        
        plt.savefig(path0)
    else:
        plt.show()
    

def get_path_name( dir0, path0, number0):
	# return file path name
	
    # make dir if the directory dir0 is not exist
    if not os.path.isdir( dir0 ):
        os.mkdir( dir0 )
    # get path0 basename without ext
    f, _= os.path.splitext( os.path.basename(path0))
    
    return dir0 + '/' + f + '_' + str(number0) + '.png'


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='estimation two tube model or three tube model ')
    parser.add_argument('--wav_file', '-w', default='wav/a_1-16k.wav', help='specify input wav-file-name(mono,16bit,16Khz)')
    parser.add_argument('--result_dir', '-r', default='result_figure', help='specify result directory')
    parser.add_argument('--frame', '-f', type=int, default=-1, help='specify the frame number, igonred if negative')
    parser.add_argument('--tube',  '-t', type=int, default=2, help='specify number of tube, 2 or 3')
    args = parser.parse_args()
    
    
    if args.tube == 2:  # try two tube model
        NUM_TUBE=2
        sampling_rate=16000
        
    else:  # try three tube model
        NUM_TUBE=3
        sampling_rate=16000
        
    
    # instance
    tube= compute_tube_peak(NUM_TUBE=NUM_TUBE, sampling_rate=sampling_rate)  #, disp=True)
    
    # load pre-computed grid data
    path0= 'pks_dpks_stack_tube' + str(NUM_TUBE) + '.npz'
    pc1= pre_comute(tube, path0=path0)
    
    # instance
    fp5= Class_get_fp(f_min= tube.f_min)
    
    # load wav and get formant candidates
    peak_list0, drop_peak_list0, spec_out, fout_index, fout_index2, pout= fp5.get_fp( args.wav_file, frame_num=args.frame)
    if sampling_rate != fp5.sr :
        print ('error: sampling rate is mismatch ', sampling_rate, fr5.sr)
        sys.exit()
    #   check if f_min is small than estimated pout. however, pout might be incorrect.
    if len( fp5.pout_f_min_check) > 0:
        print ('warning: pout_f_min_check ', fp5.pout_f_min_check)
    print ('number of frames ', len( peak_list0 ))
    
    
    if args.frame < 0:
        frame_list= range(len( peak_list0 ))
    else:
        frame_list= [args.frame ]
    for l, nfame in enumerate( frame_list):
        # set expect target value
        peak_list= peak_list0[l]
        drop_peak_list= drop_peak_list0[l]
        peaks_target=np.array( peak_list[0:NUM_TUBE])
        drop_peaks_target=np.array( drop_peak_list[0:NUM_TUBE])
        
        # get minimun cost at grid
        X = pc1.get_min_cost_candidate(peaks_target,drop_peaks_target, symmetry=True, disp=False)
        
        # try to minimize the function
        #   by "fmin" that is minimize the function using the downhill simplex algorithm.
        args1=(peaks_target,drop_peaks_target, -1)
        res_brute = optimize.fmin( tube.calc_cost, X, args=args1, full_output=True, disp=False)
        
        print ( 'frame %d min cost %f LA ' % (frame_list[l], res_brute[1]) , res_brute[0] ) 
        #print ( 'minimum ', res_brute[0] )  # minimum
        #print ( 'function value ', res_brute[1] )  # function value at minimum
        if res_brute[4] != 0:  # warnflag
            print ('warnflag is not 0')
        
        tube(res_brute[0]) 
        path0=get_path_name( args.result_dir, args.wav_file, frame_list[l])
        show_figure1(tube, spec_out[l], fout_index[l], fout_index2[l], fp5.df0, res_brute[1], res_brute[0], path0=path0)

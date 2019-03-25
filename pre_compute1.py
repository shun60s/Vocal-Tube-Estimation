#coding:utf-8

# precompute value at grid of LA_ranges and save as a npz file, Or load precomputed data from a npz file
#

import sys
import argparse
import itertools
import numpy as np


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 


class pre_comute(object):
    def __init__(self, tube, LA_ranges=None, MAX_tube_length=30., path0=None, display_count=100):
        #
        self.tube= tube
        self.NUM_TUBE= tube.NUM_TUBE
        self.display_count= display_count
        
        # compute when LA_ranges is specified
        if LA_ranges is not None:
            self.LA_ranges= LA_ranges
            self.MAX_tube_length= MAX_tube_length  # specify maximum whole tube length
            self.compute()
            # save compute data to path0
            if path0 is not None:  # tube_stack.npz
                np.savez(path0, pks=self.peaks_detail_stack, dpks=self.drop_peaks_detail_stack,\
                 iter=self.iters_stack, misc=[self.MAX_tube_length] )
                print ('pre compute data was saved to ', path0)
        
        # load pre-computed data from path0
        elif path0 is not None:
            try:
                x=np.load(path0)
            except:
                print ('error: cannot load ', path0)
                sys.exit()
            else:
                self.peaks_detail_stack=x['pks']
                self.drop_peaks_detail_stack=x['dpks']
                self.iters_stack=x['iter']
                self.MAX_tube_length=x['misc'][0]
                # check
                self.iters_len = self.peaks_detail_stack.shape[0]
                if self.peaks_detail_stack.shape[1] != self.NUM_TUBE:
                    print ('error: NUM_TUBE is mismatch.')
                    sys.exit()
                else:
                    print ('pre compute data was loaded from ', path0, self.iters_len)
        else:
            print ('error: no specified inputs.')
            sys.exit()
            
    def compute(self,):
        iters=[]
        for slice0 in self.LA_ranges:
            # add slice stop value in iteration
            iters.append( np.append( np.arange( slice0.start, slice0.stop, slice0.step), slice0.stop) )
        
        self.iters_len= len(list(itertools.product(*iters)))
        print ('iteraion number ', self.iters_len)
        
        self.peaks_detail_stack= np.zeros((self.iters_len, self.NUM_TUBE))
        self.drop_peaks_detail_stack= np.zeros((self.iters_len, self.NUM_TUBE))
        self.iters_stack= np.zeros( (self.iters_len, len(self.LA_ranges)) )
        
        c0=0
        for i, param0 in enumerate( itertools.product(*iters)):
            # skip compute if whole tube length is over than max_tube_length
            
            if len(param0) <= 4: # two tube
                if np.array(param0[0:2]).sum() > self.MAX_tube_length:
                    continue
            elif len(param0) <= 6: # three tube
                if np.array(param0[0:3]).sum() > self.MAX_tube_length: 
                    continue
            
            peaks_detail, drop_peaks_detail= tube( param0)
            self.peaks_detail_stack[c0]= peaks_detail
            self.drop_peaks_detail_stack[c0]= drop_peaks_detail
            self.iters_stack[c0]= np.array( param0)
            c0 +=1
            
            if i % self.display_count == 0 or i == (self.iters_len-1):
                sys.stdout.write("\r%d" % i)
                sys.stdout.flush()
        
        if c0 != self.iters_len:
            print ('final count number ', c0)
            self.peaks_detail_stack= self.peaks_detail_stack[0:c0]
            self.drop_peaks_detail_stack= self.drop_peaks_detail_stack[0:c0]
            self.iters_stack= self.iters_stack[0:c0]
            self.iters_len=c0
        
    
    def get_min_cost_candidate(self, peaks_target, drop_peaks_target, NRANK=10, symmetry=False, disp=False):
        #
        cost_list=np.zeros(self.iters_len)
        for i in range( self.iters_len ):
            cost_list[i]= self.tube.cost_0(self.peaks_detail_stack[i], self.drop_peaks_detail_stack[i],\
             peaks_target, drop_peaks_target)
        
        # sort, [0] is minimum candidate
        self.rank_index= np.argsort( cost_list )
        self.rank_value= np.sort( cost_list )
        
        if disp:
            # show Top NRANK value...
            for i in range (NRANK):
                print (self.rank_index[i], self.rank_value[i], self.iters_stack[ self.rank_index[i]])
        
        # select based on shape symmetry
        select_index=0
        if symmetry:
            if self.iters_stack.shape[1] == 3:  # two tube
                cost00= self.rank_value[0]
                for i in range (3): # search top 3
                    cost0i= self.rank_value[i]
                    L1= self.iters_stack[self.rank_index[i]][0]
                    L2= self.iters_stack[self.rank_index[i]][1]
                    if L1 >= L2 and abs( cost0i - cost00) < 1.0 : # get  L1 >= L2 and cost0 difference < 1.0( 1.0 is tentative value)
                        select_index=i
                        break
                #print (' select_index', select_index)
            elif self.iters_stack.shape[1] == 5:  # three tube
                # not implemented yet
                pass
        
        return self.iters_stack[ self.rank_index[select_index]].copy()


if __name__ == '__main__':
    #
    from tube_peak1 import *
    
    parser = argparse.ArgumentParser(description='make precomputed data to estimate tube model')
    parser.add_argument('--tube',  '-t', type=int, default=2, help='specify number of tube, 2 or 3')
    args = parser.parse_args()
    
    
    if args.tube == 2:  # try two tube model
        NUM_TUBE=2
        sampling_rate=16000
        # set initial value example
        X_init=[ 2.0, 5.0, 1.0, 9.0]  # X=[L1,L2,A1,A2] or X=[L1,L2,r1] 
        
        r1=get_r1(X_init[2:])
        A2=get_A2(r1, X_init[2])
        X_init2=[ 2.0, 5.0, r1]
        #print ('r1,A2', r1, A2)
        
        LA_ranges = ( slice(0.5,13,0.5), slice(0.5,13,0.5), slice(-0.9, 0.9, 0.1) )  # specify X value range
        MAX_tube_length= 26.   # specify maximum whole tube length
        
    else:  # try three tube model
        NUM_TUBE=3   
        sampling_rate=16000
        # set initial value example
        X_init=[ 2.0, 3.0, 5.0, 1.0, 3.0, 9.0]  # X=[L1,L2,L3,A1,A2,A3] or X=[L1,L2,L3,r1,r2]
        
        r1=get_r1(X_init[3:5])
        A2=get_A2(r1, X_init[3])
        r2=get_r1(X_init[4:])
        A3=get_A2(r2, X_init[4])
        X_init2=[ 2.0, 3.0, 5.0, r1,r2]
        #print ('r1,r2,A2,A3', r1,r2, A2, A3)
        
        # specify X value range
        # (1) This is rough grid for test
        #LA_ranges=(slice(0.5,13,2.0),slice(0.5,13,2.0),slice(0.5,13,2.0),slice(-0.9, 0.9, 0.3),slice(-0.9, 0.9, 0.3))
        # (2) This is rough grid for test with the special condition A2 >A1 (r1>=0) for fit /o/
        #LA_ranges=(slice(0.5,13,2.0),slice(0.5,13,2.0),slice(0.5,13,2.0),slice(0.0, 0.9, 0.3),slice(-0.9, 0.9, 0.3))
        # (3) This is small grid
        #LA_ranges=(slice(0.5,13,1.0),slice(0.5,13,1.0),slice(0.5,13,1.0),slice(-0.9, 0.9, 0.1),slice(-0.9, 0.9, 0.1))
        # (4) This is small grid with the special condition A2 >A1 (r1>=0) for fit /o/
        LA_ranges=(slice(0.5,13,1.0),slice(0.5,13,1.0),slice(0.5,13,1.0),slice(0.0, 0.9, 0.1),slice(-0.9, 0.9, 0.1)) 
        
        MAX_tube_length= 26.  # specify maximum whole tube length
        
        
    # instance
    tube= compute_tube_peak(NUM_TUBE=NUM_TUBE, sampling_rate=sampling_rate)  #, disp=True)
    
    path0= 'pks_dpks_stack_tube' + str(tube.NUM_TUBE) + '.npz'
    # new pre-compute
    pc0= pre_comute(tube, LA_ranges, MAX_tube_length, path0=path0)
    
    # test to load pre-computed gird data
    pc1= pre_comute(tube, path0=path0)
    

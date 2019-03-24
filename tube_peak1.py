#coding:utf-8

# compute peak and drop-peak frequency detail of the tube
# by scipy.optimize.minimize_scalar
#

import sys
import argparse
import numpy as np
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.0.0


class compute_tube_peak(object):
    def __init__(self, rg0=0.95, rl0=0.9 ,NUM_TUBE=2, sampling_rate=16000, disp=False):
        self.rg0=rg0
        self.rl0=0.9
        self.C0=35000.0  # speed of sound in air, round 35000 cm/second
        self.NUM_TUBE=NUM_TUBE
        self.sampling_rate=sampling_rate
        self.Delta_Freq=5
        self.f_min=200
        self.f_max=5000
        self.f_out=100000 # 候補がないときに代入する値
        self.f=np.arange(self.f_min, self.f_max, self.Delta_Freq)
        self.xw= 2.0 * np.pi * self.f
        self.sign0=1.0 # normal
        self.disp=disp
        self.counter=0

    def __call__(self, X):
        # X[0,1]= L1,L2
        # X[2,3]= A1,A2
        # they should be same as get_ft5
        
        if (len(X) == 6) or (len(X) == 5) : # X=[L1,L2,L3,A1,A2,A3] or X=[L1,L2,L3,r1,r2]   when three tube model
            tu1= X[0] / self.C0   # delay time in 1st tube
            tu2= X[1] / self.C0   # delay time in 2nd tube
            tu3= X[2] / self.C0   # delay time in 2nd tube
            if len(X) == 6:
                r1=(  X[4] -  X[3]) / (  X[4] +  X[3])  # reflection coefficient between 1st tube and 2nd tube
                r2=(  X[5] -  X[4]) / (  X[5] +  X[4])  # reflection coefficient between 2nd tube and 3rd tube
            else:
                r1=X[3]
                r2=X[4]
            
            func1= self.func_yb_t3
            args1=(tu1,tu2,tu3,r1,r2)
            
            # abs(yi) = abs( const * (cos wv + j sin wv)) becomes constant. So, max/min(abs(val)) depends on only yb
            self.yi= 0.5 * ( 1.0 +  self.rg0 ) * ( 1.0 +  r1)  * ( 1.0 +  r2)  * ( 1.0 +  self.rl0 ) * \
            np.exp( -1.0j * (  tu1 +  tu2 +  tu3 ) * self.xw) 
            # yb
            yb1= 1.0 +  r1 *  self.rg0 *  np.exp( -2.0j *  tu1 * self.xw ) 
            yb1= yb1 +  r2  *  r1 *  np.exp( -2.0j *  tu2 * self.xw ) 
            yb1= yb1 +  self.rl0 *  r2 *  np.exp( -2.0j *  tu3 *  self.xw ) 
            yb2=        r2  *  self.rg0 *  np.exp( -2.0j * ( tu1 +  tu2) * self.xw ) 
            yb2= yb2 +  self.rl0 *  r1  *  np.exp( -2.0j * ( tu2 +  tu3) *  self.xw ) 
            yb3=  self.rl0 *  r2 *  r1 *  self.rg0 *  np.exp( -2.0j * ( tu1 +  tu3) *  self.xw )
            yb4=  self.rl0 *  self.rg0 * np.exp( -2.0j * ( tu1 +  tu2 +  tu3) *  self.xw ) 
            self.yb= yb1 + yb2 + yb3 + yb4
            
        elif (len(X) == 4) or (len(X) == 3): # else X=[L1,L2,A1,A2] or X=[L1,L2,r1] two tube model
            tu1= X[0] / self.C0   # delay time in 1st tube
            tu2= X[1] / self.C0   # delay time in 2nd tube
            if len(X) == 4:
                r1=(  X[3] -  X[2]) / (  X[3] +  X[2])  # reflection coefficient between 1st tube and 2nd tube
            else:
                r1= X[2]
            
            func1= self.func_yb_t2
            args1=(tu1,tu2,r1)
            
            # compute frequency response
            # abs(yi) = abs( const * (cos wv + j sin wv)) becomes constant. So, max/min(abs(val)) depends on only yb
            self.yi= 0.5 * ( 1.0 +  self.rg0 ) * ( 1.0 +  r1)  * ( 1.0 +  self.rl0 ) * \
            np.exp( -1.0j * (  tu1 +  tu2 ) * self.xw) 
            # yb
            self.yb= 1.0 +  r1 *  self.rg0 *  np.exp( -2.0j *  tu1 * self.xw ) +  \
            self.rl0 *  r1 *  np.exp( -2.0j *  tu2 * self.xw ) + \
            self.rl0 *  self.rg0 * np.exp( -2.0j * ( tu1 +  tu2) * self.xw ) 
        else:
            print ('error: len(X) is not expected value.', len(X))
        
        val= self.yi / self.yb
        self.response=np.sqrt(val.real ** 2 + val.imag ** 2)
        
        # get peak and drop-peak list
        self.peaks_list=signal.argrelmax(self.response)[0] # signal.argrelmax output is triple
        peaks= self.f[ self.peaks_list ]
        self.drop_peaks_list=signal.argrelmin(self.response)[0]
        drop_peaks= self.f[ self.drop_peaks_list ]
        
        # 候補点がNUM_TUBEより少ないときは f_outを入れておく
        if len(peaks) < self.NUM_TUBE:
            peaks= np.concatenate( ( peaks, np.ones( self.NUM_TUBE - len(peaks)) * self.f_out ) )
        elif len(peaks) > self.NUM_TUBE:
            peaks=peaks[0: self.NUM_TUBE]
            self.peaks_list=self.peaks_list[0: self.NUM_TUBE]
        if len(drop_peaks) < self.NUM_TUBE:
            drop_peaks= np.concatenate( ( drop_peaks, np.ones( self.NUM_TUBE - len(drop_peaks)) * self.f_out ) )
        elif len(drop_peaks) > self.NUM_TUBE:
            drop_peaks=drop_peaks[0: self.NUM_TUBE]
            self.drop_peaks_list=self.drop_peaks_list[0: self.NUM_TUBE]
        
        # より詳細に探索する
        peaks_detail=np.zeros( len(peaks) )
        drop_peaks_detail=np.zeros( len(drop_peaks) )
        
        ## peak
        self.sign0=1.0 # normal
        for l, xinit in enumerate( peaks ):
            if xinit >= self.f_max:
                peaks_detail[l]= xinit
            else:
                # Use brent method: 囲い込み戦略と二次近似を組み合わせ
                b_xinit=[ xinit - self.Delta_Freq ,  xinit + self.Delta_Freq  ]
                res = optimize.minimize_scalar(func1, bracket=b_xinit, args=args1)
                peaks_detail[l]= res.x
                if self.disp:
                    print ('b_xinit', b_xinit)
                    print ('result x', res.x)
        
        ## drop-peak
        self.sign0=-1.0 # turn upside down
        for l, xinit in enumerate( drop_peaks ):
            if xinit >= self.f_max:
                drop_peaks_detail[l]= xinit
            else:
                b_xinit=[ xinit - self.Delta_Freq ,  xinit + self.Delta_Freq  ]
                res = optimize.minimize_scalar(func1, bracket=b_xinit, args=args1)
                drop_peaks_detail[l]= res.x
                if self.disp:
                    print ('b_xinit', b_xinit)
                    print ('result x', res.x)
        
        return peaks_detail, drop_peaks_detail


    def func_yb_t2(self, x, *args):  # two tube  *は可変長の引数
        x = x
        tu1,tu2,r1= args
        xw= x * 2.0 * np.pi
        yb= 1.0 +  r1 *  self.rg0 *  np.exp( -2.0j *  tu1 * xw ) +  self.rl0 *  r1 *  np.exp( -2.0j *  tu2 * xw ) + \
        self.rl0 *  self.rg0 * np.exp( -2.0j * ( tu1 +  tu2) * xw ) 
        return (yb.real**2 + yb.imag**2)  * self.sign0

    def func_yb_t3(self, x, *args):  # three tube  *は可変長の引数
        tu1,tu2,tu3,r1,r2= args
        xw= x * 2.0 * np.pi
        yb1= 1.0 +  r1 *  self.rg0 *  np.exp( -2.0j *  tu1 * xw ) 
        yb1= yb1 +  r2  *  r1 *  np.exp( -2.0j *  tu2 * xw ) 
        yb1= yb1 +  self.rl0 *  r2 *  np.exp( -2.0j *  tu3 * xw ) 
        yb2=        r2  *  self.rg0 *  np.exp( -2.0j * ( tu1 +  tu2) * xw ) 
        yb2= yb2 +  self.rl0 *  r1  *  np.exp( -2.0j * ( tu2 +  tu3) * xw ) 
        yb3=  self.rl0 *  r2 *  r1 *  self.rg0 *  np.exp( -2.0j * ( tu1 +  tu3) * xw )
        yb4=  self.rl0 *  self.rg0 * np.exp( -2.0j * ( tu1 +  tu2 +  tu3) * xw ) 
        yb= yb1 + yb2 + yb3 + yb4
        return (yb.real**2 + yb.imag**2)  * self.sign0
        

    def show_freq(self,):
        # show rough(accuracy=Delta_Freq) result
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        plt.title('frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        
        if 1: # show peak and drop peak
            ax1.semilogy(self.f, self.response, 'b', ms=2)
            ax1.semilogy(self.f[self.peaks_list] , self.response[self.peaks_list], 'ro', ms=3)
            ax1.semilogy(self.f[self.drop_peaks_list] , self.response[self.drop_peaks_list], 'co', ms=3)
        
        if 1: # show yi and yb
            ax1.plot( self.f , np.abs(self.yi), 'g')
            ax1.plot( self.f , np.abs(self.yb), 'y')
        
        plt.grid()
        plt.axis('tight')
        plt.show()
        

    def reset_counter(self,):
        self.counter=0

    def cost_0(self, peaks2, drop_peaks2, peaks, drop_peaks):
        # lower cost function
        return (abs(peaks - peaks2).mean() + abs(drop_peaks - drop_peaks2).mean()) / 2.0

    def calc_cost(self, X , peaks, drop_peaks, display_count=100, disp=False):
        # get mean of difference between target and new computed ones
        peaks2, drop_peaks2= self.__call__(X)
        cost0= self.cost_0( peaks2, drop_peaks2, peaks, drop_peaks)
        
        # add penalty if reflection coefficient abs is over than 0.9
        if len(X) == 3 and abs( X[2]) > 0.9:
            cost0 += 1000.0
        elif len(X) == 5 and  ( abs( X[3]) > 0.9  or abs( X[4] > 0.9) ):
            cost0 += 1000.0
        
        if disp :
            print (X,cost0, peaks2, drop_peaks2)
        self.counter +=1
        # show present counter value,  don't show if display_count is negative
        if display_count > 0 and self.counter % display_count == 0:
            sys.stdout.write("\r%d" % self.counter)
            sys.stdout.flush()
        return cost0



# helper functions
def get_r1( X ):
    return (  X[1] -  X[0]) / (  X[1] +  X[0])  # return reflection coefficient between 1st tube and 2nd tube

def get_A2( r1, A1 ):
    if abs(r1) >= 1.0:
        print ('error: abs(r1) > 1.0')
    return (( 1.0 + r1) / ( 1 - r1)) * A1  # return cross-section area of 2nd tube
    
def get_A1( r1, A2 ):
    if abs(r1) >= 1.0:
        print ('error: abs(r1) > 1.0')
    return (( 1.0 - r1) / ( 1 + r1)) * A2  # return cross-section area of 1st tube

def get_A1A2( r1, A_min=1.0):
    # return cross-section area A1 and A2 under the condition of 
    # minimum cross-section is fixed as A_min
	if r1 >= 0.0:
	    return A_min,  get_A2(r1, A_min)
	else:
	    return get_A1(r1, A_min), A_min

def get_A1A2A3( r1, r2, A_min=1.0):
    # return cross-section area A1 A2 and A3 under the condition of 
    # minimum cross-section is fixed as A_min
    A1=1.0
    A2=get_A2( r1, A1 )
    A3=get_A2( r2, A2 )
    min_index= np.argmin( [A1,A2,A3] )
    
    if min_index == 0:
        A1= A_min
        A2= get_A2( r1, A1 )
        A3= get_A2( r2, A2 )
    elif min_index == 1:
        A2= A_min
        A1= get_A1( r1, A2)
        A3= get_A2( r2, A2)
    elif min_index == 2:
        A3= A_min
        A2= get_A1( r2, A3)
        A1= get_A1( r1, A2)
    
    return A1, A2, A3


if __name__ == '__main__':
    
    
    # shape symmetry check
    # 全長が同じ長さの２管モデルは2種類存在するが、その多くは　ピークとドロップピークの位置がかなり近くなる対称性がある。
    # 大きく差がでるケースもある。完全な対称性はない。
    # X[ L1, L2, r1] and X2[ L2, L1, r1] are same total lenght LT=L1+L2 (tu1+tu2).
    
    # instance
    tube = compute_tube_peak(rg0=0.95, rl0=0.9)
    
    #LA_ranges = ( slice(0.5,13,0.5), slice(0.5,13,0.5), slice(-0.9, 0.9, 0.1) )  # specify X value range
    total_lenght=26.0
    L1= np.arange( 0.5,total_lenght,0.5)
    L2= total_lenght - L1
    r1= -0.8
    for l1 in L1:
        l2= total_lenght - l1
        X_init= [ l1, l2, r1]
        X_init2=[ l2, l1, r1]
        
        peaks, drop_peaks= tube(X_init)
        peaks2, drop_peaks2= tube(X_init2)
        cost0= tube.cost_0( peaks2, drop_peaks2, peaks, drop_peaks)
        
        print ( cost0)

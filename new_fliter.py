import scipy.signal as signal
import numpy as np
import pywt
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt

class Filter():
    def band_pass(self,data,order, f0, ft,fs):
        n,m = data.shape
        b, a = signal.butter(order, [f0*2/fs, ft*2/fs], 'bandpass')
        #wn1=2*f0/1000,wn2=2*ft/1000=0.8.f1,f2分别为带通范围
        for i in range(m):
            filtedData = signal.filtfilt(b, a, (data[:, i]))
            if i == 0:
                alldata = filtedData
            else:
                alldata = np.vstack((alldata, filtedData))
        alldata = alldata.T
        return alldata

    def low_pass(self, data, order, f0,fs):
        n, m = data.shape
        b, a = signal.butter(order, f0*2/fs, 'lowpass')
        for i in range(m):
            filtedData = signal.filtfilt(b, a, (data[:, i]))
            if i == 0:
                alldata = filtedData
            else:
                alldata = np.vstack((alldata, filtedData))
        alldata = alldata.T
        return alldata

    def notch_filter(self,data,f0,Q,fs):
        n, m = data.shape
        b, a = signal.iirnotch(f0*2/fs, Q)
        for i in range(m):
            filtedData = signal.lfilter(b, a, (data[:, i]))
            if i == 0:
                alldata = filtedData
            else:
                alldata = np.vstack((alldata, filtedData))
        alldata = alldata.T
        return alldata

    def smooth(self,t,data):
        n, m = data.shape
        weight = np.ones(t)
        weight /= weight.sum()
        # weight = np.linspace(1, 0, n)
        for i in range(m):
            smoothdata = np.convolve(data[:, i], weight, mode='valid')
            if i == 0:
                alldata = smoothdata
            else:
                alldata = np.vstack((alldata, smoothdata))
        alldata = alldata.T
        return alldata

    def zero_cross(self, data, t=100):
        n, m = data.shape
        array = np.zeros((n, m))
        zc = []
        for i in range(m):
            for j in range(1, n, 1):
                if data[j, i]*data[j-1, i] < 0:
                    array[j, i] = 1
        for i in range(0,n,t):
            b = i + t
            a = i
            if b > n:
                return np.array(zc)
            else:
                seg = array[a:b,:]
                zc.append(np.sum(seg,axis=0))
        return np.array(zc)

    def standard_deviation_cross(self,data,t =100):
        n, m = data.shape
        array = np.zeros((n, m))
        sd = []
        for i in range(1,n,t):
            seg = data[i:i+t,:]
            n1, _ = seg.shape
            if n1 < t:
                break
            std_seg = np.std(seg,axis=0)
            seg2 = seg-np.tile(std_seg, [t, 1])
            if i == 1:
                data_a = seg2
            else:
                data_a = np.vstack((data_a, seg2))
        n1,m1 =data_a.shape
        for i in range(m1):
            for j in range(1, n1, 1):
                if data_a[j, i]*data_a[j-1, i] < 0:
                    array[j, i] = 1
        for i in range(0,n1,t):
            b = i + t
            a = i
            if b > n1:
                return np.array(sd)
            else:
                seg = array[a:b,:]
                sd.append(np.sum(seg,axis=0))
        return np.array(sd)

    def wavelets(self,data,wave='sym4',t = 300):
        label = []
        n, m = data.shape
        # print(n,m)
        le = 5
        wavelet = pywt.Wavelet(wave)
        alist = [[] for i in range(le)]
        wl_all = [[] for i in range(int((n-t)/t)+1)]
        for j in range(0, n, t):
            b = j + t-1
            a = j
            if b >= n:
                return np.array(wl_all),label
            label.append([a,b])
            for i in range(m):
                dataseg = data[a:b,i]
                _,alist[0],alist[1],alist[2],alist[3],alist[4] = pywt.wavedec(dataseg, wavelet, mode='symmetric', level=le)
                for e in range(5):
                    wl_all[j//t].append(max(alist[e]))


    def mean_std_max(self,data,t=100):
        n, m = data.shape
        for i in range(1,n,t):
            seg = data[i:i+t,:]
            n1, _ = seg.shape
            if n1 < t:
                break
            std_seg = np.std(abs(seg),axis=0)
            mean_seg = np.mean(abs(seg),axis=0)
            max_seg = np.max(abs(seg),axis=0)
            if i == 1:
                data_std = std_seg
                data_mean = mean_seg
                data_max = max_seg
            else:
                data_std = np.vstack((data_std, std_seg))
                data_mean= np.vstack((data_mean, mean_seg))
                data_max = np.vstack((data_max, max_seg))
        return data_mean,data_std,data_max
    def psd(self,data,t = 100):
        n, m = data.shape
        for i in range(1, n, t):
            seg = data[i:i + t, :]
            n1, _ = seg.shape
            if n1 < t:
                break
            psd_seg = []
            for j in range(m):
                y1 =abs(fft(seg[:,j]))
                ps1 = y1 ** 2 / t
                psd_seg.append(sum(ps1[1:t//2]**2)/t)
            psd_seg = np.array(psd_seg)

            if i == 1:
                psd = psd_seg
            else:
                psd = np.vstack((psd, psd_seg))
        return psd
def getdata(rawdata,t1,fs = 1500):
    data_filter = Filter()
    data_bp = data_filter.band_pass(data=rawdata, order=4, f0=10, ft=450, fs=fs)
    data_nf = data_filter.notch_filter(data=data_bp, f0=50, fs=fs, Q=30)
    data_zc = data_filter.zero_cross(data_nf, t=t1)
    data_wl,_ = data_filter.wavelets(data_nf, wave='sym4', t=t1)
    # data_sd = data_filter.standard_deviation_cross(data_nf, t=t1)
    # data_asd = (data_zc + data_sd) / 2
    data_mean,_,_ = data_filter.mean_std_max(data=data_nf,t=t1)
    psd = data_filter.psd(data_nf, t=t1)
    n,m = data_zc.shape
    all_data = []
    if n==1:
        data_t=np.append(data_wl[0],data_zc[0])
        data_t = np.append(data_t,data_mean)
        data_t = np.append(data_t,psd)
        return  data_t
    else:
        for i in range(n):
            data_t = np.append(data_wl[i],data_zc[i])
            data_t = np.append(data_t, data_mean[i])
            data_t = np.append(data_t,psd[i])
            all_data.append(data_t)
    return np.array(all_data)

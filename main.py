import numpy as np
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import pyaudio
import scipy.fftpack as scifft

def sound(audio, fs):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(audio.dtype.itemsize),
                    channels=1,
                    rate=fs,
                    output=True)
    audioStream = audio.tostring()
    stream.write(audioStream)
    stream.close()

#read the file
fs, x = scipy.io.wavfile.read("speech.wav")

speech = x[:, 0]


N = 4

b = np.array([0.3235,0.2665, 0.2940,0.2655, 0.3235])

h0 = b[::4]
h1 = b[1::4]
h2 = b[2::4]
h3 = b[3::4]

y0 = signal.lfilter(h0, [1], speech)
y1 = signal.lfilter(h1, [1], speech)
y2 = signal.lfilter(h2, [1], speech)
y3 = signal.lfilter(h3, [1], speech)


#filter and upsample - Noble Identities
y_up = np.zeros(len(speech)*4,)

y_up[0::4] = y0             #upsample by 4
y_up[1::4] = y1
y_up[2::4] = y2
y_up[3::4] = y3

print ("Filtered Signal with FIR filter")
sound(y_up.astype(np.int16), fs*4)
    

#Design filter of length 32

n = 32 #filter length

F = [0, 0.1, 0.18 ,0.5] #normalized frequencies
A = [1.0, 0.0]                      
W = [1, 100]                        

taps = signal.remez(n, F, A, weight=W)
print(taps)
hh0 = taps[0::4]
hh1 = taps[1::4]
hh2 = taps[2::4]
hh3 = taps[3::4]

Y0 = signal.lfilter(hh0, [1], speech)
Y1 = signal.lfilter(hh1, [1], speech)
Y2 = signal.lfilter(hh2, [1], speech)
Y3 = signal.lfilter(hh3, [1], speech)  

#filter and upsample - Noble Identities
Y_up = np.zeros(len(speech)*4)

Y_up[0::4] = Y0
Y_up[1::4] = Y1
Y_up[2::4] = Y2
Y_up[3::4] = Y3
print ("Filtered signal with remez filter")
sound(Y_up.astype(np.int16), fs*4)


w, h = signal.freqz(taps, [1])
plt.figure(2)
plt.subplot(211)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Frequency response')
plt.subplot(212)
plt.plot(taps)
plt.title('Impulse response')   
plt.show()















"""w1, h1 = signal.freqz(y_up)
impulseSignal = np.zeros((50,))
impulseSignal[24] = 1.
FIR_Filter = signal.lfilter(b, 1, impulseSignal)

plt.figure(1)
plt.subplot(211)
plt.plot(w1, 20 * np.log10(abs(h1)))
plt.title('Frequency response of Upsampled version') 
plt.subplot(212)
plt.plot(FIR_Filter)
plt.title('Impulse response of FIR Filter')   
#plt.show()
"""




#w2, h2 = signal.freqz(Y_up)
"""plt.figure(3)
plt.plot(w2, 20 * np.log10(abs(h2)))
plt.title('Frequency response of Upsampled version(Remez)') 
#plt.show()
"""

#FFT of upsampled signal without filtering
"""
xf = np.linspace(0.0, 44000, len(Y_up))

plt.figure(4)
plt.subplot(211)
yf = scifft.fft(Y_up)
plt.plot(xf,yf)
plt.title('FFT of upsampled signal without filtering')
#FFT of upsampled signal with filtering
plt.subplot(212)
yf1 = scifft.fft(signal.lfilter(b, 1, Y_up))
plt.plot(xf,yf1)
plt.title('FFT of upsampled signal with filtering')
plt.show()
#print (yf - yf1)
"""


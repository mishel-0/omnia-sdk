"""Synthesize the promo score, cut-synced to ad.html. Usage: python3 score.py out.wav"""
import numpy as np, sys, wave
SR=48000; D=40.0; N=int(SR*D); t=np.arange(N)/SR
rng=np.random.default_rng(7)
L=np.zeros(N); Rr=np.zeros(N)
def add(sig,start,pan=0.0,g=1.0):
    i=int(start*SR); n=min(len(sig),N-i)
    if n<=0: return
    L[i:i+n]+=sig[:n]*g*np.sqrt(.5*(1-pan)); Rr[i:i+n]+=sig[:n]*g*np.sqrt(.5*(1+pan))
def env(n,a,r): x=np.arange(n)/SR; return np.minimum(1,x/max(a,1e-4))*np.exp(-x/r)
def lp(x,k):  # one-pole lowpass, k in (0,1)
    y=np.empty_like(x); s=0.; kk=np.broadcast_to(np.asarray(k,float),x.shape)
    for i in range(len(x)): s+=kk[i]*(x[i]-s); y[i]=s
    return y
def hz(n): return 440*2**((n-69)/12)
def pad(notes,dur,att=2.,rel=3.,bright=.3):
    n=int(dur*SR); x=np.arange(n)/SR; s=np.zeros(n)
    for m in notes:
        for d in (-.07,0,.06):  # detuned saws via additive partials
            f=hz(m)*2**(d/12)
            for k in range(1,8): s+=np.sin(2*np.pi*f*k*x+k*d*40)/k*bright**(k-1)
    e=np.minimum(1,x/att)*np.minimum(1,(dur-x)/rel); return s*e/len(notes)/6
def boom(dur=3.):
    n=int(dur*SR); x=np.arange(n)/SR; f=38+70*np.exp(-x*9)
    return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-x*1.6)*1.0
def noise_burst(dur,att,decay,k): n=int(dur*SR); return lp(rng.standard_normal(n),k)*env(n,att,decay)
def riser(dur):
    n=int(dur*SR); x=np.arange(n)/SR; p=x/dur
    s=lp(rng.standard_normal(n),.02+.25*p**2)*p**3*1.5
    s+=np.sin(2*np.pi*np.cumsum(200+1400*p**3)/SR)*p**4*.15; return s
def pluck(m,dur=1.2,b=.5):
    n=int(dur*SR); x=np.arange(n)/SR; f=hz(m)
    s=sum(np.sin(2*np.pi*f*k*x)*b**(k-1)/k*np.exp(-x*(3+k*2)) for k in range(1,6)); return s*np.minimum(1,x/.003)
def kick():
    n=int(.5*SR); x=np.arange(n)/SR; f=45+120*np.exp(-x*35); return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-x*7)
def tick():
    n=int(.06*SR); return rng.standard_normal(n)*np.exp(-np.arange(n)/SR*90)*.25

# 0-4.2  seam: shimmering swell
add(pad([81,88,93],4.6,att=3.2,rel=.8,bright=.15)*.5,0)
add(riser(4.0)*.35,.2)
# 4.2-12: sparse, uneven "waiting" plucks over a dark drone
add(pad([45,52],8.2,att=1.,rel=1.,bright=.2)*.9,4.2)
times=[4.6,5.5,5.9,7.1,7.5,8.6,9.4,9.7,10.6,11.1]; notes=[69,72,76,69,74,72,76,79,74,81]
for i,(s,m) in enumerate(zip(times,notes)): add(pluck(m,1.5)*.28,s,pan=(-.5 if i%2 else .5))
add(riser(2.0),10.2,g=.9)
# 12.2 impact + hero pad (A maj9)
add(boom(4)*1.1,12.2); add(noise_burst(2.5,.002,.35,.15)*.6,12.2)
add(pad([45,57,61,64,68,71],8.6,att=1.5,rel=1.2,bright=.35),12.2,g=1.1)
for i,s in enumerate(np.arange(15.6,20.4,.6)): add(pluck([76,80,83,85][i%4]+12,1.0,.3)*.12,s,pan=np.sin(i))
add(riser(1.2),19.3,g=.7)
# 20.5-29: drive, 124 bpm
beat=60/124; add(boom(2.5)*.8,20.5); add(noise_burst(1.6,.01,.4,.3)*.4,20.5)
add(pad([42,54,61,66,69],8.6,att=.6,rel=.8,bright=.45)*.9,20.5)
arp=[66,69,73,78,73,69,81,78]
for i,s in enumerate(np.arange(20.5,28.9,beat/2)):
    add(pluck(arp[i%8],.4,.6)*.16,s,pan=(-.4 if i%2 else .4))
    if i%2==0: add(kick()*.55,s)
    else: add(tick(),s,pan=.3)
for s in (22.6,25.6): add(boom(2.)*.7,s); add(noise_burst(1.,.005,.2,.4)*.25,s)
add(riser(1.0),28.0,g=.8)
# 29-34: compress
add(boom(3)*.9,29.0); add(pad([38,50,57,62,66],5.2,att=.4,rel=1.,bright=.25),29.0)
n=int(1.6*SR); x=np.arange(n)/SR; add(np.sin(2*np.pi*np.cumsum(400*np.exp(-x*2.2)+40)/SR)*np.minimum(1,x/.05)*np.exp(-x*1.2)*.25,30.0)
add(boom(2.5)*.9,31.6); add(noise_burst(.8,.001,.12,.5)*.35,31.6)
# 34-40: resolution
add(pad([45,52,57,61,64,71,76],6.2,att=1.2,rel=3.5,bright=.3)*1.1,34.0)
add(boom(4)*.7,35.6); add(pluck(88,3,.3)*.2,35.6); add(pluck(93,3,.3)*.12,36.4,pan=.4)

# reverb (FFT convolution with decaying stereo noise)
def verb(x,seed):
    n=int(2.8*SR); ir=np.random.default_rng(seed).standard_normal(n)*np.exp(-np.arange(n)/SR*2.2); ir=lp(ir,.25); ir[0]=0
    m=len(x)+n; F=2**int(np.ceil(np.log2(m))); y=np.fft.irfft(np.fft.rfft(x,F)*np.fft.rfft(ir,F),F)[:len(x)]
    return y/np.max(np.abs(ir))*.012
L2=L+verb(L,1)*.9; R2=Rr+verb(Rr,2)*.9
mix=np.stack([L2,R2],1); fade=np.minimum(1,(D-t)/2.0)[:,None]; mix*=fade
mix=np.tanh(mix/np.max(np.abs(mix))*1.6)*.89
with wave.open(sys.argv[1],'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype('<i2').tobytes())

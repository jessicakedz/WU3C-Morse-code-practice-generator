# Morse-code-practice-generator
This is a simple program to read a text file and generate morse code with noise and qsb. controll its characteristics with the parameters in the code: 

```
#============================================
# load message from this file
message_file='C:/data/message.txt' # <-set this to point at the message text file. only valid characters are used 
#=====================================
# code characteristics
wpm=20
frequency = 500  # cw freq Hz
keykClick=0.2 # 0 to 1  # sets the clickyness of the code. 0 is clicky ah heck 1 is mushy
# Parameters
SNR=10 # signal to noise in dB - this is the ratio of the noise that is not QSB to the cw tone. negative is allowed 
qSNR=1 #QSB depth dB - 0 is max QSB larger positive dB values reduce the QSB by that db Ratio
#=====================================
# channel filter - this is set to mimic your normally used  cw copy filter. use the high and low to mimic shift
lowcut = 200.0  # Low cutoff frequency in Hz
highcut = 1200.0 # High cutoff frequency in Hz

```

# future things will be:
- cut message up to generate wave in blocks vs one huge file
- maybe export as a mp4 for use anywhere
- copy check 
- signal drift, near by code, interference and lids etc
- real time control of the filter with arrows

Thanks to [OpenAI](https://www.openai.com) for their language model ChatGPT, which assisted in generating the QSO dataset.

10May25

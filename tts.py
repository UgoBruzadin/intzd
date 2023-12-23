import sys
import numpy as np # this goes into the vocaliser file
import torch  # sane
from scipy.io.wavfile import write #same


#import soundfile as sf # ogg codec crashes on long files

# set up tachotron and waveglow from torchub
def initializeTTSEngine():
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    # modified this to allow longer sentences, unsure if it did anything at all.
    tacotron2.max_decoder_steps = 3000
    tacotron2 = tacotron2.to('cuda')
    tacotron2.eval()

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()

    return tacotron2, waveglow

# just stuff i found here https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/
# This does the actual text to speech
def vocalise(subsentences, outputFile, outputFormat, speed, intermediaryFormat, tacotron2, waveglow):
    print("[+] Starting TTS on "+outputFile.split("/")[-1])

    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
            
    rate = 22050
    # load a saved state if any
    audio_numpy, currentSubsentenceIndex = loadVocaliseState(outputFile)
    try:# this is here so we can save before Ctrl+C
        # process each sentence as tacotron2 -> waveglow -> whatever
        from tqdm import tqdm
        from librosa.util import normalize
        for text in tqdm(subsentences[currentSubsentenceIndex:],
                            dynamic_ncols=True,
                            initial=currentSubsentenceIndex,
                            total=len(subsentences)):
            # preprocessing
            #text = join(text,'.')
            sequence, lengths = utils.prepare_input_sequence([text])
            #from tacotron2.text import text_to_sequence
            #print(text)
            #sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
            #sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
            #print(sequence)
            input_lengths = torch.tensor([sequence.size(0)], device='cuda')
            #print("Shape of input lengths:", input_lengths)
            
            # run the models
            with torch.no_grad():
                mel, alignment_matrix, gate_output = tacotron2.infer(sequence,lengths)
                
                # Normalize the audio data
                
                #print("shape of", mel)
                #print("shape of", b)
                #print("Shape of mel tensor:", mel)

                # Ensure mel tensor has the expected dimensions
                #mel = mel.unsqueeze(1)  # Add a channel dimension
                #plot_mel_alignment_gate(mel.cpu(), alignment_matrix.cpu(), gate_output.cpu())
                audio = waveglow.infer(mel)
            audio_numpy = np.concatenate((audio_numpy, audio[0].data.cpu().numpy()))
            #print("Min value of audio_numpy:", np.min(audio_numpy))
            #print("Max value of audio_numpy:", np.max(audio_numpy))

            # save current position in case TTS is interrupted
            currentSubsentenceIndex = currentSubsentenceIndex + 1

        # save resulting wav file
        write(outputFile+intermediaryFormat, rate, audio_numpy)
        # these are just alternative ways to write files, none seemed too good
        # sf.write(outputFile+".wav",audio_numpy, rate, format="ogg")
        # writeAudio(outputFile+".wav",audio_numpy, rate)
        convertFormat(outputFile+intermediaryFormat, outputFormat, speed)
        cleanSaves(outputFile,intermediaryFormat)
    except KeyboardInterrupt:
        saveVocaliseState(outputFile,intermediaryFormat,audio_numpy,currentSubsentenceIndex,rate)
        sys.exit()


# Saving and loading progress to disk

import io
import json, zlib, base64
import os
def saveVocaliseState(outputFile, intermediaryFormat, audio_numpy, currentSubsentenceIndex,rate):
    # saving binary array compressed and b64 json
    memfile = io.BytesIO()
    np.save(memfile, audio_numpy)
    memfile.seek(0)
    saveDict = {
        "subsentenceIndex" : currentSubsentenceIndex,
        "binaryAudio_numpy" : base64.b64encode(zlib.compress(memfile.read())).decode('ascii')
    }

    with open(outputFile+'.sav', 'w') as f:
        json.dump(saveDict, f)

    # this means there was a partial TTS, so we also save where we left off
    print("\n[!] TTS interrupted, progress saved to "+outputFile+".sav")
    print("[?] To continue from here, set the --output to the folder containing the .sav file")
    return

def loadVocaliseState(loadPath):
    loadPath=loadPath+".sav"
    if os.path.exists(loadPath):
        with open(loadPath, 'r') as f:
            saveDict = json.load(f)
            memfile = io.BytesIO()
            memfile.write(zlib.decompress(base64.b64decode(saveDict["binaryAudio_numpy"])))
            memfile.seek(0)
            audio_numpy = np.load(memfile)
            print("[+] Found saved state at "+loadPath+" continuing from subsentence " + str(saveDict["subsentenceIndex"]))
            return audio_numpy, saveDict["subsentenceIndex"]
    else:
        print("[+] Saved state not found at "+loadPath+" starting from the beginning")
        audio_numpy = np.ndarray(1)
        currentSubsentenceIndex = 0
        return audio_numpy, currentSubsentenceIndex

def cleanSaves(outputFile,intermediaryFormat):
    if os.path.exists(outputFile+".sav"):
        os.remove(outputFile+".sav")
    if os.path.exists(outputFile+intermediaryFormat+".part"):
        os.remove(outputFile+intermediaryFormat+".part")
    print("[+] Partial files "+outputFile+".sav, "+outputFile+intermediaryFormat+".part and "+outputFile+intermediaryFormat+" removed")

import ffmpeg
def convertFormat(sourceFile, format, speed):
    sourceSize=os.stat(sourceFile).st_size
    stream = ffmpeg.input(sourceFile)
    stream = stream.audio.filter("atempo", speed)  # slow it down by  10%
    stream = ffmpeg.output(stream, os.path.splitext(sourceFile)[0]+format)
    stream = stream.global_args('-loglevel', 'quiet')
    stream = stream.global_args('-y')
    ffmpeg.run(stream)
    convertedSize=os.stat(os.path.splitext(sourceFile)[0]+format).st_size
    os.remove(sourceFile)
    print("[+] Converted "+ sourceFile+ " ("+str(round(sourceSize/1024**2,2))+"MB) to "+
            os.path.splitext(sourceFile)[0]+format+" ("+str(round(convertedSize/1024**2,2))+
            "MB) and removed source file" )


import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram(mel_spectrogram):
    plt.imshow(np.transpose(mel_spectrogram), cmap='viridis', origin='lower', aspect='auto')
    plt.xlabel('Time')
    plt.ylabel('Mel Bin')
    plt.title('Mel Spectrogram')
    plt.colorbar()
    plt.show()

# Example usage:
# Replace `mel_spectrogram_data` with your actual mel spectrogram data
#mel_spectrogram_data = np.random.rand(128, 100)  # Example data
#plot_mel_spectrogram(mel_spectrogram_data)


import matplotlib.pyplot as plt
import numpy as np

def plot_mel_alignment_gate(mel, alignment_matrix, gate_output):
    plt.figure(figsize=(10, 8))

    # Plot Mel Spectrogram
    plt.subplot(3, 1, 1)
    plt.imshow(np.transpose(mel), cmap='viridis', origin='lower', aspect='auto')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Bin')
    plt.colorbar()

    # Plot Alignment Matrix
    plt.subplot(3, 1, 2)
    plt.imshow(np.transpose(alignment_matrix), cmap='viridis', origin='lower', aspect='auto')
    plt.title('Alignment Matrix')
    plt.xlabel('Time (Mel Spectrogram)')
    plt.ylabel('Time (Input Text)')
    plt.colorbar()

    # Plot Gate Output
    plt.subplot(3, 1, 3)
    plt.plot(gate_output, label='Gate Output')
    plt.title('Gate Output')
    plt.xlabel('Time (Mel Spectrogram)')
    plt.ylabel('Gate Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
# Replace `mel_data`, `alignment_data`, and `gate_data` with your actual data
#mel_data = np.random.rand(128, 100)  # Example mel spectrogram data
#alignment_data = np.random.rand(100, 50)  # Example alignment matrix data
#gate_data = np.random.rand(100)  # Example gate output data

#plot_mel_alignment_gate(mel_data, alignment_data, gate_data)

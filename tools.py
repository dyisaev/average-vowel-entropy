import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2ForCTC
import json
import parselmouth
import pandas as pd
import numpy as np


def tensor_entropy(input_tensor):
    lsm = nn.LogSoftmax(dim=-1)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    entropy=torch.distributions.Categorical(probs).entropy()
    return entropy

def extract_transcript_and_entropies(clip,model,device,tokenizer):
    model=model.to(device)
    waveform=torch.Tensor(clip.to_soundarray().mean(axis=1))
    waveform=torch.unsqueeze(waveform,0)
    waveform=waveform.to(device)
    sample_rate=clip.fps
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    with torch.inference_mode():
        emissions = model(waveform).logits
    entropy_ts=tensor_entropy(emissions[0]).cpu().detach().numpy()

    pred_ids=torch.argmax(emissions[0], axis=-1)
    
    transcript=tokenizer.decode(pred_ids, output_char_offsets=True)
    token_len=waveform.shape[1]/16000/len(emissions[0])
    
    return transcript,token_len,entropy_ts

def compute_AVE_and_MISD(clip,char_offsets,entropy_ts,token_len,vocab,vowel_indices):
    vocab_byval= {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    vowel_symbols=[list(vocab_byval.keys())[i] for i in vowel_indices]

    vowel_arr=[(elem['char'],elem['start_offset'],next_elem['start_offset'],elem['end_offset'])\
                for elem,next_elem in zip(char_offsets[:-1],char_offsets[1:]) if elem['char'] in vowel_symbols]
    ent_intens_arr=[]
    
    for char_elem,start_offset,next_start_offset,end_offset in vowel_arr:
        ent_vow=np.mean(entropy_ts[start_offset:end_offset])
        vow_clip=clip.subclip(t_start=start_offset*token_len,t_end=next_start_offset*token_len)
        waveform=vow_clip.to_soundarray().mean(axis=1)
        sample_rate=vow_clip.fps
        sound=parselmouth.Sound(waveform,sample_rate)
        try:
            pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, 75, 350)
        except Exception as e:
            pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, 150.0, 350)
        stdevF0 = parselmouth.praat.call(pitch, "Get standard deviation", 0 ,0, 'Hertz') # get standard deviation
        #if pitch is not detected - continue
        if(np.isnan(stdevF0)):
            continue
        try:
            intens = parselmouth.praat.call(sound,"To Intensity",100,0,'on')
        except Exception as e:
            #recomputation for case of higher pitch
            intens = parselmouth.praat.call(sound,"To Intensity",320,0,'on')
        stdev_intens=parselmouth.praat.call(intens,"Get standard deviation",intens.get_start_time(),intens.get_end_time())
        ent_intens_arr.append([char_elem,ent_vow,stdev_intens])
    df_vowels=pd.DataFrame(ent_intens_arr,columns=['vowel','ent_vow','stdev_intens'])
        
    AVE = df_vowels['ent_vow'].mean()
    MISD = df_vowels['stdev_intens'].mean()
    return (df_vowels,AVE,MISD)
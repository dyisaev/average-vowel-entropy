import moviepy.editor as mp
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2ForCTC
import json
import torch
from tools import extract_transcript_and_entropies, compute_AVE_and_MISD
filename='<>'

# load vocabulary and prepare vowel indices
vocab=json.load(open('vocab.json','r'))
vowel_indices=[1,2,3,4,5,11,12,13,18,19,20,26,27,34,35]

# load model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2_phonemes_model/")
tokenizer=Wav2Vec2PhonemeCTCTokenizer(vocab_file='vocab.json',pad_token='[PAD]')

clip=mp.AudioFileClip(filename)


#extract transcript with char offsets + entropies
transcript, token_len , entropy_ts = extract_transcript_and_entropies(clip,model,device,tokenizer)

#extract full dataset of vowels with entropy and intensity standard deviation per vowel + final AVE and MISD
df_vowels,ave,misd=compute_AVE_and_MISD(clip,transcript['char_offsets'],entropy_ts,token_len,vocab,vowel_indices)


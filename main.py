import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import torch
from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DURATION = 5
FS = 16000
text=""



st.title("Grammar Checker")
st.subheader("Read the following text in Hindi and translate it to English. "
             "The AI will understand what you have spoken and check you grammar")

def load_model():
    model_s = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
    d = ModelDownloader("exp/")
    speech2text = Speech2Text(
        **d.download_and_unpack(model_s),
        device="cuda",
        minlenratio=0.0,
        maxlenratio=0.0,
        ctc_weight=0.3,
        beam_size=10,
        batch_size=0,
        nbest=1
    )
    return speech2text

col1,col2,col3 = st.columns(3)

with col1:
    if st.button("Record"):
        duration = DURATION  # seconds
        myrecording = sd.rec(int(duration * FS), samplerate=FS, channels=2)
        sd.wait()
        write("recorded_audio/audio.wav",FS,myrecording)
        st.write("Recording Over")

with col2:
    if st.button("Play"):

        rec,fs = sf.read("recorded_audio/audio.wav",dtype="float32")

        sd.play(rec,fs)
        sd.wait()


with col3:
    if st.button("Transcribe"):
        model = load_model()
        rec,fs = sf.read("recorded_audio/audio.wav",dtype="float32")
        nbests = model(rec)
        text, *_ = nbests[0]

if len(text)>0:
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">'+"You said: "+text+'</p>', unsafe_allow_html=True)













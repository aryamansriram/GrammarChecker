import librosa
import numpy as np
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import torch
from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import librosa.display

DURATION = 5
FS = 16000
text=""
DEVICE = "cuda"

st.session_state["show_spec"] = 0

st.title("Grammar Checker")
st.subheader("Say anything and the AI will try to check your grammar.")

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

def predict_logits(model, tokenizer, sentence, device=DEVICE):
    input_ids = torch.tensor(tokenizer.encode(sentence)).reshape(1, -1)
    input_ids = input_ids.to(device)
    model = model.to(device)
    output_probs = torch.nn.functional.softmax(model(input_ids).logits)
    if device == "cuda":
        output_probs = output_probs.detach().cpu()

    return output_probs


def plot_spec(path="recorded_audio/audio.wav"):
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    return fig


col1,col2,col3 = st.columns(3)

with col1:
    if st.button("Record"):
        duration = DURATION  # seconds
        myrecording = sd.rec(int(duration * FS), samplerate=FS, channels=2)
        sd.wait()
        write("recorded_audio/audio.wav",FS,myrecording)
        st.write("Recording Over")
        st.session_state["show_spec"] = 1

with col2:
    if st.button("Play"):

        rec,fs = sf.read("recorded_audio/audio.wav",dtype="float32")

        sd.play(rec,fs)
        sd.wait()


with col3:
    if st.button("Check Grammar"):
        model = load_model()
        rec,fs = sf.read("recorded_audio/audio.wav",dtype="float32")
        nbests = model(rec)
        text, *_ = nbests[0]

if st.sidebar.checkbox("Show spectrogram"):
    st.subheader("Spectrogram of Recorded Audio")
    fig = plot_spec()
    st.pyplot(fig)



if len(text)>0:
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">'+"You said: "+text+'</p>', unsafe_allow_html=True)

    with st.spinner("Loading Transformer model and tokenizer....."):
        tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
        model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
        tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    st.success("Model and tokenizer loaded successfully!")
    with st.spinner("Making predictions: "):
        logits = predict_logits(model,tokenizer,text.lower())
    st.success("Inference successful")
    z_prob = logits[0][0]
    one_prob = logits[0][1]
    st.markdown("Probability of incorrectness: "+str(z_prob.item()))
    st.markdown("Probability of correctness: "+str(one_prob.item()))














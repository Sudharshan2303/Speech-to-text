import streamlit as st
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import streamlit as st
def speech_to_text(audio_data):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    input_audio, _ = librosa.load(audio_data,sr=16000)
    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription
def main():
    st.title("Speech2Text")
    audio_file = st.file_uploader("Upload audio data",type=["wav", "mp3"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav', start_time=0)
        if st.button("Convert"):
            text = speech_to_text(audio_file)
            st.success("Converted")
            st.write(text)

if __name__ == "__main__":
    main()

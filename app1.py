import streamlit as st
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf

def speech_to_text(audio_data):
    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Read audio file
    speech_array, sampling_rate = sf.read(audio_data)
    
    # Preprocess
    inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    # Inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

def main():
    st.title("Speech2Text Converter")
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        if st.button("Convert to Text"):
            with st.spinner("Processing..."):
                try:
                    text = speech_to_text(audio_file)
                    st.success("Conversion Complete")
                    st.text_area("Transcription", text, height=100)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

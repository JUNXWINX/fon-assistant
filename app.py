import os
import gradio as gr
import torch
import scipy.io.wavfile
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    VitsModel
)

# --- Port fourni par Render ---
port = int(os.environ.get("PORT", 7860))

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device)

# --- 1. ASR ---
print("Chargement ASR...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="Professor/mms-300m-fongbe",
    device=-1  # CPU sur Render
)

# --- 2. Traduction fon -> français ---
print("Chargement traduction...")
nllb_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device)

def traduire_fon_fr(texte):
    nllb_tokenizer.src_lang = "fon_Latn"
    inputs = nllb_tokenizer(texte, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("fra_Latn"),
            max_length=200
        )
    return nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 3. TTS fon ---
print("Chargement TTS...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-fon").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fon")

def parler_en_fon(texte, fichier="reponse_fon.wav"):
    inputs = tts_tokenizer(texte, return_tensors="pt").to(device)
    with torch.no_grad():
        output = tts_model(**inputs).waveform
    scipy.io.wavfile.write(
        fichier,
        rate=tts_model.config.sampling_rate,
        data=output.cpu().numpy().squeeze()
    )
    return fichier

# --- 4. Fonctions ---
def pipeline_vocal(audio_path):
    if audio_path is None:
        return "Aucun audio.", "", None
    try:
        transcription = asr_pipeline(audio_path)["text"]
        traduction = traduire_fon_fr(transcription)
        audio_reponse = parler_en_fon(f"Un se. {transcription}", "reponse.wav")
        return transcription, traduction, audio_reponse
    except Exception as e:
        return f"Erreur : {e}", "", None

def pipeline_texte(question):
    if not question.strip():
        return "", None
    try:
        traduction = traduire_fon_fr(question)
        audio_reponse = parler_en_fon(question, "reponse_texte.wav")
        return traduction, audio_reponse
    except Exception as e:
        return f"Erreur : {e}", None

# --- 5. Interface ---
with gr.Blocks(title="Assistant Fon") as demo:
    gr.Markdown("# 🎙️ Assistant vocal Fon")
    gr.Markdown("Parle ou écris en fon.")

    with gr.Tab("🎤 Parler"):
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Votre voix")
        btn = gr.Button("Analyser", variant="primary")
        out_trans = gr.Textbox(label="📝 Transcription")
        out_trad = gr.Textbox(label="🇫🇷 Traduction")
        out_audio = gr.Audio(label="🔊 Réponse")
        btn.click(pipeline_vocal, inputs=audio_in, outputs=[out_trans, out_trad, out_audio])

    with gr.Tab("⌨️ Écrire"):
        txt_in = gr.Textbox(label="Écris en fon")
        btn2 = gr.Button("Traduire", variant="primary")
        out_trad2 = gr.Textbox(label="🇫🇷 Traduction")
        out_audio2 = gr.Audio(label="🔊 Audio")
        btn2.click(pipeline_texte, inputs=txt_in, outputs=[out_trad2, out_audio2])

demo.launch(server_name="0.0.0.0", server_port=port)

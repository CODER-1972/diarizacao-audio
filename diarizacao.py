"""
diarizacao.py
Script para diarização de áudio com compatibilidade melhorada.
Identifica oradores e agrupa falas consecutivas.
"""

import os
import json
import sys
import subprocess
import tempfile
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from configparser import ConfigParser
import warnings
warnings.filterwarnings("ignore")

def create_diarization_requirements():
    """Cria requirements.txt específico para diarização"""
    requirements_content = """# Dependências para diarização - versões compatíveis
torch>=2.0.0
torchaudio>=2.0.0
pyannote.audio>=3.1.0
pytorch-lightning>=2.0.0
transformers>=4.20.0
speechbrain>=0.5.15
numpy>=1.21.0
tqdm>=4.64.0
ffmpeg-python>=0.2.0
scipy>=1.9.0
librosa>=0.9.0
"""
    
    with open("requirements_diarization.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content.strip())
    
    print("✅ Arquivo requirements_diarization.txt criado")
    print("💡 Execute: pip install -r requirements_diarization.txt")

def load_config():
    """Carrega configurações do config.ini"""
    default_config = {
        'AUDIO_PATH': '',
        'CHUNK_DURATION': 600,  # 10 minutos
        'OVERLAP': 3,
        'PAUSE_THRESHOLD': 1.5
    }
    
    try:
        config = ConfigParser()
        config.read('config.ini', encoding='utf-8')
        if 'config' in config:
            return {
                'AUDIO_PATH': config.get('config', 'audio_path', fallback=''),
                'CHUNK_DURATION': config.getint('config', 'chunk_duration_diarize', fallback=600),
                'OVERLAP': config.getint('config', 'overlap_diarize', fallback=3),
                'PAUSE_THRESHOLD': config.getfloat('config', 'pause_threshold', fallback=1.5)
            }
    except Exception as e:
        print(f"⚠️ Erro ao carregar configurações: {e}")
    
    return default_config

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    try:
        import torch
        import pyannote.audio
        from pyannote.audio import Pipeline
        
        # Verifica versões
        torch_version = torch.__version__
        pyannote_version = pyannote.audio.__version__
        
        print(f"📦 PyTorch: {torch_version}")
        print(f"📦 pyannote.audio: {pyannote_version}")
        
        # Verifica se as versões são compatíveis
        torch_major = int(torch_version.split('.')[0])
        if torch_major < 2:
            print("⚠️ PyTorch < 2.0 pode causar problemas")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Dependência não encontrada: {e}")
        return False

def get_audio_path():
    """Solicita caminho do arquivo de áudio"""
    while True:
        audio_path = input("\n📁 Digite o caminho completo do arquivo de áudio: ").strip().strip('"\'')
        
        if not audio_path:
            print("❌ Caminho não pode estar vazio")
            continue
        
        if not os.path.exists(audio_path):
            print(f"❌ Arquivo não encontrado: {audio_path}")
            continue
        
        return audio_path

def get_audio_duration(audio_path):
    """Obtém duração do áudio usando ffprobe"""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", audio_path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"❌ Erro ao obter duração: {e}")
        sys.exit(1)

def split_audio(audio_path, chunk_duration, overlap):
    """Divide o áudio em blocos temporais"""
    duration = get_audio_duration(audio_path)
    print(f"🔧 Dividindo áudio ({duration:.1f}s) em blocos de {chunk_duration//60}min")
    
    chunks = []
    temp_dir = tempfile.mkdtemp(prefix="diarize_chunks_")
    start_time = 0
    
    while start_time < duration:
        chunk_num = len(chunks) + 1
        end_time = min(start_time + chunk_duration, duration)
        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num:03d}.wav")
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start_time), "-t", str(end_time - start_time),
                "-c", "copy", chunk_path
            ], check=True, capture_output=True)
            
            chunks.append({
                "path": chunk_path,
                "start_offset": start_time,
                "end_offset": end_time,
                "chunk_num": chunk_num
            })
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro criando bloco {chunk_num}: {e}")
        
        start_time = end_time - overlap if end_time < duration else end_time
    
    print(f"✅ {len(chunks)} blocos criados em {temp_dir}")
    return chunks, temp_dir

class SpeakerTracker:
    """Rastreia oradores de forma simples baseado em características temporais"""
    
    def __init__(self):
        self.speakers = {}
        self.counter = 0
    
    def get_or_create_speaker(self, segment_info):
        """Obtém ou cria ID de orador baseado em características simples"""
        # Características simples baseadas em timing
        avg_duration = np.mean([s["end"] - s["start"] for s in segment_info])
        
        # Procura orador similar
        for speaker_id, features in self.speakers.items():
            if abs(features["avg_duration"] - avg_duration) < 0.5:  # Tolerância de 0.5s
                return speaker_id
        
        # Cria novo orador
        self.counter += 1
        new_id = f"ORADOR_{self.counter:02d}"
        self.speakers[new_id] = {"avg_duration": avg_duration}
        return new_id

def diarize_chunk_simple(chunk, pipeline):
    """Executa diarização em um bloco usando pyannote"""
    try:
        # Executa pipeline de diarização
        diarization = pipeline(chunk["path"])
        
        # Processa resultados
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": segment.start + chunk["start_offset"],
                "end": segment.end + chunk["start_offset"],
                "speaker": speaker,
                "chunk_num": chunk["chunk_num"]
            })
        
        return segments
        
    except Exception as e:
        print(f"❌ Erro na diarização do bloco {chunk['chunk_num']}: {e}")
        return []

def merge_consecutive_segments(segments, pause_threshold):
    """Agrupa segmentos consecutivos do mesmo orador"""
    if not segments:
        return []
    
    # Ordena por tempo de início
    segments.sort(key=lambda x: x["start"])
    
    merged = []
    current = segments[0].copy()
    
    for segment in segments[1:]:
        # Calcula pausa entre segmentos
        pause = segment["start"] - current["end"]
        
        # Mesmo orador e pausa curta: agrupa
        if segment["speaker"] == current["speaker"] and pause <= pause_threshold:
            current["end"] = segment["end"]
        else:
            # Diferente orador ou pausa longa: novo segmento
            merged.append(current)
            current = segment.copy()
    
    # Adiciona último segmento
    merged.append(current)
    
    return merged

def load_transcription(transcription_dir):
    """Carrega transcrição do diretório gerado pelo transcribe_audio.py"""
    json_files = list(Path(transcription_dir).glob("transcricao_temporizada.json"))
    
    if not json_files:
        print("⚠️ Arquivo de transcrição não encontrado")
        return []
    
    with open(json_files[0], "r", encoding="utf-8") as f:
        return json.load(f)

def align_transcription_with_speakers(transcription, speaker_segments):
    """Alinha transcrição com segmentos de oradores"""
    aligned = []
    
    for trans_seg in transcription:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_text = trans_seg["text"]
        
        # Encontra orador que mais se sobrepõe com este segmento
        best_speaker = "DESCONHECIDO"
        best_overlap = 0
        
        for speaker_seg in speaker_segments:
            # Calcula sobreposição temporal
            overlap_start = max(trans_start, speaker_seg["start"])
            overlap_end = min(trans_end, speaker_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_seg["speaker"]
        
        aligned.append({
            "start": trans_start,
            "end": trans_end,
            "text": trans_text,
            "speaker": best_speaker
        })
    
    return aligned

def main():
    print("🎤 Diarização de Áudio com pyannote.audio\n")
    
    # Verifica dependências
    if not check_dependencies():
        print("\n❌ Dependências incompatíveis ou não instaladas")
        create_diarization_requirements()
        print("\n🔧 Para corrigir:")
        print("1. pip install -r requirements_diarization.txt")
        print("2. Execute o script novamente")
        sys.exit(1)
    
    # Verifica token do Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Token do Hugging Face não encontrado")
        print("🔧 Para configurar:")
        print("1. Acesse: https://huggingface.co/settings/tokens")
        print("2. Crie um token com acesso de leitura")
        print("3. Execute: set HF_TOKEN=seu_token_aqui (Windows)")
        print("   ou: export HF_TOKEN=seu_token_aqui (Linux/Mac)")
        sys.exit(1)
    
    # Carregar configuração
    config = load_config()
    
    # Obter caminho do áudio
    audio_path = get_audio_path()
    
    print(f"""
⚙️ Configurações de Diarização:
- Arquivo de áudio: {audio_path}
- Duração do bloco: {config['CHUNK_DURATION']}s
- Sobreposição: {config['OVERLAP']}s
- Limiar de pausa: {config['PAUSE_THRESHOLD']}s
""")
    
    # Carregar pipeline de diarização
    print("🔧 Carregando modelo de diarização...")
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        print("✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"❌ Falha ao carregar modelo: {e}")
        print("💡 Possíveis soluções:")
        print("- Verifique se o token tem acesso ao modelo")
        print("- Aceite os termos de uso em: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("- Atualize as dependências")
        sys.exit(1)
    
    # Criar diretório de saída
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_name = Path(audio_path).stem
    output_dir = f"diarizacao_{audio_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dividir áudio em blocos
    chunks, temp_dir = split_audio(audio_path, config['CHUNK_DURATION'], config['OVERLAP'])
    
    # Processar cada bloco
    all_segments = []
    print("\n🔊 Iniciando diarização...")
    
    for chunk in tqdm(chunks, desc="Processando blocos"):
        segments = diarize_chunk_simple(chunk, pipeline)
        all_segments.extend(segments)
        print(f"   ✓ Bloco {chunk['chunk_num']}: {len(segments)} segmentos")
    
    # Agrupar segmentos consecutivos
    print("🔧 Agrupando segmentos consecutivos...")
    merged_segments = merge_consecutive_segments(all_segments, config['PAUSE_THRESHOLD'])
    
    # Procurar transcrição existente
    transcription_dirs = [d for d in os.listdir('.') if d.startswith('transcricao_')]
    if transcription_dirs:
        print(f"📝 Carregando transcrição de: {transcription_dirs[-1]}")
        transcription = load_transcription(transcription_dirs[-1])
        
        # Alinhar transcrição com oradores
        aligned_results = align_transcription_with_speakers(transcription, merged_segments)
    else:
        print("⚠️ Nenhuma transcrição encontrada - salvando apenas diarização")
        aligned_results = merged_segments
    
    # Salvar resultados
    # Arquivo de texto simples
    output_txt = os.path.join(output_dir, "diarizacao_completa.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"# Diarização de Áudio\n")
        f.write(f"# Arquivo: {audio_path}\n")
        f.write(f"# Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for item in aligned_results:
            start_min = int(item["start"] // 60)
            start_sec = int(item["start"] % 60)
            end_min = int(item["end"] // 60)
            end_sec = int(item["end"] % 60)
            
            time_str = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
            speaker = item.get("speaker", "DESCONHECIDO")
            text = item.get("text", "")
            
            if text:
                f.write(f"{time_str} {speaker}: {text}\n")
            else:
                f.write(f"{time_str} {speaker}\n")
    
    # Arquivo JSON detalhado
    output_json = os.path.join(output_dir, "diarizacao_detalhada.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(aligned_results, f, ensure_ascii=False, indent=2)
    
    # Estatísticas
    speakers = set(item.get("speaker", "DESCONHECIDO") for item in aligned_results)
    total_duration = sum(item["end"] - item["start"] for item in aligned_results)
    
    # Limpeza
    shutil.rmtree(temp_dir)
    
    print(f"\n🎉 Diarização concluída!")
    print(f"📁 Diretório de saída: {output_dir}")
    print(f"👥 Oradores identificados: {len(speakers)} ({', '.join(sorted(speakers))})")
    print(f"⏱️ Duração total processada: {total_duration:.1f}s")
    print(f"📝 Arquivo de texto: {os.path.basename(output_txt)}")
    print(f"📊 Arquivo JSON: {os.path.basename(output_json)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Execução interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

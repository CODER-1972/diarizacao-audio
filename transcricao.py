"""
transcribe_audio.py
Script completo para transcrição de áudio com:
- Criação automática do requirements.txt
- Instalação automática de dependências
- Configuração compartilhada
- Interface interativa
"""

import os
import sys
import subprocess
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from faster_whisper import WhisperModel

def create_requirements_txt():
    """Cria requirements.txt apenas com dependências"""
    requirements_content = """# Dependências do projeto
faster-whisper==0.9.0
pyannote.audio==3.1.1
numpy==1.26.4
tqdm==4.66.2
ffmpeg-python==0.2.0
"""
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content.strip())
    
    print("✅ Arquivo requirements.txt criado")

def create_config_file():
    """Cria arquivo de configuração separado"""
    config_content = f"""[config]
audio_path = {str(Path.home() / "audio-to-text.wav")}
chunk_duration_transcribe = 900
chunk_duration_diarize = 600
overlap_diarize = 3
pause_threshold = 1.5
default_model = large-v2
default_language = pt
"""
    
    with open("config.ini", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ Arquivo config.ini criado")

def install_requirements():
    """Instala dependências automaticamente"""
    print("🔍 Verificando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas")
        return True
    except subprocess.CalledProcessError:
        print("❌ Falha ao instalar dependências")
        return False

def get_config():
    """Lê configurações do config.ini"""
    config = {
        'audio_path': '',
        'chunk_duration': 900,
        'default_model': 'large-v2',
        'default_language': 'pt'
    }
    
    try:
        from configparser import ConfigParser
        config_parser = ConfigParser()
        config_parser.read('config.ini', encoding='utf-8')
        
        if 'config' in config_parser:
            config.update({
                'audio_path': config_parser.get('config', 'audio_path'),
                'chunk_duration': config_parser.getint('config', 'chunk_duration_transcribe'),
                'default_model': config_parser.get('config', 'default_model'),
                'default_language': config_parser.get('config', 'default_language')
            })
    except Exception as e:
        print(f"⚠️ Erro lendo configurações: {e}")
    
    return config

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
        
        # Verifica se é um arquivo de áudio válido
        audio_extensions = ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.wma']
        if not any(audio_path.lower().endswith(ext) for ext in audio_extensions):
            print(f"⚠️ Extensão não reconhecida. Continuando mesmo assim...")
        
        return audio_path

def get_audio_duration(audio_path):
    """Obtém duração do áudio usando ffprobe"""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", audio_path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao obter duração do áudio. Verifique se o ffmpeg está instalado e o arquivo é válido.")
        print(f"Comando executado: ffprobe -v quiet -show_entries format=duration -of csv=p=0 '{audio_path}'")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Erro ao interpretar duração: {e}")
        sys.exit(1)

def split_audio(audio_path, chunk_duration, overlap):
    """Divide o áudio em blocos temporais"""
    duration = get_audio_duration(audio_path)
    print(f"🔧 Dividindo áudio ({duration:.1f}s) em blocos de {chunk_duration//60}min")
    
    chunks = []
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
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

def transcribe_chunk(chunk, model, language):
    """Transcreve um bloco de áudio"""
    try:
        segments, _ = model.transcribe(chunk["path"], language=language)
        return [{
            "start": s.start + chunk["start_offset"],
            "end": s.end + chunk["start_offset"],
            "text": s.text.strip()
        } for s in segments]
    except Exception as e:
        print(f"❌ Erro na transcrição do bloco {chunk['chunk_num']}: {e}")
        return []

def select_option(options, prompt):
    """Menu interativo para seleção de opções"""
    print(f"\n{prompt}")
    for key, value in options.items():
        print(f"  {key}. {value[1] if isinstance(value, tuple) else value}")
    
    while True:
        choice = input("Selecione: ").strip()
        if choice in options:
            return options[choice][0] if isinstance(options[choice], tuple) else options[choice]
        print("❌ Opção inválida, tente novamente")

def check_ffmpeg():
    """Verifica se ffmpeg e ffprobe estão instalados"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg não encontrado. Instale o FFmpeg:")
        print("   - Windows: https://ffmpeg.org/download.html")
        print("   - macOS: brew install ffmpeg")
        print("   - Linux: sudo apt install ffmpeg")
        return False

def main():
    print("🎤 Transcritor de Áudio com Whisper\n")
    
    # Verificar FFmpeg
    if not check_ffmpeg():
        sys.exit(1)
    
    # Configuração inicial
    if not os.path.exists("requirements.txt"):
        create_requirements_txt()
    
    if not os.path.exists("config.ini"):
        create_config_file()
    
    if not install_requirements():
        sys.exit(1)
    
    # Obter caminho do áudio
    audio_path = get_audio_path()
    
    # Menu de seleção
    models = {
        "1": ("tiny", "Tiny (mais rápido, menos preciso)"),
        "2": ("base", "Base"),
        "3": ("small", "Small"),
        "4": ("medium", "Medium"),
        "5": ("large-v2", "Large-v2 (mais lento, mais preciso)")
    }
    
    languages = {
        "1": ("pt", "Português"),
        "2": ("en", "Inglês"),
        "3": (None, "Detectar automaticamente")
    }
    
    selected_model = select_option(models, "📋 Selecione o modelo Whisper:")
    selected_lang = select_option(languages, "🌐 Selecione o idioma:")
    
    # Preparar diretório de saída
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_name = Path(audio_path).stem
    output_dir = f"transcricao_{audio_name}_{selected_model}_{selected_lang or 'auto'}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar modelo
    print(f"\n🔧 Carregando modelo {selected_model}...")
    try:
        model = WhisperModel(selected_model, device="cpu", compute_type="int8")
        print("✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"❌ Falha ao carregar modelo: {e}")
        sys.exit(1)
    
    # Processar áudio
    chunk_duration = 900  # 15 minutos
    chunks, temp_dir = split_audio(audio_path, chunk_duration, 0)
    full_transcript = []
    
    print("\n🎤 Iniciando transcrição...")
    for i, chunk in enumerate(chunks, 1):
        print(f"🔊 Processando bloco {i}/{len(chunks)}...")
        segments = transcribe_chunk(chunk, model, selected_lang)
        full_transcript.extend(segments)
        print(f"   ✓ {len(segments)} segmentos transcritos")
    
    # Salvar resultados
    output_file = os.path.join(output_dir, "transcricao_completa.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Transcrição completa\n")
        f.write(f"# Arquivo: {audio_path}\n")
        f.write(f"# Modelo: {selected_model}\n")
        f.write(f"# Idioma: {selected_lang or 'auto'}\n")
        f.write(f"# Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(seg['text'] for seg in full_transcript))
    
    # Salvar versão com tempos
    timed_file = os.path.join(output_dir, "transcricao_temporizada.json")
    with open(timed_file, "w", encoding="utf-8") as f:
        json.dump(full_transcript, f, ensure_ascii=False, indent=2)
    
    # Versão SRT para legendas
    srt_file = os.path.join(output_dir, "transcricao_legendas.srt")
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(full_transcript, 1):
            start_time = f"{int(seg['start']//3600):02d}:{int((seg['start']%3600)//60):02d}:{seg['start']%60:06.3f}".replace('.', ',')
            end_time = f"{int(seg['end']//3600):02d}:{int((seg['end']%3600)//60):02d}:{seg['end']%60:06.3f}".replace('.', ',')
            f.write(f"{i}\n{start_time} --> {end_time}\n{seg['text']}\n\n")
    
    # Limpeza
    shutil.rmtree(temp_dir)
    
    print(f"\n🎉 Transcrição concluída!")
    print(f"📁 Diretório de saída: {output_dir}")
    print(f"📝 Texto completo: {os.path.basename(output_file)}")
    print(f"⏱️ Versão temporizada: {os.path.basename(timed_file)}")
    print(f"🎬 Legendas SRT: {os.path.basename(srt_file)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Execução interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

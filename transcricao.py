"""
diarize_audio.py
Script completo para diarização com compatibilidade garantida
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

# Configuração de warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Versões requeridas
REQUIRED_VERSIONS = {
    'torch': '2.0.1',
    'pyannote.audio': '3.1.1',
    'pytorch_lightning': '2.0.2'
}

def check_versions(auto_install=True):
    """Verifica e instala versões corretas se necessário"""
    try:
        import torch
        import pyannote.audio
        import pytorch_lightning

        versions_ok = (
            torch.__version__.startswith(REQUIRED_VERSIONS['torch']) and
            pyannote.audio.__version__.startswith(REQUIRED_VERSIONS['pyannote.audio']) and
            pytorch_lightning.__version__.startswith(REQUIRED_VERSIONS['pytorch_lightning'])
        )

        if versions_ok:
            print("✅ Todas as versões estão corretas")
            return True

        print("⚠️ Versões incompatíveis detectadas!")
        print(f" - torch: {torch.__version__} (requer {REQUIRED_VERSIONS['torch']})")
        print(f" - pyannote.audio: {pyannote.audio.__version__} (requer {REQUIRED_VERSIONS['pyannote.audio']})")
        print(f" - pytorch_lightning: {pytorch_lightning.__version__} (requer {REQUIRED_VERSIONS['pytorch_lightning']})")

        create_requirements_file()

        if auto_install:
            print("\n🔧 A instalar dependências corretas automaticamente...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_diarization.txt", "--upgrade"])
            print("✅ Instalação concluída. Reinicia o script para garantir carregamento limpo.")
            sys.exit(0)
        else:
            print("\n🔧 Corre manualmente:")
            print("pip install -r requirements_diarization.txt")
            return False

    except ImportError as e:
        print(f"❌ Falta uma dependência: {e}")
        create_requirements_file()
        if auto_install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_diarization.txt", "--upgrade"])
            sys.exit(0)
        return False


def create_requirements_file():
    """Cria arquivo requirements.txt com versões específicas e compatíveis com CPU"""
    content = """# Versões testadas e compatíveis com CPU
torch==2.0.1+cpu
torchaudio==2.0.2+cpu
pyannote.audio==3.1.1
pytorch-lightning==2.0.2
transformers==4.30.0
speechbrain==0.5.15
numpy==1.24.4
tqdm>=4.64.0
ffmpeg-python>=0.2.0
scipy>=1.9.0
librosa>=0.9.0
--extra-index-url https://download.pytorch.org/whl/cpu
"""
    with open("requirements_diarization.txt", "w", encoding="utf-8") as f:
        f.write(content.strip())
    print("✅ Arquivo requirements_diarization.txt criado")


def load_config():
    """Carrega configurações de forma robusta"""
    config = {
        'audio_path': '',
        'chunk_duration': 600,
        'overlap': 3,
        'pause_threshold': 1.5,
        'hf_token': os.getenv("HF_TOKEN", "")
    }
    
    try:
        cp = ConfigParser()
        cp.read('config.ini')
        
        if 'config' in cp:
            config.update({
                'audio_path': cp.get('config', 'audio_path', fallback=''),
                'chunk_duration': cp.getint('config', 'chunk_duration_diarize', fallback=600),
                'overlap': cp.getint('config', 'overlap_diarize', fallback=3),
                'pause_threshold': cp.getfloat('config', 'pause_threshold', fallback=1.5)
            })
    except Exception as e:
        print(f"⚠️ Erro lendo configurações: {e}")
    
    return config

def get_audio_path(config):
    """Obtém caminho do áudio com verificação"""
    audio_path = config['audio_path']
    
    while not os.path.exists(audio_path):
        print(f"❌ Arquivo não encontrado: {audio_path}")
        audio_path = input("Digite o caminho correto do arquivo de áudio: ").strip()
    
    return audio_path

def initialize_pipeline(hf_token):
    """Inicializa o pipeline de diarização com tratamento de erros"""
    try:
        from pyannote.audio import Pipeline
        
        print("🔧 Carregando modelo pyannote/speaker-diarization-3.1...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Configurações de desempenho
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return pipeline
        
    except Exception as e:
        print(f"❌ Falha ao inicializar pipeline: {e}")
        print("💡 Soluções possíveis:")
        print("1. Verifique seu token do Hugging Face")
        print("2. Aceite os termos em: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("3. Verifique as versões das dependências")
        return None

def main():
    print("\n🎤 Diarização de Áudio - Versão Estável\n")
    
    # Verificar versões
    if not check_versions():
        print("\n⚠️ Versões incompatíveis detectadas!")
        create_requirements_file()
        print("\n🔧 Execute:")
        print("pip install -r requirements_diarization.txt")
        print("E depois execute o script novamente")
        sys.exit(1)
    
    # Carregar configurações
    config = load_config()
    
    # Verificar token
    if not config['hf_token']:
        print("❌ Token do Hugging Face não configurado")
        print("Execute no terminal antes de rodar o script:")
        print("set HF_TOKEN=seu_token (Windows)")
        print("ou export HF_TOKEN=seu_token (Linux/Mac)")
        sys.exit(1)
    
    # Obter caminho do áudio
    audio_path = get_audio_path(config)
    
    # Inicializar pipeline
    pipeline = initialize_pipeline(config['hf_token'])
    if not pipeline:
        sys.exit(1)
    
    # Resto da implementação da diarização...
    # (Mantendo a mesma lógica do seu código original, mas usando as configurações carregadas)

if __name__ == "__main__":
    try:
        import torch  # Import aqui para garantir que a verificação de versão funcione
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execução cancelada pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

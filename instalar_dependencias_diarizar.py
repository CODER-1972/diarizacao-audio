import os
import sys
import subprocess
from pathlib import Path

# -------- Parâmetros principais --------
ENV_NAME = "diarizacao_env"
ENV_YAML = "env.yaml"
REQ_TXT = "requirements.txt"
# ---------------------------------------

# Conteúdo do ficheiro de requisitos pip
requirements = """
pip<24.1
faster-whisper==1.1.1
ctranslate2==4.6.0
tokenizers==0.21.1
pyannote.audio==2.1.1
torch==1.10.0
torchaudio==0.10.0
pytorch_lightning==1.5.9
huggingface-hub==0.16.4
soundfile==0.10.3.post1
tqdm
requests
humanfriendly
coloredlogs
av
numpy==1.23.5 
"""

# Conteúdo do ficheiro YAML do ambiente Conda
env_yaml = f"""
name: {ENV_NAME}
channels:
  - defaults
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - pip<24.1
"""

# -------- Funções utilitárias --------
def esta_em_ambiente_conda():
    env = os.environ.get("CONDA_DEFAULT_ENV")
    return env not in (None, "base")

def escrever_ficheiros():
    Path(ENV_YAML).write_text(env_yaml.strip(), encoding="utf-8")
    Path(REQ_TXT).write_text(requirements.strip(), encoding="utf-8")
    print(f"✅ Ficheiros '{ENV_YAML}' e '{REQ_TXT}' criados.")

def ambiente_existe(nome):
    try:
        output = subprocess.check_output(["conda", "env", "list"], encoding="utf-8")
        return any(nome in linha for linha in output.splitlines())
    except Exception:
        return False

def criar_ambiente():
    if ambiente_existe(ENV_NAME):
        print(f"ℹ️ Ambiente '{ENV_NAME}' já existe. A saltar criação.")
        return
    print(f"🐍 A criar o ambiente Conda '{ENV_NAME}'…")
    try:
        subprocess.check_call(["conda", "env", "create", "-f", ENV_YAML])
        print(f"✅ Ambiente '{ENV_NAME}' criado com sucesso.")
    except subprocess.CalledProcessError:
        print("❌ Erro ao criar o ambiente. Verifica se o Conda está instalado no PATH.")
        sys.exit(1)

def instalar_pacotes():
    print("📦 A instalar (ou atualizar) pacotes com pip…")
    try:
        subprocess.check_call([
            "conda", "run", "-n", ENV_NAME, "python", "-m", "pip", "install", "-U", "-r", REQ_TXT
        ])
        print("✅ Pacotes instalados com sucesso.")
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar os pacotes com pip.")
        sys.exit(1)

# -------------- MAIN --------------
def main():
    print("📦 Configurador de ambiente para transcrição + diarização local")

    if esta_em_ambiente_conda():
        active = os.environ.get("CONDA_DEFAULT_ENV")
        print(f"⚠️  Estás dentro do ambiente Conda: {active}")
        print("   Por favor, sai primeiro com: conda deactivate")
        sys.exit(1)

    escrever_ficheiros()
    criar_ambiente()
    instalar_pacotes()

    print("\n✅ Tudo pronto!")
    print("💡 O script principal será agora corrido através do ambiente Conda.")

if __name__ == "__main__":
    main()

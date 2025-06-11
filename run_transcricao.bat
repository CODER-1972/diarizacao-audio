@echo off
chcp 65001 > nul

REM ═══════════════════════════════════════════════════════════════
REM 🗂️ 1. Mudar para a pasta de trabalho (usar nome curto, sem acentos)
REM ═══════════════════════════════════════════════════════════════
cd /d "C:\Users\pnico\Videos\TRANSC~1" || (
    echo ❌ ERRO: Pasta de trabalho não encontrada.
    pause
    exit /b
)

REM ═══════════════════════════════════════════════════════════════
REM 🔄 2. Desativar qualquer ambiente Conda ativo
REM ═══════════════════════════════════════════════════════════════
CALL conda deactivate > nul 2>&1

REM ═══════════════════════════════════════════════════════════════
REM 🔧 3. Criar ambiente e instalar dependências
REM ═══════════════════════════════════════════════════════════════
echo 🔧 A verificar/criar ambiente Conda e dependências...
IF NOT EXIST instalar_dependencias_diarizar.py (
    echo ❌ ERRO: Ficheiro instalar_dependencias_diarizar.py não encontrado.
    pause
    exit /b
)

CALL conda run -n base python instalar_dependencias_diarizar.py || (
    echo ❌ ERRO: Falha ao instalar dependências.
    pause
    exit /b
)

REM ═══════════════════════════════════════════════════════════════
REM 4) Definir token se ainda não existir
REM ═══════════════════════════════════════════════════════════════
echo 🔑 Verificar token...
IF "%HF_TOKEN%"=="" (
    set HF_TOKEN=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
    echo 🔑 HF_TOKEN definido temporariamente
)

REM ═══════════════════════════════════════════════════════════════
REM 🧪 5. Verificar scripts necessários
REM ═══════════════════════════════════════════════════════════════
IF NOT EXIST transcricao.py (
    echo ❌ ERRO: Ficheiro transcricao.py não encontrado.
    pause
    exit /b
)

IF NOT EXIST diarizacao.py (
    echo ❌ ERRO: Ficheiro diarizacao.py não encontrado.
    pause
    exit /b
)

REM ═══════════════════════════════════════════════════════════════
REM 🚀 6. Ativar ambiente e correr os scripts
REM ═══════════════════════════════════════════════════════════════
CALL conda.bat activate diarizacao_env || (
    echo ❌ ERRO: Não foi possível ativar o ambiente diarizacao_env.
    pause
    exit /b
)

echo 🚀 A iniciar transcrição de áudio...
python transcricao.py || (
    echo ❌ ERRO: Falha ao correr transcricao.py
    pause
    exit /b
)

echo 🔊 A iniciar diarização com pyannote...
python diarizacao.py || (
    echo ❌ ERRO: Falha ao correr diarizacao.py
    pause
    exit /b
)

REM ═══════════════════════════════════════════════════════════════
REM ✅ 7. Sucesso!
REM ═══════════════════════════════════════════════════════════════
echo.
echo ✅ Processo concluído com sucesso!
pause

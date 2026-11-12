import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

import dados

BASE = Path(__file__).parent
PASTA_UPLOADS = BASE / "uploads"
ULTIMO_CSV = PASTA_UPLOADS / "ultimo_limpo.csv"             
ULTIMO_RELATORIO = PASTA_UPLOADS / "ultimo_relatorio.json"  

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  
app.secret_key = os.environ.get("SECRET_KEY", "chave-de-desenvolvimento") 


def salvar_resultado(df_limpo, relatorio, origem, conteudo_original=None):
    """Guarda o CSV original sem alterações e o resultado da limpeza em uploads/."""
    PASTA_UPLOADS.mkdir(exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    if conteudo_original is not None:
        nome = f"original_{carimbo}_{secure_filename(origem) or 'arquivo.csv'}"
        (PASTA_UPLOADS / nome).write_bytes(conteudo_original)
    relatorio["origem"] = origem
    relatorio["processado_em"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    df_limpo.to_csv(ULTIMO_CSV, index=False)
    ULTIMO_RELATORIO.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")


def processar(conteudo, origem):
    """Valida, limpa e salva. Retorna lista de erros (vazia se deu certo)."""
    df, erros = dados.ler_csv(conteudo)
    if erros:
        return erros
    df_limpo, relatorio = dados.limpar(df)
    if len(df_limpo) < dados.MINIMO_REGISTROS:
        sobra = len(df_limpo)
        return [f"Depois da limpeza {'sobrou só 1 registro válido' if sobra == 1 else f'sobraram só {sobra} registros válidos'} "
                f"(mínimo: {dados.MINIMO_REGISTROS}). Confira os valores de área, quartos, ano e preço."]
    salvar_resultado(df_limpo, relatorio, origem, conteudo_original=conteudo)
    return []


@app.route("/")
def index():
    return render_template("index.html", erros=[])


@app.route("/upload", methods=["POST"])
def upload():
    arquivo = request.files.get("arquivo")
    if arquivo is None or arquivo.filename == "":
        erros = ["Nenhum arquivo selecionado. Escolha um CSV para enviar."]
    elif not arquivo.filename.lower().endswith(".csv"):
        erros = ["Formato inválido: envie um arquivo com extensão .csv."]
    else:
        erros = processar(arquivo.read(), arquivo.filename)

    if erros:
        return render_template("index.html", erros=erros), 400
    flash(f"Arquivo '{arquivo.filename}' enviado e analisado.", "ok")
    return redirect(url_for("index"))  # no passo 4 trocar por url_for("analise")


@app.route("/usar-exemplo", methods=["POST"])
def usar_exemplo():
    erros = processar(dados.CAMINHO_EXEMPLO.read_bytes(), dados.CAMINHO_EXEMPLO.name)
    if erros:
        return render_template("index.html", erros=erros), 400
    flash("CSV de exemplo carregado.", "ok")
    return redirect(url_for("index"))  # no passo 4 trocar por url_for("analise")


if __name__ == "__main__":
    app.run(debug=True)
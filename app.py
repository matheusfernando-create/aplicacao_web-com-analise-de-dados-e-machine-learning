import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

import dados
import graficos

BASE = Path(__file__).parent
PASTA_UPLOADS = BASE / "uploads"
ULTIMO_CSV = PASTA_UPLOADS / "ultimo_limpo.csv"            
ULTIMO_RELATORIO = PASTA_UPLOADS / "ultimo_relatorio.json"  
PASTA_GRAFICOS = PASTA_UPLOADS / "graficos"                    

CAMINHO_MODELO = BASE / "models" / "modelo.joblib"
CAMINHO_METRICAS = BASE / "models" / "metricas.json"

LIMITES = {
    "area_habitavel": (30, 1000),
    "quartos": (1, 10),
    "ano_construcao": (1800, datetime.now().year),
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024 
app.secret_key = os.environ.get("SECRET_KEY", "chave-de-desenvolvimento")  

app.jinja_env.filters["moeda"] = dados.formatar_moeda
app.jinja_env.filters["numero"] = dados.formatar_numero


@app.context_processor
def estado_da_sessao():
    """Variáveis disponíveis em todos os templates (usadas na barra de etapas)."""
    return {"tem_dados": ULTIMO_CSV.exists()}

# Upload: guardar e processar o CSV

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
    graficos.gerar(df_limpo, PASTA_GRAFICOS)  # gera os gráficos uma vez, no upload
    return []

# Modelo: carregar o que o treinar.py gerou e fazer a previsão

_cache = {"data_arquivo": None, "pipeline": None}


def carregar_info():
    """Lê models/metricas.json. Retorna None se o treinar.py ainda não foi executado."""
    if not (CAMINHO_MODELO.exists() and CAMINHO_METRICAS.exists()):
        return None
    return json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))


def carregar_pipeline():
    """Carrega o pipeline salvo (pré-processamento + modelo). Recarrega se o arquivo mudar."""
    data_arquivo = CAMINHO_MODELO.stat().st_mtime
    if _cache["pipeline"] is None or _cache["data_arquivo"] != data_arquivo:
        _cache["pipeline"] = joblib.load(CAMINHO_MODELO)
        _cache["data_arquivo"] = data_arquivo
    return _cache["pipeline"]


def prever(valores, colunas_x):
    """Monta uma linha com as MESMAS colunas e ordem do treino e pede o preço ao modelo."""
    linha = pd.DataFrame([valores])[colunas_x]
    return max(float(carregar_pipeline().predict(linha)[0]), 0.0)


def resumir_modelo(info):
    """Frase explicando a escolha do modelo, calculada a partir do metricas.json."""
    resultados = info["resultados"]
    escolhido = info["modelo_escolhido"]
    mae_teste = resultados[escolhido]["teste"]["mae"]
    mae_treino = resultados[escolhido]["treino"]["mae"]
    texto = (f"{escolhido} teve o menor MAE no conjunto de teste ({dados.formatar_moeda(mae_teste)}) "
             "e foi escolhido para a aplicação. ")
    baseline = [v["teste"]["mae"] for nome, v in resultados.items() if nome.startswith("Baseline")]
    if baseline:
        melhora = (1 - mae_teste / baseline[0]) * 100
        texto += (f"Ele erra {dados.formatar_numero(melhora)}% menos que o baseline, que sempre prevê a mediana "
                  f"({dados.formatar_moeda(baseline[0])}). ")
    texto += (f"A diferença entre o erro no teste e no treino é de {dados.formatar_moeda(abs(mae_teste - mae_treino))}: "
              "quanto menor, menos o modelo decorou o treino (overfitting).")
    return texto


def validar_formulario(form, info):
    """Converte e valida os campos. Retorna (valores, erros, avisos)."""
    erros, avisos, valores = [], [], {}
    rotulos = {"area_habitavel": "Área habitável", "quartos": "Quartos", "ano_construcao": "Ano de construção"}

    for campo, rotulo in rotulos.items():
        bruto = (form.get(campo) or "").strip().replace(",", ".")
        minimo, maximo = LIMITES[campo]
        if not bruto:
            erros.append(f"{rotulo}: preencha o campo.")
            continue
        try:
            numero = float(bruto)
        except ValueError:
            erros.append(f"{rotulo}: digite apenas números.")
            continue
        if campo != "area_habitavel" and numero != int(numero):
            erros.append(f"{rotulo}: use um número inteiro.")
            continue
        if not minimo <= numero <= maximo:
            erros.append(f"{rotulo}: informe um valor entre {minimo} e {maximo}.")
            continue
        valores[campo] = numero if campo == "area_habitavel" else int(numero)
        faixa = info["faixas_validas"][campo]
        if not faixa["min"] <= numero <= faixa["max"]:
            avisos.append(f"{rotulo} fora da faixa vista no treino ({dados.formatar_numero(faixa['min'], 0, False)} a "
                          f"{dados.formatar_numero(faixa['max'], 0, False)}): a previsão é menos confiável.")

    bairro = (form.get("bairro") or "").strip()
    por_nome = {b.lower(): b for b in info["bairros"]}
    if not bairro:
        erros.append("Bairro: escolha um bairro da lista.")
    elif bairro.lower() not in por_nome:
        erros.append("Bairro: não encontramos esse bairro. Escolha um da lista de sugestões.")
    else:
        valores["bairro"] = por_nome[bairro.lower()]

    return valores, erros, avisos

# Rotas

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
    return redirect(url_for("analise"))


@app.route("/usar-exemplo", methods=["POST"])
def usar_exemplo():
    erros = processar(dados.CAMINHO_EXEMPLO.read_bytes(), dados.CAMINHO_EXEMPLO.name)
    if erros:
        return render_template("index.html", erros=erros), 400
    flash("CSV de exemplo carregado.", "ok")
    return redirect(url_for("analise"))


@app.route("/analise")
def analise():
    if not ULTIMO_CSV.exists():
        flash("Envie um CSV (ou use o de exemplo) para ver a análise.", "aviso")
        return redirect(url_for("index"))

    df = pd.read_csv(ULTIMO_CSV)
    relatorio = json.loads(ULTIMO_RELATORIO.read_text(encoding="utf-8"))
    resultado = dados.analisar(df)
    imagens = graficos.carregar(PASTA_GRAFICOS)
    if not imagens:  
        graficos.gerar(df, PASTA_GRAFICOS)
        imagens = graficos.carregar(PASTA_GRAFICOS)
    return render_template(
        "analise.html", relatorio=relatorio, amostra=df.head(10).to_dict("records"),
        colunas=list(df.columns), resultado=resultado, imagens=imagens,
    )


@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    info = carregar_info()
    valores_form, erros, avisos, previsto = {}, [], [], None

    if info is not None and request.method == "POST":
        valores_form = request.form.to_dict()
        valores, erros, avisos = validar_formulario(request.form, info)
        if not erros:
            try:
                previsto = round(prever(valores, info["colunas_x"]), -3)  
            except Exception:  
                erros.append("Não foi possível usar o modelo salvo. Rode python treinar.py de novo e reinicie o servidor.")

    return render_template(
        "previsao.html", info=info, form=valores_form, erros=erros, avisos=avisos, previsto=previsto,
        limites=LIMITES, resumo=resumir_modelo(info) if info else None,
        versao_diferente=bool(info) and info.get("versao_scikit_learn") != sklearn.__version__,
        versao_atual=sklearn.__version__,
    )


@app.errorhandler(413)
def arquivo_grande(_erro):
    flash("O arquivo é maior que 20 MB. Envie um arquivo menor.", "aviso")
    return redirect(url_for("index"))


@app.errorhandler(404)
def nao_encontrada(_erro):
    flash("Página não encontrada.", "aviso")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
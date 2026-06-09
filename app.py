# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- NOVO
import numpy as np
from scipy.optimize import minimize
import jsonfrom flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from scipy.optimize import minimize
import json
import io
import os

app = Flask(__name__)
CORS(app)

# ---------- FUNÇÕES GEOQUÍMICAS (existentes) ----------
def recalcular_FeOtotal(FeOtotal):
    return FeOtotal / 1.17996, FeOtotal * 0.1695

def recalcular_Fe2O3total(Fe2O3total):
    return Fe2O3total / 1.31, Fe2O3total * 0.1515

def validar(comp):
    base = ['SiO2','TiO2','Al2O3','MnO','MgO','CaO','Na2O','K2O','P2O5']
    for x in base:
        if x not in comp:
            raise ValueError(f"Falta {x}")
    if "Fe_total" in comp and "FeO" in comp:
        raise ValueError("Use Fe_total OU FeO+Fe2O3")
    if "Fe_total" not in comp and "FeO" not in comp:
        raise ValueError("Ferro não informado")

def converter(comp, tipo):
    validar(comp)
    base = [comp['SiO2'], comp['TiO2'], comp['Al2O3']]
    if "FeO" in comp:
        FeO = comp['FeO']
        Fe2O3 = comp['Fe2O3']
    else:
        if tipo == "Fe2O3total":
            FeO, Fe2O3 = recalcular_Fe2O3total(comp['Fe_total'])
        else:
            FeO, Fe2O3 = recalcular_FeOtotal(comp['Fe_total'])
    resto = [comp['MnO'], comp['MgO'], comp['CaO'],
             comp['Na2O'], comp['K2O'], comp['P2O5']]
    return np.array(base + [FeO, Fe2O3] + resto)

def residuo(x, Cp, Cf, M):
    F = x[0]
    w = x[1:]
    if F < 0 or F > 1 or np.any(w < 0):
        return 1e6
    E = w @ M
    modelo = F*Cf + (1-F)*E
    return np.sum((Cp - modelo)**2)

def otimizar(Cp, Cf, M, tau):
    n = M.shape[0]
    N_MC = 20000
    N_TOP = 30
    F_rand = np.random.rand(N_MC)
    W_rand = np.random.dirichlet(np.ones(n), size=N_MC)
    E = W_rand @ M
    modelo = F_rand[:,None]*Cf + (1-F_rand[:,None])*E
    erros = np.sum((modelo - Cp)**2, axis=1)
    melhores = np.argsort(erros)[:N_TOP]
    sol = []
    for i in melhores:
        x0 = np.concatenate([[F_rand[i]], W_rand[i]])
        bounds = [(0,1)] + [(0,1)]*n
        cons = [{'type':'eq', 'fun': lambda x: np.sum(x[1:]) - 1}]
        r = minimize(residuo, x0, args=(Cp, Cf, M),
                     method='SLSQP', bounds=bounds, constraints=cons)
        if r.success and r.fun <= tau:
            sol.append({'F': r.x[0], 'w': r.x[1:].tolist(), 'erro': r.fun})
    sol.sort(key=lambda s: s['erro'])
    return sol

# ---------- FUNÇÕES PARA GERAR MARKDOWN ----------
ox = ['SiO2','TiO2','Al2O3','FeO','Fe2O3',
      'MnO','MgO','CaO','Na2O','K2O','P2O5']

def analise_erro(Cp, modelo):
    diff = Cp - modelo
    pct = diff / Cp * 100
    SS_res = np.sum((Cp - modelo)**2)
    SS_tot = np.sum((Cp - np.mean(Cp))**2)
    R2 = 1 - SS_res/SS_tot if SS_tot != 0 else 0
    return diff, pct, R2

def tabela(Cp, Cf, M, modelo, nomes):
    diff, pct, R2 = analise_erro(Cp, modelo)
    header = "| Tipo | " + " | ".join(ox) + " |\n"
    header += "|------|" + "|".join(["------"]*len(ox)) + "|\n"
    def row(nome, vals):
        return "| " + nome + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |\n"
    txt = header
    txt += row("Parental", Cp)
    txt += row("Evoluído", Cf)
    txt += row("Modelado", modelo)
    txt += row("Δ (Par-Model)", diff)
    txt += row("Erro (%)", pct)
    for n, comp in zip(nomes, M):
        txt += row(n, comp)
    txt += f"\n**R² ajustado:** {R2:.4f}\n\n"
    return txt

def gerar_markdown(dados, tau):
    md = "# RESULTADOS DO BALANÇO DE MASSA\n\n"
    md += f"Tau utilizado: {tau}\n\n"
    for modelo in dados['modelos']:
        nome = modelo['nome']
        md += f"## {nome}\n\n"
        Cp = converter(modelo['parental'], dados.get('tipo_ferro', 'FeOtotal'))
        Cf = converter(modelo['filho'], dados.get('tipo_ferro', 'FeOtotal'))
        nomes = [m['nome'] for m in modelo['minerais']]
        M = np.array([converter(m, dados.get('tipo_ferro', 'FeOtotal')) for m in modelo['minerais']])
        sol = otimizar(Cp, Cf, M, tau)
        if not sol:
            md += "❌ **Sem solução** (nenhuma combinação atingiu o erro ≤ tau)\n\n"
            continue
        best = sol[0]
        md += f"**F (fração líquido remanescente):** {best['F']:.4f}\n\n"
        md += f"**Erro:** {best['erro']:.6f}\n\n"
        md += "### Assembleia mineral (fração cristalizada)\n"
        for n, w in zip(nomes, best['w']):
            md += f"- **{n}:** {w*100:.2f}%\n"
        w = best['w']
        F = best['F']
        E = np.array(w) @ M
        modelo_calc = F*Cf + (1-F)*E
        md += "\n### Composição química modelada\n\n"
        md += tabela(Cp, Cf, M, modelo_calc, nomes)
        md += "\n---\n\n"
    return md

# ---------- ENDPOINTS ----------
@app.route('/balanco', methods=['POST'])
def balanco():
    try:
        dados = request.get_json()
        tau = dados['tau']
        tipo = dados.get('tipo_ferro', 'FeOtotal')
        resultados = []
        for modelo in dados['modelos']:
            nome = modelo['nome']
            Cp = converter(modelo['parental'], tipo)
            Cf = converter(modelo['filho'], tipo)
            M = np.array([converter(m, tipo) for m in modelo['minerais']])
            sol = otimizar(Cp, Cf, M, tau)
            if sol:
                best = sol[0]
                resultados.append({
                    'modelo': nome,
                    'F': best['F'],
                    'erro': best['erro'],
                    'assembleia': [
                        {'mineral': modelo['minerais'][i]['nome'],
                         'fracao': best['w'][i]}
                        for i in range(len(best['w']))
                    ]
                })
            else:
                resultados.append({'modelo': nome, 'F': None, 'erro': None, 'assembleia': []})
        return jsonify({'status': 'ok', 'resultados': resultados})
    except Exception as e:
        return jsonify({'status': 'erro', 'mensagem': str(e)}), 400

@app.route('/relatorio', methods=['POST'])
def relatorio():
    """Retorna um arquivo .md para download"""
    try:
        dados = request.get_json()
        tau = dados['tau']
        markdown_content = gerar_markdown(dados, tau)
        return send_file(
            io.BytesIO(markdown_content.encode('utf-8')),
            mimetype='text/markdown',
            as_attachment=True,
            download_name='resultado_balanco.md'
        )
    except Exception as e:
        return jsonify({'status': 'erro', 'mensagem': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
import os  # <-- NOVO (para a porta)

app = Flask(__name__)
CORS(app)  # <-- NOVO: permite requisições de qualquer origem



# ---------- FUNÇÕES DO SEU PROGRAMA (COPIADAS INTEGRALMENTE) ----------
def recalcular_FeOtotal(FeOtotal):
    return FeOtotal / 1.17996, FeOtotal * 0.1695

def recalcular_Fe2O3total(Fe2O3total):
    return Fe2O3total / 1.31, Fe2O3total * 0.1515

def validar(comp):
    base = ['SiO2','TiO2','Al2O3','MnO','MgO','CaO','Na2O','K2O','P2O5']
    for x in base:
        if x not in comp:
            raise ValueError(f"Falta {x}")
    if "Fe_total" in comp and "FeO" in comp:
        raise ValueError("Use Fe_total OU FeO+Fe2O3")
    if "Fe_total" not in comp and "FeO" not in comp:
        raise ValueError("Ferro não informado")

def converter(comp, tipo):
    validar(comp)
    base = [comp['SiO2'], comp['TiO2'], comp['Al2O3']]
    if "FeO" in comp:
        FeO = comp['FeO']
        Fe2O3 = comp['Fe2O3']
    else:
        if tipo == "Fe2O3total":
            FeO, Fe2O3 = recalcular_Fe2O3total(comp['Fe_total'])
        else:
            FeO, Fe2O3 = recalcular_FeOtotal(comp['Fe_total'])
    resto = [comp['MnO'], comp['MgO'], comp['CaO'],
             comp['Na2O'], comp['K2O'], comp['P2O5']]
    return np.array(base + [FeO, Fe2O3] + resto)

def residuo(x, Cp, Cf, M):
    F = x[0]
    w = x[1:]
    if F < 0 or F > 1 or np.any(w < 0):
        return 1e6
    E = w @ M
    modelo = F*Cf + (1-F)*E
    return np.sum((Cp - modelo)**2)

def otimizar(Cp, Cf, M, tau):
    n = M.shape[0]
    N_MC = 20000
    N_TOP = 30
    F_rand = np.random.rand(N_MC)
    W_rand = np.random.dirichlet(np.ones(n), size=N_MC)
    E = W_rand @ M
    modelo = F_rand[:,None]*Cf + (1-F_rand[:,None])*E
    erros = np.sum((modelo - Cp)**2, axis=1)
    melhores = np.argsort(erros)[:N_TOP]
    sol = []
    for i in melhores:
        x0 = np.concatenate([[F_rand[i]], W_rand[i]])
        bounds = [(0,1)] + [(0,1)]*n
        cons = [{'type':'eq', 'fun': lambda x: np.sum(x[1:]) - 1}]
        r = minimize(residuo, x0, args=(Cp, Cf, M),
                     method='SLSQP', bounds=bounds, constraints=cons)
        if r.success and r.fun <= tau:
            sol.append({'F': r.x[0], 'w': r.x[1:].tolist(), 'erro': r.fun})
    sol.sort(key=lambda s: s['erro'])
    return sol

# ---------------------------------------------------------------

@app.route('/balanco', methods=['POST'])
def balanco():
    try:
        dados = request.get_json()
        tau = dados['tau']
        tipo = dados.get('tipo_ferro', 'FeOtotal')
        resultados = []
        for modelo in dados['modelos']:
            nome = modelo['nome']
            Cp = converter(modelo['parental'], tipo)
            Cf = converter(modelo['filho'], tipo)
            M = np.array([converter(m, tipo) for m in modelo['minerais']])
            sol = otimizar(Cp, Cf, M, tau)
            if sol:
                best = sol[0]
                resultados.append({
                    'modelo': nome,
                    'F': best['F'],
                    'erro': best['erro'],
                    'assembleia': [
                        {'mineral': modelo['minerais'][i]['nome'],
                         'fracao': best['w'][i]}
                        for i in range(len(best['w']))
                    ]
                })
            else:
                resultados.append({'modelo': nome, 'F': None, 'erro': None, 'assembleia': []})
        return jsonify({'status': 'ok', 'resultados': resultados})
    except Exception as e:
        return jsonify({'status': 'erro', 'mensagem': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- novo import
import numpy as np
from scipy.optimize import minimize
import json

app = Flask(__name__)
CORS(app)  # <-- permite qualquer origem (para testes)


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

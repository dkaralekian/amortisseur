import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ==============================================================================
# CONFIGURATION DE LA PAGE ET GESTION DES DONNÉES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Calculateur Principal")
st.title("Dimensionnement de l'amortisseur de traînée")

DB_FILE = "helicopters.json"
SESSION_FILE = "sessions.json"

def load_helicopter_db():
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'G5': {'Omega_rpm': 404, 'ms': 40.0, 'Ip': 125.0, 'e': 0.166, 'b': 4, 'f_fuselage': 2.5,
                   'coords': {'B': "0.166,0.003,0", 'P': "0.372,0.003,0", 'M': "0.141,0.147,0", 'A_local': "0.203,0.079,0"}},
            'G2': {'Omega_rpm': 530, 'ms': 23.6, 'Ip': 56.2, 'e': 0.154, 'b': 3, 'f_fuselage': 2.2,
                   'material': {'G_mpa': 1.2, 'phi_deg': 22.0}, 'elastomer': {'L': 120.0, 'd_int': 24.0, 'ep': 10.0},
                   'coords': {'B': "0.154,0,0", 'P': "0.350,0,0", 'M': "0.130,0.130,0", 'A_local': "0.190,0.060,0"}},
            'EC120': {'Omega_rpm': 408, 'ms': 35.0, 'Ip': 110.0, 'e': 0.160, 'b': 3, 'f_fuselage': 2.0,
                      'material': {'G_mpa': 1.3, 'phi_deg': 28.0}, 'elastomer': {'L': 100.0, 'd_int': 28.0, 'ep': 12.0},
                      'coords': {'B': "0.160,0.01,0", 'P': "0.360,0.01,0", 'M': "0.135,0.140,0", 'A_local': "0.195,0.070,0"}},
        }

HELICOPTER_PARAMS = load_helicopter_db()

def load_sessions():
    try:
        with open(SESSION_FILE, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return {}
def save_session(state_dict):
    sessions = load_sessions()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sessions[timestamp] = state_dict
    with open(SESSION_FILE, 'w') as f: json.dump(sessions, f, indent=4)
    return timestamp
STATE_KEYS = ['selected_heli', 'Omega_rpm', 'ms', 'Ip', 'e', 'b', 'phi_deg', 'omega_delta_bar', 'f_fuselage','B_coords', 'P_coords', 'M_coords', 'A_local_coords', 'alpha_deg','G_mpa', 'tau_adm_mpa', 'L_etude']
if 'initialized' not in st.session_state:
    default_heli = 'G5'
    st.session_state.selected_heli = default_heli
    st.session_state.Omega_rpm = HELICOPTER_PARAMS[default_heli]['Omega_rpm']
    st.session_state.ms = HELICOPTER_PARAMS[default_heli]['ms']
    st.session_state.Ip = HELICOPTER_PARAMS[default_heli]['Ip']
    st.session_state.e = HELICOPTER_PARAMS[default_heli]['e']
    st.session_state.b = HELICOPTER_PARAMS[default_heli]['b']
    st.session_state.phi_deg = 25.0
    st.session_state.omega_delta_bar = 0.4
    st.session_state.f_fuselage = HELICOPTER_PARAMS[default_heli]['f_fuselage']
    st.session_state.B_coords = HELICOPTER_PARAMS[default_heli]['coords']['B']
    st.session_state.P_coords = HELICOPTER_PARAMS[default_heli]['coords']['P']
    st.session_state.M_coords = HELICOPTER_PARAMS[default_heli]['coords']['M']
    st.session_state.A_local_coords = HELICOPTER_PARAMS[default_heli]['coords']['A_local']
    st.session_state.alpha_deg = 3.0
    st.session_state.G_mpa = 1.29
    st.session_state.tau_adm_mpa = 2.0
    st.session_state.L_etude = 80.0
    st.session_state.initialized = True

def show_comparison_metric(label, values_dict, unit, main_item):
    st.markdown(f"**{label}**")
    cols = st.columns(len(values_dict))
    for i, (heli_name, value) in enumerate(values_dict.items()):
        with cols[i]:
            if heli_name == main_item:
                with st.container(border=True): st.metric(label=f"**{heli_name}**", value=f"{value} {unit}")
            else: st.metric(label=heli_name, value=f"{value} {unit}")

def plot_kg_diagram(rotor_params, selected_phi_deg, operating_point, heli_name):
    e, Omega, Ip, ms = rotor_params
    op_point_omega_bar, op_point_sigma_bar = operating_point
    omega_delta_bar_grid, sigma_bar_grid = np.linspace(0.1, 1.2, 200), np.linspace(0, 0.10, 200)
    X, Y = np.meshgrid(omega_delta_bar_grid, sigma_bar_grid)
    C = e * ms / Ip
    valid_region = X**2 - C > 0
    K_delta = np.full_like(X, np.nan)
    K_delta[valid_region] = (Ip * Omega**2) * (X[valid_region]**2 - C)
    tan_phi_grid = np.full_like(X, np.nan)
    tan_phi_grid[valid_region] = (2 * Y[valid_region] * X[valid_region]) / (X[valid_region]**2 - C + 1e-9)
    Phi = np.degrees(np.arctan(tan_phi_grid))
    fig, ax = plt.subplots(figsize=(10, 7))
    omega_ticks = np.arange(0.2, 0.51, 0.05)
    ax.set_xticks(omega_ticks)
    valid_ticks = omega_ticks[omega_ticks**2 > C]
    kd_levels = (Ip * Omega**2) * (valid_ticks**2 - C)
    ax.contour(X, Y*100, K_delta, levels=kd_levels, cmap='viridis', alpha=0.8).clabel(inline=True, fontsize=9, fmt='Kδ=%.0f')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.contour(X, Y*100, Phi, levels=[15, 20, 25, 30, 35], cmap='plasma', linestyles='dashed').clabel(inline=True, fontsize=10, fmt='φ=%.0f°')
    tan_phi_selected = np.tan(np.radians(selected_phi_deg))
    sigma_bar_curve = (tan_phi_selected * (omega_delta_bar_grid**2 - C)) / (2 * omega_delta_bar_grid)
    sigma_bar_curve[omega_delta_bar_grid**2 <= C] = np.nan
    ax.plot(omega_delta_bar_grid, sigma_bar_curve * 100, 'r-', linewidth=3, label=f"Courbe pour φ={selected_phi_deg}°")
    ax.plot(op_point_omega_bar, op_point_sigma_bar * 100, 'kx', markersize=12, markeredgewidth=3, label="Point de fonctionnement")
    ax.set_xlabel("Fréquence propre relative, $\\bar{\\omega}_\\delta = \\omega_\\delta/\\Omega$", fontsize=12)
    ax.set_ylabel("Amortissement relatif, $\\bar{\\sigma}$ [%]", fontsize=12)
    ax.set_title(f"Diagramme K-G pour {heli_name}", fontsize=14)
    ax.legend()
    ax.set_xlim(left=0.2, right=0.5)
    ax.set_ylim(bottom=0, top=10)
    return fig

def plot_geometry(B, P, A_zero, Apt, M):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(A_zero[0], A_zero[1], 'k+', markersize=10, label='A à 0° (ref)')
    ax.plot([A_zero[0], M[0]], [A_zero[1], M[1]], 'k--', alpha=0.7, label='Amortisseur à 0°')
    ax.plot(Apt[0], Apt[1], 'bo', markersize=10, label='A pré-calé (actuel)')
    ax.plot([Apt[0], M[0]], [Apt[1], M[1]], 'b-', linewidth=2, label='Amortisseur pré-calé')
    ax.plot(B[0], B[1], 'ro', markersize=10, label='B: Butée Sphérique (Pivot)')
    ax.plot(M[0], M[1], 'gs', markersize=10, label='M: Ferrure Moyeu')
    ax.plot([B[0], P[0]], [B[1], P[1]], 'k-', label='Axe de la pale')
    v, w = M - Apt, B - Apt
    t = np.dot(w, v) / np.dot(v, v)
    proj_point = Apt + t * v
    if 0 <= t <= 1: ax.plot([B[0], proj_point[0]], [B[1], proj_point[1]], 'r--', label='ρ (Bras de levier)')
    ax.set_xlabel("Axe X (m)"); ax.set_ylabel("Axe Y (m)"); ax.set_title("Visualisation Géométrique 2D")
    ax.grid(True); ax.axis('equal'); ax.legend(fontsize='small')
    return fig

st.sidebar.title("Configuration de la Simulation")
demo_mode = st.sidebar.toggle("Afficher les démonstrations", value=True)
st.sidebar.markdown("---")
def update_heli_params():
    params = HELICOPTER_PARAMS[st.session_state.selected_heli_selector]
    st.session_state.Omega_rpm = params['Omega_rpm']
    st.session_state.ms = params['ms']
    st.session_state.Ip = params['Ip']
    st.session_state.e = params['e']
    st.session_state.b = params['b']
    st.session_state.f_fuselage = params.get('f_fuselage', 2.5)
    st.session_state.B_coords = params['coords']['B']
    st.session_state.P_coords = params['coords']['P']
    st.session_state.M_coords = params['coords']['M']
    st.session_state.A_local_coords = params['coords']['A_local']
st.sidebar.selectbox("1. Choisir l'hélicoptère de travail", list(HELICOPTER_PARAMS.keys()), key='selected_heli_selector', on_change=update_heli_params)
st.sidebar.subheader("2. Paramètres Rotor (éditables)")
c1, c2 = st.sidebar.columns(2)
with c1:
    st.number_input("$\\Omega$ (tr/min)", key='Omega_rpm')
    st.number_input("$m_s$ (m.kg)", key='ms')
    st.number_input("$I_p$ (m².kg)", key='Ip')
with c2:
    st.number_input("$e$ (m)", key='e')
    st.number_input("$b$ (pales)", key='b')
Omega_rad_s = -st.session_state.Omega_rpm * np.pi / 30
sessions = load_sessions()
session_list = ["-"] + sorted(sessions.keys(), reverse=True)
def load_selected_session():
    session_key = st.session_state.session_selector
    if session_key != "-":
        loaded_state = sessions[session_key]
        for key, value in loaded_state.items():
            if key in st.session_state: st.session_state[key] = value
st.sidebar.selectbox("3. Charger une session", session_list, key="session_selector", on_change=load_selected_session)

st.header("Phase 1: Choix du Point de Fonctionnement")
col1, col2, col3 = st.columns([1.5, 2, 2.5])
with col1:
    st.subheader("1.1: Paramètres de Conception")
    phi_deg_design = st.slider("Angle de perte du design, $\\phi$ (°)", 15.0, 35.0, key='phi_deg')
    omega_delta_bar_design = st.slider("Fréquence propre du design, $\\bar{\\omega}_\\delta$", 0.2, 0.8, step=0.005, key='omega_delta_bar')
    f_fuselage_design = st.number_input("Fréquence fuselage du design, $f_{fus}$ (Hz)", key='f_fuselage')

# --- NOUVELLE LOGIQUE DE CALCUL DÉCOUPLÉE ---
omega_cs, k_deltas, k2_deltas, beat_freqs, sigma_bars = {}, {}, {}, {}, {}
for name, params in HELICOPTER_PARAMS.items():
    heli_omega_rad = abs(params['Omega_rpm'] * np.pi / 30)
    heli_C = params['e'] * params['ms'] / params['Ip']
    
    if name == st.session_state.selected_heli: # Pour le G5, on utilise les sliders
        omega_delta_bar_heli = st.session_state.omega_delta_bar
        tan_phi_heli = np.tan(np.radians(st.session_state.phi_deg))
        k_deltas[name] = (params['Ip'] * (heli_omega_rad**2)) * (omega_delta_bar_heli**2 - heli_C)
    else: # Pour G2/EC120, on utilise leurs data physiques
        L, d_int, ep = params['elastomer']['L'], params['elastomer']['d_int'], params['elastomer']['ep']
        G = params['material']['G_mpa']
        D_ext = d_int + 2 * ep
        K1_nm = (2 * np.pi * G * L) / np.log(D_ext / d_int) * 1000
        h_coords = params['coords']
        h_B, h_M, h_A_local = (np.array([float(x.strip()) for x in h_coords[c].split(',')]) for c in ['B','M','A_local'])
        alpha_rad_default = np.radians(3.0)
        h_Apt = (np.array([[np.cos(alpha_rad_default),-np.sin(alpha_rad_default),0],[np.sin(alpha_rad_default),np.cos(alpha_rad_default),0],[0,0,1]]) @ h_A_local) + h_B
        rho_h = np.linalg.norm(np.cross(h_Apt - h_B, h_M - h_Apt)) / np.linalg.norm(h_M - h_Apt)
        k_deltas[name] = K1_nm * rho_h**2
        omega_delta_bar_heli = np.sqrt(k_deltas[name] / (params['Ip'] * heli_omega_rad**2) + heli_C)
        tan_phi_heli = np.tan(np.radians(params['material']['phi_deg']))

    # Calculs communs
    sigma_bars[name] = (tan_phi_heli * (omega_delta_bar_heli**2 - heli_C)) / (2 * omega_delta_bar_heli)
    beat_freqs[name] = abs(heli_omega_rad * (1 - omega_delta_bar_heli)) / (2 * np.pi)
    f_delta_h = (omega_delta_bar_heli * heli_omega_rad) / (2*np.pi)
    f_nominal_h = heli_omega_rad / (2*np.pi)
    heli_f_fus = params.get('f_fuselage', st.session_state.f_fuselage)
    omega_cs[name] = (heli_f_fus + f_delta_h) / f_nominal_h * 100

with col2:
    st.subheader("1.2: Comparaison des Résultats")
    show_comparison_metric("Fréquence de battement $|f_n - f_\\delta|$", {k: f"{v:.2f}" for k, v in beat_freqs.items()}, "Hz", st.session_state.selected_heli)
    st.markdown("---")
    show_comparison_metric("Fréquence de croisement $\\Omega_c$", {k: f"{v:.1f}%" for k, v in omega_cs.items()}, "", st.session_state.selected_heli)
    st.markdown("---")
    show_comparison_metric("Amortissement réduit $\\bar{\\sigma}$", {k: f"{v:.2%}" for k, v in sigma_bars.items()}, "", st.session_state.selected_heli)

with st.expander("Formules de Comparaison"):
    st.latex(r"K_{1,ref} = \frac{2 \pi G_{ref} L_{ref}}{\ln(D_{ext,ref} / d_{int,ref})} \implies K_{\delta, ref} = K_{1,ref} \cdot \rho_{ref}^2")
    st.latex(r"\bar{\omega}_{\delta, ref} = \sqrt{\frac{K_{\delta, ref}}{I_{p, ref} \Omega_{ref}^2} + C_{phys, ref}}")
    st.latex(r"|f_n - f_\delta| = \frac{|\Omega(1 - \bar{\omega}_\delta)|}{2\pi} \quad ; \quad \Omega_c = \frac{f_{fus} + (\bar{\omega}_\delta \Omega / 2\pi)}{(\Omega/2\pi)} \quad ; \quad \bar{\sigma} = \frac{\tan(\phi) (\bar{\omega}_\delta^2 - C_{phys})}{2 \bar{\omega}_\delta}")

with col3:
    st.subheader("1.3: Diagramme K-G")
    fig_kg = plot_kg_diagram((st.session_state.e, abs(Omega_rad_s), st.session_state.Ip, st.session_state.ms), st.session_state.phi_deg, (st.session_state.omega_delta_bar, sigma_bars[st.session_state.selected_heli]), st.session_state.selected_heli)
    st.pyplot(fig_kg)
st.markdown("---")

# ... (Le reste du code, Phases 2, 3, etc. est complet et identique à la version précédente)
st.header("Phase 2: Définition Géométrique")
col_geo1, col_geo2 = st.columns(2)
with col_geo1:
    st.subheader("2.1: Définition des Points")
    B_coords = st.text_input("B (global)", key='B_coords')
    P_coords = st.text_input("P (global)", key='P_coords')
    M_coords = st.text_input("M (global)", key='M_coords')
    A_local_coords = st.text_input("A (local/B)", key='A_local_coords')
    alpha_deg = st.slider("Angle de pré-traînée, $\\alpha$ (°)", -5.0, 5.0, key='alpha_deg')
    try:
        B, P, M, A_local = (np.array([float(x.strip()) for x in st.session_state[c].split(',')]) for c in ['B_coords', 'P_coords', 'M_coords', 'A_local_coords'])
    except:
        st.error("Format de coordonnées invalide."); st.stop()
    alpha_rad = np.radians(st.session_state.alpha_deg)
    Apt = (np.array([[np.cos(alpha_rad),-np.sin(alpha_rad),0],[np.sin(alpha_rad),np.cos(alpha_rad),0],[0,0,1]]) @ A_local) + B

    st.markdown("##### Comparaison Géométrique")
    rhos, k1s = {}, {}
    for name, params in HELICOPTER_PARAMS.items():
        h_coords = params['coords']
        h_B, h_M, h_A_local = (np.array([float(x.strip()) for x in h_coords[c].split(',')]) for c in ['B','M','A_local'])
        # On utilise l'alpha du slider pour une comparaison "toutes choses égales par ailleurs"
        h_Apt = (np.array([[np.cos(alpha_rad),-np.sin(alpha_rad),0],[np.sin(alpha_rad),np.cos(alpha_rad),0],[0,0,1]]) @ h_A_local) + h_B
        rhos[name] = np.linalg.norm(np.cross(h_Apt - h_B, h_M - h_Apt)) / np.linalg.norm(h_M - h_Apt)
        k1s[name] = k_deltas[name] / rhos[name]**2 / 1000 if rhos[name] > 0 else 0
    show_comparison_metric("Bras de levier $\\rho$", {k: f"{v:.3f}" for k, v in rhos.items()}, "m", st.session_state.selected_heli)
    show_comparison_metric("Raideur Linéaire $K_1$", {k: f"{v:,.0f}" for k, v in k1s.items()}, "N/mm", st.session_state.selected_heli)

with col_geo2:
    st.subheader("2.2: Visualisation")
    fig_geo = plot_geometry(B, P, B + A_local, Apt, M)
    st.pyplot(fig_geo)
st.markdown("---")


# ==============================================================================
# PHASE 3: ANALYSE DES CAS DE VOL (RÉINTÉGRÉE)
# ==============================================================================
st.header("Phase 3: Analyse des Cas de Vol")
try:
    excel_data = pd.read_excel('simplified_spectrum_p5.xlsx')
    a0_values, a1_values = excel_data['a0'].to_numpy(), excel_data['a1'].to_numpy()
    Pu_factors = np.array([0.9, 0.9, 0.9, 1.0, 0.0, 0.9])
    if len(excel_data) == len(Pu_factors): Pu_values = Pu_factors * 250 * 1e3
    else: st.error(f"Incohérence du nombre de cas de vol."); st.stop()
    
    K_delta_current = k_deltas[st.session_state.selected_heli]
    omega_d_bar_val = np.sqrt((st.session_state.e * st.session_state.ms / st.session_state.Ip) + (K_delta_current / (st.session_state.Ip * Omega_rad_s**2)))
    theta_0_deg = Pu_values * 0.0001 - 6
    delta_0_deg = np.rad2deg((((Pu_values/(Omega_rad_s*st.session_state.b)) + (K_delta_current * alpha_rad)) / (Omega_rad_s**2*st.session_state.e*st.session_state.ms + K_delta_current)))
    delta_deg = -np.rad2deg((np.deg2rad(a0_values) * np.deg2rad(a1_values)) / (1 - omega_d_bar_val**2))
    
    results = []
    L_ref = np.linalg.norm(Apt - M)
    A_zero_global = B + A_local
    for i in range(len(excel_data)):
        a0_rad, theta_0_rad, delta_0_rad, delta_rad = map(np.radians, [a0_values[i], theta_0_deg[i], delta_0_deg[i], delta_deg[i]])
        Rx_stat = np.array([[1,0,0],[0,np.cos(a0_rad),-np.sin(a0_rad)],[0,np.sin(a0_rad),np.cos(a0_rad)]])
        Ry_stat = np.array([[np.cos(theta_0_rad),0,-np.sin(theta_0_rad)],[0,1,0],[np.sin(theta_0_rad),0,np.cos(theta_0_rad)]])
        Rz_stat = np.array([[np.cos(delta_0_rad),-np.sin(delta_0_rad),0],[np.sin(delta_0_rad),np.cos(delta_0_rad),0],[0,0,1]])
        R_stat_total = Rx_stat @ Ry_stat @ Rz_stat
        Astat = (R_stat_total @ (A_zero_global - B)) + B
        L0 = np.linalg.norm(Astat - M)
        DeltaL0_mm = (L0 - L_ref) * 1000
        Rz_dyn = np.array([[np.cos(delta_rad),-np.sin(delta_rad),0],[np.sin(delta_rad),np.cos(delta_rad),0],[0,0,1]])
        Adyn = (Rz_dyn @ (Astat - B)) + B
        L1 = np.linalg.norm(Adyn - M)
        DeltaL_mm = (L1 - L0) * 1000
        results.append({"Cas de vol": excel_data['FlightCase'][i], "δ0 (°)": delta_0_deg[i], "δ (°)": delta_deg[i], "ΔL0 (mm)": DeltaL0_mm, "ΔL (mm)": DeltaL_mm})
    results_df = pd.DataFrame(results)

    with st.expander("Démonstrations des calculs de cas de vol"):
        st.markdown("Formules générales utilisées pour chaque ligne du tableau :")
        st.latex(r"\delta_0 = \frac{P_u/(\Omega b) + K_\delta \alpha}{\Omega^2 e m_s + K_\delta} \quad ; \quad \delta = -\frac{a_0 a_1}{1 - \bar{\omega}_\delta^2}")
        st.latex(r"\Delta L_0 = \| R_{stat}(\dots) (\vec{A}_0-\vec{B}) + \vec{B} - \vec{M} \| - L_{ref}")
    
    st.dataframe(results_df.style.format({"δ0 (°)":"{:.2f}","δ (°)":"{:.2f}","ΔL0 (mm)":"{:.2f}","ΔL (mm)":"{:+.2f}"}))
    delta_L_max_mm_base = results_df['ΔL (mm)'].abs().max()

except FileNotFoundError:
    st.error("Le fichier 'simplified_spectrum_p5.xlsx' est introuvable.")
    delta_L_max_mm_base = 0.1 * rhos[st.session_state.selected_heli] * 1000
except Exception as e:
    st.error(f"Une erreur est survenue lors du calcul des cas de vol : {e}")
    delta_L_max_mm_base = 0.1 * rhos[st.session_state.selected_heli] * 1000
st.markdown("---")

# ==============================================================================
# PHASE 4: DIMENSIONNEMENT ET ANALYSE OPTIMALE
# ==============================================================================
st.header("Phase 4: Cartographie de Conception Optimale")

# --- PARAMÈTRES D'ENTRÉE (INCHANGÉS) ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("4.1: Paramètres Matériau")
    G_mpa = st.number_input("Module de cisaillement, G (MPa)", key='G_mpa')
    tau_adm_mpa = st.number_input("Contrainte admissible, τ_adm (MPa)", format="%.3f", key='tau_adm_mpa')
with c2:
    st.subheader("4.2: Plage d'étude du Diamètre")
    d_int_min, d_int_max = st.slider(
        "Plage du diamètre intérieur d_int (mm)",
        min_value=10, max_value=150, value=(20, 100))

# --- PRÉ-REQUIS POUR LE CALCUL ---
K_delta_req_base = k_deltas[st.session_state.selected_heli]
theta_max_rad = np.nan
gamma_max = np.nan

if 'results_df' in locals() and not results_df.empty:
    theta_max_deg = results_df['δ (°)'].abs().max()
    theta_max_rad = np.radians(theta_max_deg)
    if st.session_state.G_mpa > 0:
        gamma_max = st.session_state.tau_adm_mpa / st.session_state.G_mpa
    else:
        st.warning("Le module G doit être > 0.")
else:
    st.warning("Les résultats des cas de vol (Phase 3) sont requis pour le calcul.")

# --- CALCUL ET GÉNÉRATION DE LA CARTE ---
st.subheader("4.3: Graphique des Dimensions Optimales (L et ep)")

if not np.isnan(theta_max_rad) and not np.isnan(gamma_max) and gamma_max > 0:
    # 1. Création de la grille de calcul
    rho_values = np.linspace(0.1, 0.2, 50)
    d_int_values = np.linspace(d_int_min, d_int_max, 50)
    RHO, D_INT = np.meshgrid(rho_values, d_int_values)

    # 2. Calcul de l'épaisseur optimale 'ep' (en mm)
    # ep = (theta_max * rho) / gamma_max. Ne dépend que de rho.
    EP = (theta_max_rad * RHO * 1000) / gamma_max # Conversion en mm

    # 3. Calcul de la longueur optimale 'L' (en mm)
    # K1 = K_delta / rho^2
    # L = K1 * ln((d_int + 2ep)/d_int) / (2*pi*G)
    K_delta_Nmm = K_delta_req_base * 1000 # Nm -> Nmm
    
    # Terme de raideur linéaire K1 (en N/mm)
    K1 = K_delta_Nmm / (RHO**2 * 1000**2) # rho est en m, on le veut en mm^2 pour la division
    
    # Calcul de L
    # On ajoute un petit epsilon à D_INT pour éviter le log(1) si EP est nul
    LOG_TERM = np.log(1 + (2 * EP) / (D_INT + 1e-9))
    L = K1 * LOG_TERM / (2 * np.pi * G_mpa)
    
    # 4. Traçage du graphique
    fig, ax = plt.subplots(figsize=(12, 9))

    # Courbes de niveau pour la Longueur L
    contour_L = ax.contour(RHO, D_INT, L, levels=15, cmap='viridis')
    ax.clabel(contour_L, inline=True, fontsize=9, fmt='L=%.0f mm')

    # Courbes de niveau pour l'Épaisseur ep
    contour_EP = ax.contour(RHO, D_INT, EP, levels=10, cmap='plasma', linestyles='--')
    ax.clabel(contour_EP, inline=True, fontsize=9, fmt='ep=%.1f mm')

    ax.set_xlabel("Bras de levier, ρ (m)", fontsize=12)
    ax.set_ylabel("Diamètre intérieur, d_int (mm)", fontsize=12)
    ax.set_title("Cartographie de Conception Optimale\n(Lignes pleines: Longueur L, Pointillés: Épaisseur ep)", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Ajout d'une colorbar pour la longueur L pour une meilleure lecture
    cbar = fig.colorbar(contour_L, ax=ax)
    cbar.set_label('Longueur L (mm)')

    st.pyplot(fig)

    with st.expander("Comment lire ce graphique ?"):
        st.markdown("""
        Ce graphique vous aide à faire des choix de conception :
        1.  **Choisissez un point de fonctionnement :** Par exemple, vous visez un bras de levier $\rho=0.15$ m et un diamètre intérieur $d_{int}=50$ mm.
        2.  **Lisez les dimensions requises :**
            * Regardez la ligne de contour **pleine (verte/jaune)** qui passe par votre point. Elle vous donne la **longueur `L`** nécessaire (ex: L ≈ 80 mm).
            * Regardez la ligne de contour en **pointillés (rouge/violette)**. Elle vous donne l'**épaisseur `ep`** requise (ex: ep ≈ 4.5 mm).

        * **Analyse des tendances :**
            * Les lignes d'épaisseur (`ep`) sont verticales : l'épaisseur requise ne dépend que du bras de levier $\rho$.
            * Pour un $\rho$ donné, si vous augmentez le diamètre intérieur ($d_{int}$), la longueur `L` requise augmente également.
            * Pour un $d_{int}$ donné, si vous augmentez le bras de levier $\rho$, la longueur `L` requise diminue.
        """)
else:
    st.error("Calcul impossible. Vérifiez les paramètres en entrée et les résultats de la Phase 3.")

st.markdown("---")
if st.button("Enregistrer la session actuelle", use_container_width=True):
    current_state = {key: st.session_state[key] for key in STATE_KEYS}
    timestamp = save_session(current_state)
    st.success(f"Session enregistrée avec succès : {timestamp}")
    st.rerun()
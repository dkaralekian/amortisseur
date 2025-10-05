import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION DE LA PAGE ET DONNÉES DE BASE
# ==============================================================================
st.set_page_config(layout="wide", page_title="Outil Comparatif de Dimensionnement")
st.title("Dimensionnement de l'amortisseur de traînée")

HELICOPTER_PARAMS = {
    'G5': {'Omega_rpm': 404, 'ms': 40.0, 'Ip': 125.0, 'e': 0.166, 'b': 4, 
           'coords': {'B': "0.166, 0.003, 0", 'P': "0.372, 0.003, 0", 'M': "0.141, 0.147, 0", 'A_local': "0.203, 0.079, 0"}},
    'G2': {'Omega_rpm': 530, 'ms': 23.6, 'Ip': 56.2, 'e': 0.154, 'b': 3, 
           'coords': {'B': "0.154, 0, 0", 'P': "0.350, 0, 0", 'M': "0.130, 0.130, 0", 'A_local': "0.190, 0.060, 0"}},
    'EC120': {'Omega_rpm': 408, 'ms': 35.0, 'Ip': 110.0, 'e': 0.160, 'b': 3, 
              'coords': {'B': "0.160, 0.01, 0", 'P': "0.360, 0.01, 0", 'M': "0.135, 0.140, 0", 'A_local': "0.195, 0.070, 0"}},
}

def show_comparison_metric(label, values_dict, unit, main_item):
    st.markdown(f"**{label}**")
    cols = st.columns(len(values_dict))
    for i, (heli_name, value) in enumerate(values_dict.items()):
        with cols[i]:
            if heli_name == main_item:
                with st.container(border=True):
                    st.metric(label=f"**{heli_name}**", value=f"{value} {unit}")
            else:
                st.metric(label=heli_name, value=f"{value} {unit}")

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
demo_mode = st.sidebar.toggle("Afficher les démonstrations de calcul", value=True)
st.sidebar.markdown("---")
selected_heli = st.sidebar.selectbox("1. Choisir l'hélicoptère de travail", list(HELICOPTER_PARAMS.keys()), index=0)
base_params = HELICOPTER_PARAMS[selected_heli]
st.sidebar.subheader("2. Paramètres Rotor (éditables)")
c1, c2 = st.sidebar.columns(2)
with c1:
    Omega_rpm = st.number_input("$\\Omega$ (tr/min)", value=base_params['Omega_rpm'])
    ms = st.number_input("$m_s$ (m.kg)", value=base_params['ms'])
    Ip = st.number_input("$I_p$ (m².kg)", value=base_params['Ip'])
with c2:
    e = st.number_input("$e$ (m)", value=base_params['e'])
    b = st.number_input("$b$ (pales)", value=base_params['b'])
Omega_rad_s = -Omega_rpm * np.pi / 30

st.header("Phase 1: Choix du Point de Fonctionnement")
col1, col2, col3 = st.columns([1.5, 2, 2.5])
with col1:
    st.subheader("1.1: Paramètres Adimensionnels")
    phi_deg = st.slider("Angle de perte, $\\phi$ (°)", 15.0, 35.0, 26.0, 0.5)
    tan_phi = np.tan(np.radians(phi_deg))
    st.metric("$\\tan(\\phi)$", f"{tan_phi:.3f}")
    C_phys = e * ms / Ip
    omega_delta_bar = st.slider("Fréquence propre, $\\bar{\\omega}_\\delta$", min_value=max(0.2, np.sqrt(C_phys) + 0.001), max_value=0.8, value=0.4, step=0.005)
    f_fuselage = st.number_input("Fréquence fuselage, $f_{fus}$ (Hz)", value=2.5, step=0.1, format="%.1f")

with col2:
    st.subheader("1.2: Résultats Fréquentiels")
    beat_freq_hz = abs(abs(Omega_rad_s) * (1 - omega_delta_bar)) / (2 * np.pi)
    omega_c_percent = (f_fuselage + (omega_delta_bar * abs(Omega_rad_s) / (2*np.pi))) / (abs(Omega_rad_s) / (2*np.pi)) * 100
    if demo_mode: st.write(f"Pour **{selected_heli}**:")
    st.metric(label=f"$|f_n - f_\\delta|$", value=f"{beat_freq_hz:.2f} Hz")
    st.metric(label=f"$\\Omega_c$", value=f"{omega_c_percent:.1f} % $\\Omega_n$")

st.markdown("##### Comparaison des Résultats")
omega_cs = {}
for name, params in HELICOPTER_PARAMS.items():
    heli_omega_rad = abs(params['Omega_rpm'] * np.pi / 30)
    f_delta_h = (omega_delta_bar * heli_omega_rad) / (2*np.pi)
    f_nominal_h = heli_omega_rad / (2*np.pi)
    omega_cs[name] = (f_fuselage + f_delta_h) / f_nominal_h * 100
show_comparison_metric("Fréquence de croisement $\\Omega_c$ (% $\\Omega_n$)", {k: f"{v:.1f}" for k, v in omega_cs.items()}, "", selected_heli)

k_deltas, k2_deltas = {}, {}
for name, params in HELICOPTER_PARAMS.items():
    heli_C = params['e'] * params['ms'] / params['Ip']
    k_deltas[name] = (params['Ip'] * (params['Omega_rpm']*np.pi/30)**2) * (omega_delta_bar**2 - heli_C)
    k2_deltas[name] = k_deltas[name] * tan_phi
show_comparison_metric("Raideur Angulaire $K_\\delta$ (Nm/rad)", {k: f"{v:,.0f}" for k, v in k_deltas.items()}, "", selected_heli)
show_comparison_metric("Amortissement Angulaire $K_{2\\delta}$ (Nm/rad)", {k: f"{v:,.0f}" for k, v in k2_deltas.items()}, "", selected_heli)
st.session_state['K_delta'] = k_deltas[selected_heli]

with col3:
    st.subheader("1.3: Diagramme K-G")
    sigma_bar = (tan_phi * (omega_delta_bar**2 - C_phys)) / (2 * omega_delta_bar)
    fig_kg = plot_kg_diagram((e, abs(Omega_rad_s), Ip, ms), phi_deg, (omega_delta_bar, sigma_bar), selected_heli)
    st.pyplot(fig_kg)
st.markdown("---")

# ==============================================================================
# LE RESTE DU SCRIPT EST INCHANGÉ
# ==============================================================================
st.header("Phase 2: Définition Géométrique")
col_geo1, col_geo2 = st.columns(2)
with col_geo1:
    st.subheader("2.1: Définition des Points (éditable)")
    coords = HELICOPTER_PARAMS[selected_heli]['coords']
    B_coords = st.text_input("B (global)", value=coords['B'])
    P_coords = st.text_input("P (global)", value=coords['P'])
    M_coords = st.text_input("M (global)", value=coords['M'])
    A_local_coords = st.text_input("A (local/B)", value=coords['A_local'])
    alpha_deg = st.slider("Angle de pré-traînée, $\\alpha$ (°)", -5.0, 5.0, 3.0, 0.1)
    try:
        B, P, M, A_local = (np.array([float(x.strip()) for x in c.split(',')]) for c in [B_coords, P_coords, M_coords, A_local_coords])
    except:
        st.error("Format de coordonnées invalide."); st.stop()
    alpha_rad = np.radians(alpha_deg)
    Apt = (np.array([[np.cos(alpha_rad),-np.sin(alpha_rad),0],[np.sin(alpha_rad),np.cos(alpha_rad),0],[0,0,1]]) @ A_local) + B

    st.markdown("##### Comparaison des Résultats Géométriques")
    rhos, k1s = {}, {}
    for name, params in HELICOPTER_PARAMS.items():
        h_B, h_M, h_A_local = (np.array([float(x.strip()) for x in params['coords'][c].split(',')]) for c in ['B','M','A_local'])
        h_Apt = (np.array([[np.cos(alpha_rad),-np.sin(alpha_rad),0],[np.sin(alpha_rad),np.cos(alpha_rad),0],[0,0,1]]) @ h_A_local) + h_B
        rhos[name] = np.linalg.norm(np.cross(h_Apt - h_B, h_M - h_Apt)) / np.linalg.norm(h_M - h_Apt)
        k1s[name] = k_deltas[name] / rhos[name]**2 / 1000 if rhos[name] > 0 else 0
    show_comparison_metric("Bras de levier $\\rho$", {k: f"{v:.3f}" for k, v in rhos.items()}, "m", selected_heli)
    show_comparison_metric("Raideur Linéaire $K_1$", {k: f"{v:,.0f}" for k, v in k1s.items()}, "N/mm", selected_heli)

with col_geo2:
    st.subheader("2.2: Visualisation Géométrique")
    fig_geo = plot_geometry(B, P, B + A_local, Apt, M)
    st.pyplot(fig_geo)
st.markdown("---")

st.header("Phase 3: Dimensionnement et Analyse de l'Influence de $\\rho$")
c1,c2 = st.columns(2)
with c1:
    st.subheader("3.1: Paramètres d'analyse")
    G_mpa = st.number_input("Module de cisaillement, G (MPa)", value=1.29)
    tau_adm_mpa = st.number_input("Contrainte admissible, τ_adm (MPa)", value=0.05, format="%.3f")
with c2:
    L_etude = st.number_input("Longueur d'élastomère L (mm)", 50.0, 200.0, 80.0, 5.0)

rho_base = rhos[selected_heli]
K_delta_req_base = k_deltas[selected_heli]
delta_L_max_mm_base = 0.1 * rho_base * 1000 

with st.expander("Démonstration Mathématique de l'Influence de Rho"):
    st.markdown("##### Fonctions de base")
    st.latex(r"K_1(\rho) = K_\delta / \rho^2 \quad ; \quad F_{max}(\rho) \propto 1/\rho")
    st.markdown("##### Résolution pour les diamètres (Modèle Logarithmique)")
    st.latex(r"d_{int}(\rho) = \frac{F_{max}(\rho)}{\pi L \tau_{adm}}")
    st.latex(r"K_1(\rho) = \frac{2 \pi G L}{\ln(D_{ext} / d_{int})} \implies D_{ext}(\rho) = d_{int}(\rho) \cdot \exp\left(\frac{2 \pi G L}{K_1(\rho)}\right)")
    st.markdown("##### Fonction Volume résultante")
    st.latex(r"V(\rho) = \frac{\pi L}{4} \left( D_{ext}(\rho)^2 - d_{int}(\rho)^2 \right) \propto \frac{e^{C \cdot \rho^2} - 1}{\rho^2}")

rho_analyse_values = np.linspace(0.1, 0.2, 100)
analysis_results = []
for r in rho_analyse_values:
    K1_req = K_delta_req_base / r**2 / 1000
    delta_L_max = delta_L_max_mm_base * (r / rho_base)
    F_max = delta_L_max * K1_req
    d_int = F_max / (np.pi * L_etude * tau_adm_mpa)
    D_ext_log = d_int * np.exp((2*np.pi*G_mpa*L_etude)/K1_req)
    ep_log = (D_ext_log - d_int) / 2
    vol_log = (np.pi/4 * (D_ext_log**2 - d_int**2) * L_etude) / 1000
    analysis_results.append({ "rho": r, "ep_log": ep_log, "D_ext_log": D_ext_log, "Volume_cm3_log": vol_log})
analysis_df = pd.DataFrame(analysis_results).dropna()

st.subheader("3.2: Graphiques d'influence de $\\rho$")
fig_final, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
ax1.plot(analysis_df['rho'], analysis_df['ep_log'], '-', label=f'{selected_heli}')
ax2.plot(analysis_df['rho'], analysis_df['D_ext_log'], '-', label=f'{selected_heli}')
ax3.plot(analysis_df['rho'], analysis_df['Volume_cm3_log'], '-', label=f'{selected_heli}')
ax1.set_ylabel("Épaisseur ep (mm)"); ax1.set_title(f"Impact de ρ (pour L = {L_etude} mm)"); ax1.grid(True); ax1.legend()
ax2.set_ylabel("Diamètre extérieur D_ext (mm)"); ax2.grid(True); ax2.legend()
ax3.set_ylabel("Volume d'élastomère (cm³)"); ax3.set_xlabel("Bras de levier, ρ (m)"); ax3.grid(True); ax3.legend()
for ax, col_name in [(ax1, 'ep_log'), (ax2, 'D_ext_log'), (ax3, 'Volume_cm3_log')]:
    if not analysis_df.empty:
        y_min, y_max = analysis_df[col_name].min(), analysis_df[col_name].max()
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(bottom=max(0, y_min - margin), top=y_max + margin)
fig_final.tight_layout()
st.pyplot(fig_final)
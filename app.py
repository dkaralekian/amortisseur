import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(layout="wide", page_title="Dimensionnement Amortisseur G5")
st.title("Dimensionnement de l'amortisseur de traînée du G5")
st.markdown("---")

# ==============================================================================
# FONCTIONS DE CALCUL ET DE PLOT (INCHANGÉES)
# ==============================================================================
def plot_kg_diagram(rotor_params, selected_phi_deg, operating_point):
    e, Omega, Ip, ms = rotor_params
    op_point_omega_bar, op_point_sigma_bar = operating_point
    omega_delta_bar_grid = np.linspace(0.1, 1.2, 200)
    sigma_bar_grid = np.linspace(0, 0.10, 200)
    X, Y = np.meshgrid(omega_delta_bar_grid, sigma_bar_grid)
    C = e * ms / Ip
    valid_region = X**2 - C > 0
    K_delta = np.full_like(X, np.nan)
    K_delta[valid_region] = (Ip * Omega**2) * (X[valid_region]**2 - C)
    tan_phi_grid = np.full_like(X, np.nan)
    epsilon = 1e-9
    tan_phi_grid[valid_region] = (2 * Y[valid_region] * X[valid_region]) / (X[valid_region]**2 - C + epsilon)
    Phi = np.degrees(np.arctan(tan_phi_grid))
    fig, ax = plt.subplots(figsize=(10, 7))
    omega_ticks = np.arange(0.2, 0.51, 0.05)
    ax.set_xticks(omega_ticks)
    valid_ticks = omega_ticks[omega_ticks**2 > C]
    kd_levels = (Ip * Omega**2) * (valid_ticks**2 - C)
    kd_contours = ax.contour(X, Y*100, K_delta, levels=kd_levels, cmap='viridis', alpha=0.8)
    ax.clabel(kd_contours, inline=True, fontsize=9, fmt='Kδ=%.0f')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    phi_contours = ax.contour(X, Y*100, Phi, levels=[15, 20, 25, 30, 35], cmap='plasma', linestyles='dashed')
    ax.clabel(phi_contours, inline=True, fontsize=10, fmt='φ=%.0f°')
    tan_phi_selected = np.tan(np.radians(selected_phi_deg))
    sigma_bar_curve = (tan_phi_selected * (omega_delta_bar_grid**2 - C)) / (2 * omega_delta_bar_grid)
    sigma_bar_curve[omega_delta_bar_grid**2 <= C] = np.nan
    ax.plot(omega_delta_bar_grid, sigma_bar_curve * 100, 'r-', linewidth=3, label=f"Courbe pour φ={selected_phi_deg}°")
    ax.plot(op_point_omega_bar, op_point_sigma_bar * 100, 'kx', markersize=12, markeredgewidth=3, label="Point de fonctionnement")
    ax.set_xlabel("Fréquence propre relative, $\\bar{\\omega}_\\delta = \\omega_\\delta/\\Omega$", fontsize=12)
    ax.set_ylabel("Amortissement relatif, $\\bar{\\sigma}$ [%]", fontsize=12)
    ax.set_title("Diagramme de Krysinski-Guimbal (G5)", fontsize=14)
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
    v = M - Apt
    w = B - Apt
    t = np.dot(w, v) / np.dot(v, v)
    proj_point = Apt + t * v
    if 0 <= t <= 1:
        ax.plot([B[0], proj_point[0]], [B[1], proj_point[1]], 'r--', label='ρ (Bras de levier)')
    ax.set_xlabel("Axe X (m)")
    ax.set_ylabel("Axe Y (m)")
    ax.set_title("Visualisation Géométrique 2D (Vue de dessus)")
    ax.grid(True)
    ax.axis('equal')
    ax.legend(fontsize='small')
    return fig

# ==============================================================================
# PARAMETRES GLOBAUX
# ==============================================================================
st.sidebar.title("Paramètres Rotor G5")
c1, c2 = st.sidebar.columns([2,1])
with c1:
    Omega_rpm = st.number_input("Régime nominal, $\\Omega$ (tr/min)", value=404)
    ms = st.number_input("Moment statique, $m_s$ (m.kg)", value=40.0)
    Ip = st.number_input("Inertie pale, $I_p$ (m².kg)", value=125.0)
    e = st.number_input("Excentricité, $e$ (m)", value=0.166)
    b = st.number_input("Nombre de pales, $b$", value=4)

Omega_rad_s = -Omega_rpm * np.pi / 30
rotor_params = (e, Omega_rad_s, Ip, ms)

with st.sidebar.expander("Formules"):
    st.latex(r"\Omega [rad/s] = \frac{\Omega [tr/min] \cdot \pi}{30}")

# ==============================================================================
# PHASE 1: POINT DE FONCTIONNEMENT
# ==============================================================================
st.header("1 Choix du matériau et du point de fonctionnement")
st.subheader("1.1: Angle de Perte φ")
if 'phi_choice' not in st.session_state:
    st.session_state.phi_choice = 'Opt 2'
def set_phi_choice(choice):
    st.session_state.phi_choice = choice
c1, c2, c3 = st.columns([0.5, 1.5, 2])
with c1:
    st.radio("choix", ["Opt 1"], key="r1", index=0 if st.session_state.phi_choice == 'Opt 1' else None, on_change=set_phi_choice, args=('Opt 1',), label_visibility="collapsed")
with c2:
    phi1_deg = st.number_input("Option 1, $\\phi_1$ (°)", value=20.0, step=0.5, format="%.1f", key="phi1_val", label_visibility="visible")
with c3:
    st.markdown(f"<div style='margin-top: 35px;'>$\\tan({phi1_deg}^\\circ) = {np.tan(np.radians(phi1_deg)):.3f}$</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([0.5, 1.5, 2])
with c1:
    st.radio("choix", ["Opt 2"], key="r2", index=0 if st.session_state.phi_choice == 'Opt 2' else None, on_change=set_phi_choice, args=('Opt 2',), label_visibility="collapsed")
with c2:
    phi2_deg = st.number_input("Option 2, $\\phi_2$ (°)", value=24.0, step=0.5, format="%.1f", key="phi2_val", label_visibility="visible")
with c3:
    st.markdown(f"<div style='margin-top: 35px;'>$\\tan({phi2_deg}^\\circ) = {np.tan(np.radians(phi2_deg)):.3f}$</div>", unsafe_allow_html=True)
if st.session_state.phi_choice == 'Opt 1': phi_deg = phi1_deg
else: phi_deg = phi2_deg
tan_phi = np.tan(np.radians(phi_deg))
st.subheader("1.2: Point de Fonctionnement")
C_phys = e * ms / Ip
st.markdown("##### Analyse Fréquentielle")
col_freq1, col_freq2, col_freq3 = st.columns(3)
with col_freq1:
    f_fuselage = st.number_input("Fréquence fuselage, $f_{fus}$ (Hz)", value=2.5, step=0.1, format="%.1f")
current_omega_bar_value = st.session_state.get('omega_slider', 0.4)
omega_delta_rad_s = current_omega_bar_value * abs(Omega_rad_s)
beat_freq_hz = abs(abs(Omega_rad_s) - omega_delta_rad_s) / (2 * np.pi)
f_delta_hz = omega_delta_rad_s / (2*np.pi)
f_nominal_hz = abs(Omega_rad_s) / (2*np.pi)
omega_c_percent = (f_fuselage + f_delta_hz) / f_nominal_hz * 100
with col_freq2:
    st.metric(label="$|f_n - f_\\delta|$ (Hz)", value=f"{beat_freq_hz:.2f}")
with col_freq3:
    st.metric(label="$\\Omega_c$ (% $\\Omega_n$)", value=f"{omega_c_percent:.1f}")
c1, c2 = st.columns([2,1])
with c1:
    omega_delta_bar = st.slider("Fréquence propre relative, $\\bar{\\omega}_\\delta$", min_value=max(0.2, np.sqrt(C_phys) + 0.001), max_value=0.5, value=0.4, step=0.005, key='omega_slider')
sigma_bar = (tan_phi * (omega_delta_bar**2 - C_phys)) / (2 * omega_delta_bar)
with st.expander("Formules du diagramme"):
    st.latex(r"C_{phys} = \frac{e \cdot m_s}{I_p}")
    st.latex(r"\bar{\sigma} = \frac{\tan(\phi) \cdot (\bar{\omega}_\delta^2 - C_{phys})}{2 \cdot \bar{\omega}_\delta}")
    st.latex(r"K_\delta = I_p \Omega^2 (\bar{\omega}_\delta^2 - C_{phys})")
    st.latex(r"K_{2\delta} = K_\delta \cdot \tan(\phi)")
    st.latex(r"f_\delta = \frac{\bar{\omega}_\delta \cdot |\Omega|}{2\pi} \quad ; \quad |f_n - f_\delta| = \frac{||\Omega| - \omega_\delta|}{2\pi}")
    st.latex(r"\Omega_c = \frac{f_{fus} + f_\delta}{f_n} \cdot 100")
col1_results, col2_chart = st.columns([0.4, 0.6])
with col1_results:
    st.markdown("#### Résultats")
    st.metric(label="Amortissement relatif, $\\bar{\\sigma}$", value=f"{sigma_bar:.2%}")
    K_delta = (Ip * Omega_rad_s**2) * (omega_delta_bar**2 - C_phys)
    K2_delta = K_delta * tan_phi
    st.metric(label="Raideur angulaire, $K_\\delta$ (Nm/rad)", value=f"{K_delta:,.0f}")
    st.metric(label="Amortissement angulaire, $K_{2\\delta}$ (Nm/rad)", value=f"{K2_delta:,.0f}")
    st.session_state['K_delta'] = K_delta
with col2_chart:
    fig = plot_kg_diagram(rotor_params, phi_deg, (omega_delta_bar, sigma_bar))
    st.pyplot(fig)
st.markdown("---")

# ==============================================================================
# PHASE 2: GÉOMÉTRIE
# ==============================================================================
st.header("2 Définition Géométrique et Visualisation")
col_geo1, col_geo2 = st.columns([0.5, 0.5])
with col_geo1:
    st.markdown("##### Coordonnées (Repère Global)")
    c1,c2 = st.columns([2,1])
    with c1:
        B_coords = st.text_input("B (Butée Sphérique)", value="0.16605, 0.003, 0")
        P_coords = st.text_input("P (Point de ref. pale)", value="0.371599, 0.003, 0")
        M_coords = st.text_input("M (Attache Moyeu)", value="0.14144, 0.147, 0")
    st.markdown("##### Géométrie Pale (Repère Local, origine en B)")
    c1,c2 = st.columns([2,1])
    with c1:
        A_local_coords = st.text_input("A (Attache Pale)", value="0.2034, 0.07868, 0")
    c1,c2 = st.columns([2,1])
    with c1:
        alpha_deg = st.slider("Angle de pré-traînée, $\\alpha$ (°), >0 la pale recule", -5.0, 5.0, 3.0, 0.1)
    try:
        B, P, M, A_local = (np.array([float(x.strip()) for x in c.split(',')]) for c in [B_coords, P_coords, M_coords, A_local_coords])
    except:
        st.error("Format de coordonnées invalide.")
        st.stop()
    A_zero_global = B + A_local
    alpha_rad = np.radians(alpha_deg)
    Rz_alpha = np.array([[np.cos(alpha_rad),-np.sin(alpha_rad),0],[np.sin(alpha_rad),np.cos(alpha_rad),0],[0,0,1]])
    Apt = (Rz_alpha @ A_local) + B
    L_ref = np.linalg.norm(Apt - M)
    
    # MODIFIÉ: Correction de la formule de Rho
    # Vecteur de la ligne de l'amortisseur
    v_damper_line = M - Apt
    # Vecteur du point de pivot (B) à un point sur la ligne (Apt)
    v_point_to_line = Apt - B
    # Calcul de rho
    rho = np.linalg.norm(np.cross(v_point_to_line, v_damper_line)) / np.linalg.norm(v_damper_line)

    v_blade = P - B
    epsilon = 1e-9
    angle_rad = np.arccos(np.dot(v_blade[:2],v_damper_line[:2])/(np.linalg.norm(v_blade[:2])*np.linalg.norm(v_damper_line[:2])+epsilon))
    angle_pale_amortisseur_deg = 180 - np.degrees(angle_rad)
    
    # MODIFIÉ: Arrondi de l'affichage de rho
    st.metric("Bras de levier, ρ", f"{rho:.2f} m")
    st.metric("Angle Pale/Amortisseur", f"{angle_pale_amortisseur_deg:.2f}°")
    st.markdown("*(Réf. G2: ~18°)*")

    with st.expander("Formules Géométriques"):
        st.latex(r"\vec{A}_{0,global} = \vec{B} + \vec{A}_{local}")
        st.latex(r"R_z(\alpha) = \begin{bmatrix} \cos\alpha & -\sin\alpha & 0 \\ \sin\alpha & \cos\alpha & 0 \\ 0 & 0 & 1 \end{bmatrix}")
        st.latex(r"\vec{A}_{pt} = R_z(\alpha) \cdot \vec{A}_{local} + \vec{B}")
        st.latex(r"\rho = \frac{\| (\vec{A}_{pt} - \vec{B}) \times (\vec{M} - \vec{A}_{pt}) \|}{\| \vec{M} - \vec{A}_{pt} \|}")
        st.latex(r"\theta = 180^\circ - \arccos\left(\frac{\vec{v}_{pale} \cdot \vec{v}_{amortisseur}}{\|\vec{v}_{pale}\| \cdot \|\vec{v}_{amortisseur}\|}\right)")
with col_geo2:
    fig_geo = plot_geometry(B, P, A_zero_global, Apt, M)
    st.pyplot(fig_geo)
st.markdown("---")

# ==============================================================================
# PHASES 3, 4, 5... (inchangées)
# ==============================================================================
#<editor-fold desc="Phases 3, 4, 5">
st.header("3 Analyse des Performances")
with st.expander("Formules de la Dynamique de la Pale"):
    st.latex(r"\text{Angle de traînée statique : } \delta_0 [rad] = \frac{\frac{P_u}{\Omega \cdot b} + K_\delta \cdot \alpha_{rad}}{\Omega^2 e m_s + K_\delta}")
    st.latex(r"\text{Angle de traînée dynamique : } \delta [rad] = -\frac{\alpha_{0,rad} \cdot \alpha_{1,rad}}{1 - \bar{\omega}_\delta^2}")
    st.latex(r"\text{Position statique : } \vec{A}_{stat} = (R_x(a_0) R_y(\theta_0) R_z(\delta_0)) \cdot (\vec{A}_{0} - \vec{B}) + \vec{B}")
    st.latex(r"\text{Allongement statique : } \Delta L_0 = \| \vec{M} - \vec{A}_{stat} \| - L_{ref}")
results_df = pd.DataFrame()
try:
    excel_data = pd.read_excel('simplified_spectrum_p5.xlsx')
    COL_A0, COL_A1 = 'a0', 'a1'
    a0_values, a1_values = excel_data[COL_A0].to_numpy(), excel_data[COL_A1].to_numpy()
    Pu_factors = np.array([0.9, 0.9, 0.9, 1.0, 0.0, 0.9])
    if len(excel_data) == len(Pu_factors): Pu_values = Pu_factors * 250 * 1e3
    else: st.error("Incohérence du nombre de cas de vol."); st.stop()
    K_delta_val = st.session_state.get('K_delta', 0)
    if K_delta_val > 0:
        omega_d_bar_val = np.sqrt((e*ms/Ip) + (K_delta_val / (Ip * Omega_rad_s**2)))
        theta_0_deg = Pu_values * 0.0001 - 6
        delta_0_deg = np.rad2deg((((Pu_values/(Omega_rad_s*b)) + (K_delta_val * alpha_rad)) / (Omega_rad_s**2*e*ms + K_delta_val)))
        delta_deg = -np.rad2deg((np.deg2rad(a0_values) * np.deg2rad(a1_values)) / (1 - omega_d_bar_val**2))
        results = []
        for i in range(len(excel_data)):
            a0_rad, theta_0_rad, delta_0_rad, delta_rad = map(np.radians, [a0_values[i], theta_0_deg[i], delta_0_deg[i], delta_deg[i]])
            Rx_stat=np.array([[1,0,0],[0,np.cos(a0_rad),-np.sin(a0_rad)],[0,np.sin(a0_rad),np.cos(a0_rad)]])
            Ry_stat=np.array([[np.cos(theta_0_rad),0,-np.sin(theta_0_rad)],[0,1,0],[np.sin(theta_0_rad),0,np.cos(theta_0_rad)]])
            Rz_stat=np.array([[np.cos(delta_0_rad),-np.sin(delta_0_rad),0],[np.sin(delta_0_rad),np.cos(delta_0_rad),0],[0,0,1]])
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
        st.dataframe(results_df.style.format({"δ0 (°)":"{:.2f}","δ (°)":"{:.2f}","ΔL0 (mm)":"{:.2f}","ΔL (mm)":"{:+.2f}"}))
except FileNotFoundError: st.error("Le fichier 'simplified_spectrum_p5.xlsx' est introuvable."); st.stop()
except KeyError as e: st.error(f"Erreur: Colonne {e} introuvable."); st.write("Colonnes disponibles:", excel_data.columns.tolist()); st.stop()
st.markdown("---")
st.header("4 Paramètres de Dimensionnement")
if 'K_delta' not in st.session_state or results_df.empty: st.warning("Veuillez compléter les phases 1 à 3."); st.stop()
c1,c2 = st.columns([2,1])
with c1:
    G_mpa = st.number_input("Module de cisaillement, G (MPa)", value=1.29)
    tau_adm_mpa = st.number_input("Contrainte admissible, τ_adm (MPa)", value=0.05, format="%.3f")
    L_etude = st.number_input("Longueur d'élastomère L (mm) à étudier", 50.0, 200.0, 80.0, 5.0)
K_delta_req_base, rho_base = st.session_state.get('K_delta', 0), rho
delta_L_max_mm_base = results_df['ΔL (mm)'].abs().max()
st.markdown("---")
st.header("5 Comparaison des Modèles et Influence de Rho")
st.markdown(f"Analyse de l'impact de `ρ` sur la géométrie pour une longueur d'élastomère fixée à **L = {L_etude} mm**.")
with st.expander("Formules de Dimensionnement et d'Analyse"):
    st.markdown("##### Exigences")
    st.latex(r"K_1 = K_\delta / \rho^2 \quad ; \quad \Delta L_{max} \propto \rho \quad ; \quad F_{max} = \Delta L_{max} \cdot K_1")
    st.markdown("##### Modèle Logarithmique")
    st.latex(r"d_{int} = \frac{F_{max}}{\pi L \tau_{adm}} \quad ; \quad D_{ext} = d_{int} \cdot \exp\left(\frac{2 \pi G L}{K_1}\right)")
    st.markdown("##### Modèle Diamètre Moyen")
    st.latex(r"d_{int} = \frac{F_{max}}{\pi L \tau_{adm}} \quad ; \quad D_{ext} = d_{int} \cdot \frac{K_1 + G \pi L}{K_1 - G \pi L}")
    st.markdown("##### Grandeurs Calculées")
    st.latex(r"ep = \frac{D_{ext} - d_{int}}{2} \quad ; \quad V = \frac{\pi}{4} (D_{ext}^2 - d_{int}^2) \cdot L")
rho_analyse_values = np.linspace(0.1, 0.2, 100)
analysis_results = []
for r in rho_analyse_values:
    K1_req = K_delta_req_base / r**2 / 1000
    delta_L_max = delta_L_max_mm_base * (r / rho_base)
    F_max = delta_L_max * K1_req
    d_int_log = F_max / (np.pi * L_etude * tau_adm_mpa)
    exp_term_log = (2 * np.pi * G_mpa * L_etude) / K1_req
    D_ext_log = d_int_log * np.exp(exp_term_log)
    ep_log = (D_ext_log - d_int_log) / 2
    volume_cm3_log = (np.pi / 4 * (D_ext_log**2 - d_int_log**2) * L_etude) / 1000
    d_int_moy = d_int_log
    numerator_moy = K1_req + G_mpa * np.pi * L_etude
    denominator_moy = K1_req - G_mpa * np.pi * L_etude
    if denominator_moy > 0:
        D_ext_moy = d_int_moy * numerator_moy / denominator_moy
        ep_moy = (D_ext_moy - d_int_moy) / 2
        volume_cm3_moy = (np.pi / 4 * (D_ext_moy**2 - d_int_moy**2) * L_etude) / 1000
    else:
        ep_moy, D_ext_moy, volume_cm3_moy = np.nan, np.nan, np.nan
    analysis_results.append({
        "rho": r, "ep_log": ep_log, "D_ext_log": D_ext_log, "Volume_cm3_log": volume_cm3_log,
        "ep_moy": ep_moy, "D_ext_moy": D_ext_moy, "Volume_cm3_moy": volume_cm3_moy
    })
analysis_df = pd.DataFrame(analysis_results).dropna()
fig_final, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
ax1.plot(analysis_df['rho'], analysis_df['ep_log'], 'b-', label='Modèle Logarithmique')
ax1.plot(analysis_df['rho'], analysis_df['ep_moy'], 'b--', label='Modèle Diamètre Moyen')
ax1.set_ylabel("Épaisseur ep (mm)")
ax1.set_title(f"Impact de ρ sur la Géométrie de l'Amortisseur (pour L = {L_etude} mm)")
ax1.grid(True)
ax1.legend()
if not analysis_df.empty:
    y_min,y_max=analysis_df[['ep_log','ep_moy']].min().min(),analysis_df[['ep_log','ep_moy']].max().max()
    margin=(y_max-y_min)*0.1; ax1.set_ylim(bottom=max(0,y_min-margin),top=y_max+margin)
ax2.plot(analysis_df['rho'], analysis_df['D_ext_log'], 'r-', label='Modèle Logarithmique')
ax2.plot(analysis_df['rho'], analysis_df['D_ext_moy'], 'r--', label='Modèle Diamètre Moyen')
ax2.set_ylabel("Diamètre extérieur D_ext (mm)")
ax2.grid(True)
ax2.legend()
if not analysis_df.empty:
    y_min,y_max=analysis_df[['D_ext_log','D_ext_moy']].min().min(),analysis_df[['D_ext_log','D_ext_moy']].max().max()
    margin=(y_max-y_min)*0.1; ax2.set_ylim(bottom=max(0,y_min-margin),top=y_max+margin)
ax3.plot(analysis_df['rho'], analysis_df['Volume_cm3_log'], 'g-', label='Modèle Logarithmique')
ax3.plot(analysis_df['rho'], analysis_df['Volume_cm3_moy'], 'g--', label='Modèle Diamètre Moyen')
ax3.set_ylabel("Volume d'élastomère (cm³)")
ax3.set_xlabel("Bras de levier, ρ (m)")
ax3.grid(True)
ax3.legend()
if not analysis_df.empty:
    y_min,y_max=analysis_df[['Volume_cm3_log','Volume_cm3_moy']].min().min(),analysis_df[['Volume_cm3_moy']].max().max()
    margin=(y_max-y_min)*0.1; ax3.set_ylim(bottom=max(0,y_min-margin),top=y_max+margin)
fig_final.tight_layout()
st.pyplot(fig_final)
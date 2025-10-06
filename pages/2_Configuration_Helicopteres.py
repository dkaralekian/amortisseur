import streamlit as st
import json

# Nom du fichier de base de données
DB_FILE = "helicopters.json"

# Fonctions pour charger et sauvegarder la base de données
def load_helicopter_db():
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Retourne les valeurs par défaut si le fichier est manquant ou corrompu
        st.warning(f"Fichier {DB_FILE} non trouvé ou corrompu. Utilisation des valeurs par défaut.")
        return {
            'G5': {'Omega_rpm': 404, 'ms': 40.0, 'Ip': 125.0, 'e': 0.166, 'b': 4, 'coords': {'B': "0.166,0.003,0", 'P': "0.372,0.003,0", 'M': "0.141,0.147,0", 'A_local': "0.203,0.079,0"}},
            'G2': {'Omega_rpm': 530, 'ms': 23.6, 'Ip': 56.2, 'e': 0.154, 'b': 3, 'coords': {'B': "0.154,0,0", 'P': "0.350,0,0", 'M': "0.130,0.130,0", 'A_local': "0.190,0.060,0"}},
            'EC120': {'Omega_rpm': 408, 'ms': 35.0, 'Ip': 110.0, 'e': 0.160, 'b': 3, 'coords': {'B': "0.160,0.01,0", 'P': "0.360,0.01,0", 'M': "0.135,0.140,0", 'A_local': "0.195,0.070,0"}},
        }

def save_helicopter_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    st.success("Base de données des hélicoptères mise à jour !")

# --- Configuration de la Page ---
st.set_page_config(layout="wide", page_title="Configuration Hélicoptères")
st.title("Configuration des Hélicoptères de Référence")

# Chargement des données actuelles
heli_data = load_helicopter_db()

# --- Création des onglets pour chaque hélicoptère modifiable ---
tab_g2, tab_ec120 = st.tabs(["G2", "EC120"])

with tab_g2:
    st.subheader("Paramètres du G2")
    p = heli_data.get('G2', {})
    c = p.get('coords', {})
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Omega (tr/min)", value=p.get('Omega_rpm', 0.0), key="g2_omega")
        st.number_input("Moment statique (m.kg)", value=p.get('ms', 0.0), key="g2_ms", format="%.2f")
        st.number_input("Inertie pale (m².kg)", value=p.get('Ip', 0.0), key="g2_ip", format="%.2f")
    with c2:
        st.number_input("Excentricité (m)", value=p.get('e', 0.0), key="g2_e", format="%.3f")
        st.number_input("Nombre de pales", value=p.get('b', 0), step=1, key="g2_b")
    st.subheader("Coordonnées Géométriques (G2)")
    st.text_input("Point B (global)", value=c.get('B', "0,0,0"), key="g2_b_coord")
    st.text_input("Point P (global)", value=c.get('P', "0,0,0"), key="g2_p_coord")
    st.text_input("Point M (global)", value=c.get('M', "0,0,0"), key="g2_m_coord")
    st.text_input("Point A (local/B)", value=c.get('A_local', "0,0,0"), key="g2_a_coord")

with tab_ec120:
    st.subheader("Paramètres de l'EC120")
    p = heli_data.get('EC120', {})
    c = p.get('coords', {})

    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Omega (tr/min)", value=p.get('Omega_rpm', 0.0), key="ec120_omega")
        st.number_input("Moment statique (m.kg)", value=p.get('ms', 0.0), key="ec120_ms", format="%.2f")
        st.number_input("Inertie pale (m².kg)", value=p.get('Ip', 0.0), key="ec120_ip", format="%.2f")
    with c2:
        st.number_input("Excentricité (m)", value=p.get('e', 0.0), key="ec120_e", format="%.3f")
        st.number_input("Nombre de pales", value=p.get('b', 0), step=1, key="ec120_b")

    st.subheader("Coordonnées Géométriques (EC120)")
    st.text_input("Point B (global)", value=c.get('B', "0,0,0"), key="ec120_b_coord")
    st.text_input("Point P (global)", value=c.get('P', "0,0,0"), key="ec120_p_coord")
    st.text_input("Point M (global)", value=c.get('M', "0,0,0"), key="ec120_m_coord")
    st.text_input("Point A (local/B)", value=c.get('A_local', "0,0,0"), key="ec120_a_coord")

st.markdown("---")
if st.button("Enregistrer les modifications", use_container_width=True):
    # Recréer le dictionnaire de données à partir des widgets de la session
    updated_data = load_helicopter_db() # On part de la base existante pour ne pas écraser le G5
    
    updated_data['G2']['Omega_rpm'] = st.session_state.g2_omega
    updated_data['G2']['ms'] = st.session_state.g2_ms
    updated_data['G2']['Ip'] = st.session_state.g2_ip
    updated_data['G2']['e'] = st.session_state.g2_e
    updated_data['G2']['b'] = st.session_state.g2_b
    updated_data['G2']['coords']['B'] = st.session_state.g2_b_coord
    updated_data['G2']['coords']['P'] = st.session_state.g2_p_coord
    updated_data['G2']['coords']['M'] = st.session_state.g2_m_coord
    updated_data['G2']['coords']['A_local'] = st.session_state.g2_a_coord
    
    updated_data['EC120']['Omega_rpm'] = st.session_state.ec120_omega
    updated_data['EC120']['ms'] = st.session_state.ec120_ms
    updated_data['EC120']['Ip'] = st.session_state.ec120_ip
    updated_data['EC120']['e'] = st.session_state.ec120_e
    updated_data['EC120']['b'] = st.session_state.ec120_b
    updated_data['EC120']['coords']['B'] = st.session_state.ec120_b_coord
    updated_data['EC120']['coords']['P'] = st.session_state.ec120_p_coord
    updated_data['EC120']['coords']['M'] = st.session_state.ec120_m_coord
    updated_data['EC120']['coords']['A_local'] = st.session_state.ec120_a_coord
    
    save_helicopter_db(updated_data)
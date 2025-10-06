import streamlit as st
import json

DB_FILE = "helicopters.json"

def load_helicopter_db():
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'G5': {'Omega_rpm': 404, 'ms': 40.0, 'Ip': 125.0, 'e': 0.166, 'b': 4, 'f_fuselage': 2.5,
                   'coords': {'B': "0.166,0.003,0", 'P': "0.372,0.003,0", 'M': "0.141,0.147,0", 'A_local': "0.203,0.079,0"}},
            'G2': {'Omega_rpm': 530, 'ms': 23.6, 'Ip': 56.2, 'e': 0.148, 'b': 3, 'f_fuselage': 2.0,
                   'material': {'G_mpa': 1.2, 'phi_deg': 22.0}, 'elastomer': {'L': 86.0, 'd_int': 30.0, 'ep': 5.0},
                   'coords': {'B': "0.154,0,0", 'P': "0.350,0,0", 'M': "0.130,0.130,0", 'A_local': "0.190,0.060,0"}},
            'EC120': {'Omega_rpm': 408, 'ms': 61.1, 'Ip': 200.8, 'e': 0.183, 'b': 3, 'f_fuselage': 1.5,
                      'material': {'G_mpa': 0.94, 'phi_deg': 19.0}, 'elastomer': {'L': 153.0, 'd_int': 44.0, 'ep': 7.0},
                      'coords': {'B': "0.160,0.01,0", 'P': "0.360,0.01,0", 'M': "0.135,0.140,0", 'A_local': "0.195,0.070,0"}},
        }

def save_helicopter_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    st.success("Base de données des hélicoptères mise à jour !")

st.set_page_config(layout="wide", page_title="Configuration Hélicoptères")
st.title("Configuration des Hélicoptères de Référence")

heli_data = load_helicopter_db()
tab_g2, tab_ec120 = st.tabs(["G2", "EC120"])

def build_tab(heli_name, data):
    p = data.get(heli_name, {})
    st.subheader(f"Paramètres Rotor du {heli_name}")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Omega (tr/min)", value=p.get('Omega_rpm', 0.0), key=f"{heli_name}_omega")
        st.number_input("Moment statique (m.kg)", value=p.get('ms', 0.0), key=f"{heli_name}_ms", format="%.2f")
        st.number_input("Inertie pale (m².kg)", value=p.get('Ip', 0.0), key=f"{heli_name}_ip", format="%.2f")
    with c2:
        st.number_input("Excentricité (m)", value=p.get('e', 0.0), key=f"{heli_name}_e", format="%.3f")
        st.number_input("Nombre de pales", value=p.get('b', 0), step=1, key=f"{heli_name}_b")
        st.number_input("Fréquence fuselage (Hz)", value=p.get('f_fuselage', 0.0), key=f"{heli_name}_f_fuselage", format="%.2f")

    st.subheader(f"Coordonnées Géométriques ({heli_name})")
    coords = p.get('coords', {})
    st.text_input("Point B (global)", value=coords.get('B', "0,0,0"), key=f"{heli_name}_b_coord")
    st.text_input("Point P (global)", value=coords.get('P', "0,0,0"), key=f"{heli_name}_p_coord")
    st.text_input("Point M (global)", value=coords.get('M', "0,0,0"), key=f"{heli_name}_m_coord")
    st.text_input("Point A (local/B)", value=coords.get('A_local', "0,0,0"), key=f"{heli_name}_a_coord")
    
    st.subheader(f"Paramètres Amortisseur ({heli_name})")
    mat = p.get('material', {})
    elas = p.get('elastomer', {})
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Module G (MPa)", value=mat.get('G_mpa', 0.0), key=f"{heli_name}_g")
        st.number_input("Angle de perte phi (°)", value=mat.get('phi_deg', 0.0), key=f"{heli_name}_phi")
    with c2:
        st.number_input("Longueur L (mm)", value=elas.get('L', 0.0), key=f"{heli_name}_L")
        st.number_input("Diamètre intérieur d (mm)", value=elas.get('d_int', 0.0), key=f"{heli_name}_d_int")
        st.number_input("Épaisseur ep (mm)", value=elas.get('ep', 0.0), key=f"{heli_name}_ep")

with tab_g2:
    build_tab("G2", heli_data)
with tab_ec120:
    build_tab("EC120", heli_data)

st.markdown("---")
if st.button("Enregistrer les modifications", use_container_width=True):
    updated_data = load_helicopter_db()
    for name in ["G2", "EC120"]:
        updated_data[name] = {
            'Omega_rpm': st.session_state[f'{name}_omega'], 'ms': st.session_state[f'{name}_ms'], 'Ip': st.session_state[f'{name}_ip'],
            'e': st.session_state[f'{name}_e'], 'b': st.session_state[f'{name}_b'], 'f_fuselage': st.session_state[f'{name}_f_fuselage'],
            'coords': {'B': st.session_state[f'{name}_b_coord'], 'P': st.session_state[f'{name}_p_coord'], 'M': st.session_state[f'{name}_m_coord'], 'A_local': st.session_state[f'{name}_a_coord']},
            'material': {'G_mpa': st.session_state[f'{name}_g'], 'phi_deg': st.session_state[f'{name}_phi']},
            'elastomer': {'L': st.session_state[f'{name}_L'], 'd_int': st.session_state[f'{name}_d_int'], 'ep': st.session_state[f'{name}_ep']}
        }
    save_helicopter_db(updated_data)
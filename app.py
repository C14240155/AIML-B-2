import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
import random
import os

class MamdaniFuzzySystem:
    @staticmethod
    def triangular(x, a, b, c):
        if x <= a or x >= c: return 0.0
        elif a < x <= b: return (x - a) / (b - a)
        elif b < x < c: return (c - x) / (c - b)
        return 0.0

    @staticmethod
    def trapezoid(x, a, b, c, d):
        if x <= a or x >= d: return 0.0
        elif a < x <= b: return (x - a) / (b - a)
        elif b < x <= c: return 1.0
        elif c < x < d: return (d - x) / (d - c)
        return 0.0

    @staticmethod
    def left_shoulder(x, a, b):
        if x <= a: return 1.0
        elif a < x <= b: return (b - x) / (b - a)
        return 0.0

    @staticmethod
    def right_shoulder(x, a, b):
        if x <= a: return 0.0
        elif a < x <= b: return (x - a) / (b - a)
        return 1.0

    # Fuzzy Set
    def evaluate(self, dw, cog, overstowage, slot_error):
        # 1. Fuzzifikasi Dw (Keseimbangan Berat Kapal)
        dw_seimbang = self.left_shoulder(dw, 10, 30)
        dw_miring = self.triangular(dw, 20, 80, 150)
        dw_sangat_miring = self.right_shoulder(dw, 100, 250)

        # 2. Fuzzifikasi Center of Gravity (Stabilitas Vertikal)
        cog_aman = self.left_shoulder(cog, 0.4, 0.5)
        cog_waspada = self.triangular(cog, 0.45, 0.6, 0.75)
        cog_berbahaya = self.right_shoulder(cog, 0.7, 0.85)

        # 3. Fuzzifikasi Overstowage (Pelanggaran Urutan Bongkar)
        ovr_ideal = self.left_shoulder(overstowage, 0, 3)
        ovr_toleransi = self.triangular(overstowage, 2, 10, 25)
        ovr_buruk = self.right_shoulder(overstowage, 15, 40)

        # 4. Fuzzifikasi Slot Error (Kesalahan Penempatan Container)
        slt_valid = self.left_shoulder(slot_error, 0, 4) 
        slt_fatal = self.right_shoulder(slot_error, 2, 20)

        # RULE BASE INFERENCE
        rules = []
        rules.append((slt_fatal, 'SANGAT_RENDAH')) 
        rules.append((cog_berbahaya, 'SANGAT_RENDAH')) 
        rules.append((dw_sangat_miring, 'SANGAT_RENDAH')) 
        rules.append((min(dw_seimbang, cog_aman, ovr_ideal, slt_valid), 'SANGAT_TINGGI')) 
        rules.append((min(dw_miring, cog_aman, ovr_ideal, slt_valid), 'TINGGI')) 
        rules.append((min(dw_seimbang, cog_aman, ovr_toleransi, slt_valid), 'TINGGI')) 
        rules.append((min(dw_seimbang, cog_waspada, ovr_ideal, slt_valid), 'SEDANG')) 
        rules.append((min(dw_seimbang, cog_waspada, ovr_toleransi, slt_valid), 'RENDAH')) 
        rules.append((min(dw_miring, cog_aman, ovr_toleransi, slt_valid), 'SEDANG')) 
        rules.append((min(dw_miring, cog_waspada, ovr_ideal, slt_valid), 'RENDAH')) 
        rules.append((min(dw_miring, cog_waspada, ovr_toleransi, slt_valid), 'SANGAT_RENDAH')) 
        rules.append((min(dw_seimbang, cog_aman, ovr_buruk, slt_valid), 'SEDANG')) 
        rules.append((min(dw_miring, cog_aman, ovr_buruk, slt_valid), 'RENDAH')) 
        rules.append((min(dw_seimbang, cog_waspada, ovr_buruk, slt_valid), 'SANGAT_RENDAH')) 

        # DEFUZZIFIKASI 
        x_output = np.linspace(0, 100, 100)
        aggregated_membership = np.zeros_like(x_output)

        for activation, label in rules:
            if activation <= 0: continue 
            if label == 'SANGAT_RENDAH':
                fuzzy_vals = np.array([self.left_shoulder(val, 0, 20) for val in x_output])
                mu = np.minimum(activation, fuzzy_vals)
            elif label == 'RENDAH':
                fuzzy_vals = np.array([self.triangular(val, 0, 25, 50) for val in x_output])
                mu = np.minimum(activation, fuzzy_vals)
            elif label == 'SEDANG':
                fuzzy_vals = np.array([self.triangular(val, 25, 50, 75) for val in x_output])
                mu = np.minimum(activation, fuzzy_vals)
            elif label == 'TINGGI':
                fuzzy_vals = np.array([self.triangular(val, 50, 75, 100) for val in x_output])
                mu = np.minimum(activation, fuzzy_vals)
            elif label == 'SANGAT_TINGGI':
                fuzzy_vals = np.array([self.right_shoulder(val, 80, 100) for val in x_output])
                mu = np.minimum(activation, fuzzy_vals)
            
            aggregated_membership = np.maximum(aggregated_membership, mu)

        sum_membership = np.sum(aggregated_membership)
        
        # FITUR BARU: Pastikan setiap individu punya skor unik lewat penalti desimal
        raw_penalty = (dw * 0.001) + (overstowage * 0.01) + (slot_error * 0.02)

        if sum_membership == 0:
            penyebut = 1.0 + (dw * 0.01) + (overstowage * 0.1) + (slot_error * 0.2)
            return 15.0 / (penyebut + raw_penalty) 
        
        base_fuzzy_score = np.sum(x_output * aggregated_membership) / sum_membership
        
        return base_fuzzy_score / (1.0 + raw_penalty)


class StowageGA:
    def __init__(self, vessel_info, cargo_df, pop_size=30, generations=20, mutation_rate=0.1):
        self.vessel = vessel_info
        self.cargo = cargo_df.to_dict(orient='records')
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fuzzy_sys = MamdaniFuzzySystem()
        
        self.max_b = int(vessel_info['Max Bay'])
        self.max_r = int(vessel_info['Max Row'])
        self.max_t = int(vessel_info['Max Tier'])
        
        self.reefer_plugs = self.parse_coordinates(vessel_info['Reefer Plugs'])
        self.dg_zones = self.parse_coordinates(vessel_info['DG Zone'])

    def parse_coordinates(self, coord_str):
        try:
            raw = ast.literal_eval(f"[{coord_str}]")
            return set(tuple(int(v) for v in c) for c in raw)
        except:
            return set()

    def apply_gravity(self, chromosome):
        new_chromosome = {}
        columns = {}
        
        for (b, r, t), item in chromosome.items():
            if (b, r) not in columns:
                columns[(b, r)] = []
            columns[(b, r)].append((t, item))
            
        for (b, r), items in columns.items():
            items.sort(key=lambda x: x[0])
            for new_t, (old_t, item) in enumerate(items):
                if new_t < self.max_t:
                    new_chromosome[(b, r, new_t)] = item
                
        return new_chromosome

    def create_individual(self):
        chromosome = {}
        
        reefers = [c for c in self.cargo if c['Tipe Cargo'].upper() == 'REEFER']
        dgs = [c for c in self.cargo if c['Tipe Cargo'].upper() == 'DG']
        drys = [c for c in self.cargo if c['Tipe Cargo'].upper() not in ['REEFER', 'DG']]
        
        random.shuffle(reefers)
        random.shuffle(dgs)
        random.shuffle(drys)
        
        reefer_slots = list(self.reefer_plugs)
        dg_slots = list(self.dg_zones)
        all_slots = set((b, r, t) for b in range(self.max_b) for r in range(self.max_r) for t in range(self.max_t))
        
        dead_slots = set()
        for b, r, t in self.reefer_plugs.union(self.dg_zones):
            for t_above in range(t + 1, self.max_t):
                dead_slots.add((b, r, t_above))
                
        regular_slots = list(all_slots - self.reefer_plugs - self.dg_zones - dead_slots)
        
        random.shuffle(reefer_slots)
        random.shuffle(dg_slots)
        random.shuffle(regular_slots)
        
        for item in reefers:
            if reefer_slots: chromosome[reefer_slots.pop()] = item
        for item in dgs:
            if dg_slots: chromosome[dg_slots.pop()] = item
        for item in drys:
            if regular_slots: chromosome[regular_slots.pop()] = item
            
        return self.apply_gravity(chromosome)

    def evaluate_fitness(self, chromosome):
        overstowage_penalty = 0
        slot_error = 0
        weight_matrix = np.zeros((self.max_b, self.max_r, self.max_t))
        
        total_weight = 0
        vertical_moment = 0
        
        for coord, item in chromosome.items():
            b, r, t = coord
            w = float(item['Berat (Ton)'])
            weight_matrix[b, r, t] = w
            
            total_weight += w
            vertical_moment += w * (t + 1) 
            
            if item['Tipe Cargo'].upper() == 'REEFER':
                if coord not in self.reefer_plugs: slot_error += 1
                if t != 0: slot_error += 1
                for t_above in range(t + 1, self.max_t):
                    if (b, r, t_above) in chromosome: slot_error += 1

            elif item['Tipe Cargo'].upper() == 'DG':
                if coord not in self.dg_zones: slot_error += 1
                if t != 0: slot_error += 1
                if b == 0 or b == self.max_b - 1: slot_error += 1
                if r == 0 or r == self.max_r - 1: slot_error += 1
                for t_above in range(t + 1, self.max_t):
                    if (b, r, t_above) in chromosome: slot_error += 1
            
            elif item['Tipe Cargo'].upper() in ['DRY', 'GENERAL']:
                if coord in self.reefer_plugs or coord in self.dg_zones:
                    slot_error += 1

        for b in range(self.max_b):
            for r in range(self.max_r):
                for t_under in range(self.max_t):
                    for t_above in range(t_under + 1, self.max_t):
                        item_under = chromosome.get((b, r, t_under))
                        item_above = chromosome.get((b, r, t_above))
                        if item_under and item_above:
                            if int(item_under['Urutan Pelabuhan']) < int(item_above['Urutan Pelabuhan']):
                                overstowage_penalty += 1

        left_side_weight = np.sum(weight_matrix[:, :self.max_r//2, :])
        right_side_weight = np.sum(weight_matrix[:, (self.max_r+1)//2:, :])
        dw = abs(left_side_weight - right_side_weight) 

        cog_index = 0.0
        if total_weight > 0:
            avg_tier_height = vertical_moment / total_weight
            raw_cog = (avg_tier_height - 1) / (self.max_t - 1) if self.max_t > 1 else 0.0
            cog_index = max(0.0, min(1.0, raw_cog))

        fitness = self.fuzzy_sys.evaluate(dw, cog_index, overstowage_penalty, slot_error)
        stability_display = max(0, 100 - ((dw / total_weight) * 200)) if total_weight > 0 else 100
        
        return fitness, stability_display, overstowage_penalty, slot_error

    def crossover(self, parent1, parent2):
        child = {}
        used_slots = set()
        unassigned_cargo = []
        
        all_slots = [(b, r, t) for b in range(self.max_b) for r in range(self.max_r) for t in range(self.max_t)]
        
        for item in self.cargo:
            c_id = item['Container ID']
            
            slot1 = next((k for k, v in parent1.items() if v['Container ID'] == c_id), None)
            slot2 = next((k for k, v in parent2.items() if v['Container ID'] == c_id), None)
            
            chosen_slot = slot1 if random.random() < 0.5 else slot2
            
            if chosen_slot and chosen_slot not in used_slots:
                child[chosen_slot] = item
                used_slots.add(chosen_slot)
            else:
                unassigned_cargo.append(item) 
                
        available_slots = list(set(all_slots) - used_slots)
        random.shuffle(available_slots)
        
        for item in unassigned_cargo:
            if available_slots:
                new_slot = available_slots.pop()
                child[new_slot] = item
                used_slots.add(new_slot)
                
        return self.apply_gravity(child)
    
    def run(self):
        population = []
        
        for _ in range(self.pop_size):
            population.append(self.create_individual())
            
        best_individual = None
        best_fitness = -1.0
        best_metrics = {}
        fit_history = []
        
        all_slots = [(b, r, t) for b in range(self.max_b) for r in range(self.max_r) for t in range(self.max_t)]
        
        for gen in range(self.generations):
            pop_fitness_details = []
            for ind in population:
                fit, stab, over, slot = self.evaluate_fitness(ind)
                pop_fitness_details.append((fit, ind, {"stability": stab, "overstowage": over, "slot_error": slot}))
            
            pop_fitness_details.sort(key=lambda x: x[0], reverse=True)
            current_best_fit, current_best_ind, current_best_met = pop_fitness_details[0]
            fit_history.append(current_best_fit)
            
            if current_best_fit > best_fitness:
                best_fitness = current_best_fit
                best_individual = current_best_ind
                best_metrics = current_best_met
                
            elite_count = min(2, len(pop_fitness_details))
            new_population = [pop_fitness_details[i][1] for i in range(elite_count)]
            
            limit_crossover = self.pop_size - max(1, int(self.pop_size * 0.10))
            
            while len(new_population) < limit_crossover:
                pool_size = max(1, self.pop_size // 2)
                parent1 = random.choice(pop_fitness_details[:pool_size])[1]
                parent2 = random.choice(pop_fitness_details[:pool_size])[1]
                
                child = self.crossover(parent1, parent2)
                
                if random.random() < self.mutation_rate and len(child) > 1:
                    num_swaps = random.randint(1, 3)
                    for _ in range(num_swaps):
                        c_isi = random.choice(list(child.keys()))  
                        
                        if random.random() < 0.5:
                            t_isi = c_isi[2]
                            c_acak = (random.randint(0, self.max_b - 1), random.randint(0, self.max_r - 1), t_isi)
                        else:
                            c_acak = random.choice(all_slots)  
                        
                        if c_acak != c_isi:
                            temp = child.get(c_acak)
                            child[c_acak] = child[c_isi]
                            if temp is not None:
                                child[c_isi] = temp
                            else:
                                del child[c_isi]  
                
                child = self.apply_gravity(child)
                        
                new_population.append(child)
                
            while len(new_population) < self.pop_size:
                new_population.append(self.create_individual())
                
            population = new_population
            
        return best_individual, best_fitness, best_metrics, fit_history

def draw_3d_cube(fig, b, r, t, color, hover_text, text_id):
    dx, dy, dz = 0.4, 0.4, 0.4
    x = [b-dx, b+dx, b+dx, b-dx, b-dx, b+dx, b+dx, b-dx]
    y = [r-dy, r-dy, r+dy, r+dy, r-dy, r-dy, r+dy, r+dy]
    z = [t-dz, t-dz, t-dz, t-dz, t+dz, t+dz, t+dz, t+dz]
    
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=0.9,
        hoverinfo="text", text=hover_text,
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[b], y=[r], z=[t], mode="text",
        text=[text_id], textposition="middle center",
        textfont=dict(color="white", size=11, family="Arial Black"),
        hoverinfo="skip", showlegend=False
    ))

def draw_wireframe_outline(fig, b, r, t, color, name, show_legend=False):
    dx, dy, dz = 0.45, 0.45, 0.45  
    x = [b-dx, b+dx, b+dx, b-dx, b-dx, b-dx, b+dx, b+dx, b-dx, b-dx, None, b+dx, b+dx, None, b+dx, b+dx, None, b-dx, b-dx]
    y = [r-dy, r-dy, r+dy, r+dy, r-dy, r-dy, r-dy, r+dy, r+dy, r-dy, None, r-dy, r-dy, None, r+dy, r+dy, None, r+dy, r+dy]
    z = [t-dz, t-dz, t-dz, t-dz, t-dz, t+dz, t+dz, t+dz, t+dz, t+dz, None, t-dz, t+dz, None, t-dz, t+dz, None, t-dz, t+dz]
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(color=color, width=4),
        name=name, showlegend=show_legend, hoverinfo="skip"
    ))

st.set_page_config(page_title="Stowage Optimizer AI", layout="wide")
st.title("Sistem Optimasi Tata Letak Kargo Palka Kapal PT Pelabuhan Kelompok 2 AIML")
st.subheader("Hybrid Genetic Algorithm & Fuzzy Inference System Mamdani")
st.markdown("---")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_VESSEL_CSV = os.path.join(_BASE_DIR, "dataset_kapal.csv")

try:
    vessel_df = pd.read_csv(_VESSEL_CSV, sep=";")
    vessel_df.columns = vessel_df.columns.str.strip()
except FileNotFoundError:
    st.error(f"File 'dataset_kapal.csv' tidak ditemukan di: {_BASE_DIR}\nTaruh file di folder yang sama dengan app.py.")
    st.stop()

st.sidebar.header("Panel Konfigurasi Simulasi")
selected_vessel_name = st.sidebar.selectbox("Pilih Kapal Evaluasi:", vessel_df["Nama Kapal"].unique())
vessel_info = vessel_df[vessel_df["Nama Kapal"] == selected_vessel_name].iloc[0]

st.sidebar.info(f"**Rute Kapal:**\n{vessel_info['Rute Pelabuhan']}\n\n"
                f"**Dimensi Matriks Palka Kapal:**\n"
                f"Bay: {vessel_info['Max Bay']} | Row: {vessel_info['Max Row']} | Tier: {vessel_info['Max Tier']}")

st.sidebar.markdown("---")
uploaded_cargo = st.sidebar.file_uploader("Upload Dataset Kargo (.csv):", type=["csv"])

pop_size = st.sidebar.slider("Ukuran Populasi GA:", min_value=20, max_value=300, value=100, step=10)
generations = st.sidebar.slider("Jumlah Generasi Simulasi:", min_value=10, max_value=500, value=200, step=10)
mutation_rate = st.sidebar.slider("Mutation Rate:", min_value=0.01, max_value=1.0, value=0.3, step=0.05)

if uploaded_cargo is not None:
    cargo_df = pd.read_csv(uploaded_cargo, sep=";")
    cargo_df.columns = cargo_df.columns.str.strip()
    
    total_hold_capacity = int(vessel_info['Max Bay']) * int(vessel_info['Max Row']) * int(vessel_info['Max Tier'])
    
    def _parse_coord_list(coord_str):
        try:
            raw = ast.literal_eval(f"[{coord_str}]")
            return [tuple(int(v) for v in c) for c in raw]
        except:
            return []

    reefer_plugs_list = _parse_coord_list(vessel_info['Reefer Plugs']) if pd.notna(vessel_info['Reefer Plugs']) else []
    dg_zones_list = _parse_coord_list(vessel_info['DG Zone']) if pd.notna(vessel_info['DG Zone']) else []
    
    total_kargo_reefer = len(cargo_df[cargo_df['Tipe Cargo'].str.upper() == 'REEFER'])
    total_kargo_dg = len(cargo_df[cargo_df['Tipe Cargo'].str.upper() == 'DG'])
    total_kargo_dry = len(cargo_df[cargo_df['Tipe Cargo'].str.upper().isin(['DRY', 'GENERAL'])])
    
    is_valid = True
    
    st.sidebar.markdown("### Status Validasi Pre-Stowage")
    
    if len(cargo_df) > total_hold_capacity:
        st.sidebar.error(f"**Kapasitas Penuh!** Total kargo yang diunggah ({len(cargo_df)}) melebihi slot maksimal kapal ({total_hold_capacity}).")
        is_valid = False
        
    if total_kargo_reefer > len(reefer_plugs_list):
        st.sidebar.error(f"**Reefer Ditolak (Rollover):** Anda membawa {total_kargo_reefer} kontainer Reefer, tetapi kapal hanya memiliki {len(reefer_plugs_list)} Reefer Plugs.")
        is_valid = False
        
    if total_kargo_dg > len(dg_zones_list):
        st.sidebar.error(f"**DG Ditolak (Rollover):** Anda membawa {total_kargo_dg} kontainer DG, tetapi kapal hanya memiliki {len(dg_zones_list)} DG Zone yang aman.")
        is_valid = False
        
    if total_kargo_dry > (total_hold_capacity - len(reefer_plugs_list) - len(dg_zones_list)) and is_valid:
         st.sidebar.warning("**Peringatan Kepadatan:** Slot reguler tidak cukup. Kargo DRY berisiko merampas area slot khusus!")

    if is_valid:
        st.sidebar.success("**Validasi Sukses:** Semua tipe peti kemas memenuhi kriteria kapasitas dan pembagian zona kapal.")

    if is_valid and st.sidebar.button("Optimasi Susunan Kargo"):
        with st.spinner("Mengeksekusi Pencarian Solusi Terbaik..."):
            optimizer = StowageGA(vessel_info, cargo_df, pop_size, generations, mutation_rate)
            best_layout, final_fit, metrics, fit_history = optimizer.run()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Fuzzy Fitness Score", f"{final_fit:.2f} / 100")
        col_m2.metric("Skor Kestabilan Kapal", f"{metrics['stability']:.2f}%")
        col_m3.metric("Pelanggaran Overstowage", f"{metrics['overstowage']} Kasus")
        col_m4.metric("Pelanggaran Slot Khusus", f"{metrics['slot_error']} Unit")

        tab1, tab2, tab3 = st.tabs(["Grafik Visualisasi 3D", "Grafik Progress GA", "Data Hasil Optimasi"])

        with tab1:
            st.write("Denah Digital Tata Letak Kubus Kontainer Palka Tiga Dimensi")
            fig = go.Figure()

            color_map = {1: "#2ecc71", 2: "#2980b9", 3: "#8e44ad"}

            for coord, item in best_layout.items():
                b, r, t = coord
                dest_seq = int(item['Urutan Pelabuhan'])
                base_color = color_map.get(dest_seq, "#7f8c8d")
                
                display_text = item['Container ID']
                if item['Tipe Cargo'].upper() == 'REEFER': display_text += "\n[R]"
                elif item['Tipe Cargo'].upper() == 'DG': display_text += "\n[DG]"

                hover_info = (f"<b>Kargo: {item['Container ID']}</b><br>"
                              f"Dimensi Size: {item['Size Type']}<br>"
                              f"Berat: {item['Berat (Ton)']} Ton<br>"
                              f"Jenis: {item['Tipe Cargo']}<br>"
                              f"Tujuan Bongkar: {item['Rute Pelabuhan']}<br>"
                              f"Koordinat Posisi: (Bay {b}, Row {r}, Tier {t})")

                draw_3d_cube(fig, b, r, t, base_color, hover_info, display_text)

            legend_drawn_reefer = False
            for rx, ry, rz in optimizer.reefer_plugs:
                draw_wireframe_outline(fig, rx, ry, rz, "yellow", "Zona Listrik Reefer Plugs", not legend_drawn_reefer)
                legend_drawn_reefer = True

            legend_drawn_dg = False
            for dx, dy, dz in optimizer.dg_zones:
                draw_wireframe_outline(fig, dx, dy, dz, "orange", "Zona Aman Dangerous Goods", not legend_drawn_dg)
                legend_drawn_dg = True

            fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    xaxis=dict(title='Bay (Panjang Kapal)', tickvals=list(range(optimizer.max_b)), gridcolor='lightgray'),
                    yaxis=dict(title='Row (Lebar Kiri-Kanan)', tickvals=list(range(optimizer.max_r)), gridcolor='lightgray'),
                    zaxis=dict(title='Tier (Tinggi Tumpukan)', tickvals=list(range(optimizer.max_t)), gridcolor='lightgray')
                ),
                margin=dict(l=0, r=0, b=0, t=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Petunjuk Navigasi: Klik + tahan mouse kiri untuk memutar (Rotate), mouse kanan untuk menggeser, dan gunakan scroll untuk memperbesar kubus (Zoom).")

        with tab2:
            st.write("Kurva Kemajuan Fitness Score Genetika Berdasarkan Generasi")
            history_df = pd.DataFrame({"Generasi": list(range(1, len(fit_history)+1)), "Fitness Score": fit_history})
            st.line_chart(data=history_df, x="Generasi", y="Fitness Score")

        with tab3:
            st.write("Manifest Hasil Penempatan Stowage Plan Teroptimal")
            export_data = []
            for coord, item in best_layout.items():
                export_data.append({
                    "Slot (Bay,Row,Tier)": str(coord),
                    "Container ID": item["Container ID"],
                    "Size Type": item["Size Type"],
                    "Berat (Ton)": item["Berat (Ton)"],
                    "Tujuan": item["Rute Pelabuhan"],
                    "Tipe Cargo": item["Tipe Cargo"]
                })
            res_df = pd.DataFrame(export_data)
            st.dataframe(res_df, use_container_width=True)
            
            csv_file = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Laporan Stowage Plan (.CSV)", data=csv_file, file_name=f"Stowage_Plan_{selected_vessel_name}.csv", mime="text/csv")
else:
    st.info("Silakan pilih Kapal di panel kiri, lalu unggah file kargo dummy Anda untuk memetakan susunan kubus palka kapal otomatis.")
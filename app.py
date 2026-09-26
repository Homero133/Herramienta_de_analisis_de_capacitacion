import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import os
import glob

# ==========================================
# MOTOR DE DATOS (Antes data_engine.py)
# ==========================================

def load_uploaded_data(forms_file, ventas_file):
    if forms_file.name.endswith('.csv'):
        df_forms = pd.read_csv(forms_file)
    else:
        df_forms = pd.read_excel(forms_file)
        
    if ventas_file.name.endswith('.csv'):
        df_ventas = pd.read_csv(ventas_file, low_memory=False)
    else:
        df_ventas = pd.read_excel(ventas_file)
        
    return df_forms, df_ventas

def process_data(df_forms, df_ventas, target_date, window_days):
    target_date = pd.to_datetime(target_date)
    
    def safe_col(df, idx):
        if idx < len(df.columns):
            return df.columns[idx]
        return None

    c_region = safe_col(df_ventas, 2)
    c_tienda = safe_col(df_ventas, 9)
    c_canal = safe_col(df_ventas, 12) # Store Type / Channel
    c_trainer = safe_col(df_ventas, 26)
    c_id_sup = safe_col(df_ventas, 27)
    c_nom_sup = safe_col(df_ventas, 28)
    c_puesto = safe_col(df_ventas, 29)
    c_id_prom = safe_col(df_ventas, 30)
    c_nombre = safe_col(df_ventas, 31)
    c_modelo = safe_col(df_ventas, 37)
    c_fecha = safe_col(df_ventas, 51)

    c_forms_id = safe_col(df_forms, 2)
    
    df_ventas['fecha_parsed'] = pd.to_datetime(df_ventas[c_fecha], errors='coerce')
    
    df_ventas[c_id_prom] = df_ventas[c_id_prom].astype(str).str.strip().str.upper()
    df_ventas[c_id_sup] = df_ventas[c_id_sup].astype(str).str.strip().str.upper()
    
    df_ventas = df_ventas[df_ventas[c_id_prom] != 'NAN']
    df_ventas = df_ventas[df_ventas[c_id_prom] != '']
    
    mask_promotor = df_ventas[c_puesto].astype(str).str.contains('Promotor|PC|Farmer', case=False, na=False)
    df_promotores = df_ventas[mask_promotor].copy()
    
    if len(df_promotores) == 0:
        df_promotores = df_ventas.copy()
    
    supervisores_ids = df_ventas[c_id_sup].unique()
    df_promotores = df_promotores[~df_promotores[c_id_prom].isin(supervisores_ids)]
    
    df_forms[c_forms_id] = df_forms[c_forms_id].astype(str).str.strip().str.upper()
    trained_ids = df_forms[c_forms_id].unique()
    
    c_foto = None
    for col in df_forms.columns:
        if 'foto' in str(col).lower() or 'imagen' in str(col).lower():
            c_foto = col
            break
            
    if c_foto:
        trained_with_tool = df_forms[df_forms[c_foto].notna()][c_forms_id].unique()
    else:
        trained_with_tool = []
    
    master = df_promotores[[c_id_prom, c_nombre, c_region, c_tienda, c_canal, c_trainer, c_nom_sup]].drop_duplicates(subset=[c_id_prom]).copy()
    master['Estatus_Capacitacion'] = np.where(master[c_id_prom].isin(trained_ids), 'Capacitado', 'No Capacitado')
    
    if c_foto:
        master['Tiene_Herramienta'] = np.where(master[c_id_prom].isin(trained_with_tool), 'Con Herramienta (Equipo)', 
                                      np.where(master[c_id_prom].isin(trained_ids), 'Sin Herramienta', 'Control (No Cap)'))
    else:
        master['Tiene_Herramienta'] = 'N/A'
    
    pre_mask = (df_promotores['fecha_parsed'] >= (target_date - pd.Timedelta(days=window_days))) & (df_promotores['fecha_parsed'] < target_date)
    post_mask = (df_promotores['fecha_parsed'] > target_date) & (df_promotores['fecha_parsed'] <= (target_date + pd.Timedelta(days=window_days)))
    reno_mask = df_promotores[c_modelo].astype(str).str.contains('Reno 16', case=False, na=False)
    
    pre_reno = df_promotores[pre_mask & reno_mask].groupby(c_id_prom).size().reset_index(name='Ventas_Pre_Reno16')
    post_reno = df_promotores[post_mask & reno_mask].groupby(c_id_prom).size().reset_index(name='Ventas_Post_Reno16')
    
    pre_total = df_promotores[pre_mask].groupby(c_id_prom).size().reset_index(name='Ventas_Pre_Total')
    post_total = df_promotores[post_mask].groupby(c_id_prom).size().reset_index(name='Ventas_Post_Total')
    
    # Total Histórico Reno 16 sin importar fechas
    total_reno = df_promotores[reno_mask].groupby(c_id_prom).size().reset_index(name='Ventas_Totales_Reno16')
    
    master = master.merge(pre_reno, on=c_id_prom, how='left') \
                   .merge(post_reno, on=c_id_prom, how='left') \
                   .merge(pre_total, on=c_id_prom, how='left') \
                   .merge(post_total, on=c_id_prom, how='left') \
                   .merge(total_reno, on=c_id_prom, how='left')
                   
    numeric_cols = ['Ventas_Pre_Reno16', 'Ventas_Post_Reno16', 'Ventas_Pre_Total', 'Ventas_Post_Total', 'Ventas_Totales_Reno16']
    for col in numeric_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0)
    # Extraer las ventas válidas en el periodo de tiempo para la línea de tiempo
    mask_timeline = (pre_mask | post_mask) & reno_mask
    df_ventas_reno = df_promotores[mask_timeline][[c_id_prom, 'fecha_parsed']].copy()
    
    return master, c_region, c_canal, c_trainer, df_ventas_reno

def process_all_capsules(archivos_forms, archivo_ventas):
    if archivo_ventas.name.endswith('.csv'):
        df_ventas = pd.read_csv(archivo_ventas, low_memory=False)
    else:
        df_ventas = pd.read_excel(archivo_ventas)
        
    def safe_col(df, idx):
        if idx < len(df.columns):
            return df.columns[idx]
        return None
        
    c_region = safe_col(df_ventas, 2)
    c_id_sup = safe_col(df_ventas, 27)
    c_nom_sup = safe_col(df_ventas, 28)
    c_puesto = safe_col(df_ventas, 29)
    c_id_prom = safe_col(df_ventas, 30)
    c_nombre = safe_col(df_ventas, 31)
    c_modelo = safe_col(df_ventas, 37)
    
    df_ventas[c_id_prom] = df_ventas[c_id_prom].astype(str).str.strip().str.upper()
    df_ventas[c_id_sup] = df_ventas[c_id_sup].astype(str).str.strip().str.upper()
    df_ventas = df_ventas[(df_ventas[c_id_prom] != 'NAN') & (df_ventas[c_id_prom] != '')]
    
    mask_promotor = df_ventas[c_puesto].astype(str).str.contains('Promotor|PC|Farmer', case=False, na=False)
    df_promotores = df_ventas[mask_promotor].copy()
    if len(df_promotores) == 0:
        df_promotores = df_ventas.copy()
        
    supervisores_ids = df_ventas[c_id_sup].unique()
    df_promotores = df_promotores[~df_promotores[c_id_prom].isin(supervisores_ids)]
    
    master = df_promotores[[c_id_prom, c_nombre, c_region, c_nom_sup]].drop_duplicates(subset=[c_id_prom]).copy()
    
    reno_mask = df_promotores[c_modelo].astype(str).str.contains('Reno 16', case=False, na=False)
    total_reno = df_promotores[reno_mask].groupby(c_id_prom).size().reset_index(name='Ventas_Totales_Reno16')
    master = master.merge(total_reno, on=c_id_prom, how='left')
    master['Ventas_Totales_Reno16'] = master['Ventas_Totales_Reno16'].fillna(0).astype(int)
    
    # Procesar todos los forms
    dict_capacitaciones = {} # {id_prom: [capsula1, capsula2]}
    for form_file in archivos_forms:
        if form_file.name.endswith('.csv'):
            df_f = pd.read_csv(form_file)
        else:
            df_f = pd.read_excel(form_file)
        c_f_id = safe_col(df_f, 2)
        trained_ids = df_f[c_f_id].astype(str).str.strip().str.upper().unique()
        cap_name = form_file.name.replace('.xlsx','').replace('.csv','')
        for pid in trained_ids:
            if pid not in dict_capacitaciones:
                dict_capacitaciones[pid] = []
            dict_capacitaciones[pid].append(cap_name)
            
    master['Capsulas_Tomadas'] = master[c_id_prom].apply(lambda x: len(dict_capacitaciones.get(x, [])))
    master['Nombres_Capsulas'] = master[c_id_prom].apply(lambda x: ", ".join(dict_capacitaciones.get(x, [])))
    master['Nivel_Capacitacion'] = master['Capsulas_Tomadas'].apply(lambda x: f"{x} Cápsula{'s' if x != 1 else ''}" if x > 0 else "Ninguna (0)")
    
    return master, df_ventas, c_region, c_nom_sup

st.set_page_config(page_title="Impacto Capacitación - OPPO Reno 16", layout="wide")

st.title("📊 Medición de Impacto: Capacitación vs Ventas (Reno 16)")
st.markdown("Esta herramienta utiliza la metodología **Diferencias en Diferencias (Diff-in-Diff)** para medir el impacto real de las capacitaciones, aislando factores externos.")

st.sidebar.markdown("📂 **Carga de Archivos**\nSube tus reportes de ventas y formularios aquí.")
archivo_ventas = st.sidebar.file_uploader("1️⃣ Sube tu archivo Mariana (Ventas)", type=["csv", "xlsx"])
archivos_forms = st.sidebar.file_uploader("2️⃣ Sube las respuestas de Forms (Cápsulas)", type=["csv", "xlsx"], accept_multiple_files=True)

capsula_seleccionada = None
archivo_forms_seleccionado = None

if archivos_forms:
    nombres_capsulas = [f.name for f in archivos_forms]
    if len(archivos_forms) > 1:
        nombres_capsulas.insert(0, "Todas las Cápsulas (Impacto Global)")
        
    capsula_seleccionada = st.sidebar.selectbox("Selecciona la Cápsula a analizar", nombres_capsulas)
    if capsula_seleccionada != "Todas las Cápsulas (Impacto Global)":
        archivo_forms_seleccionado = next((f for f in archivos_forms if f.name == capsula_seleccionada), None)
        target_date = st.sidebar.date_input("Fecha de la Capacitación", datetime.date(2026, 8, 21))
        window_days = st.sidebar.number_input("Ventana de Análisis (Días)", min_value=1, max_value=30, value=7)
    else:
        st.sidebar.info("Modo de Impacto Global Activado. Se evaluarán los niveles de capacitación de todos los promotores.")

if st.sidebar.button("Procesar y Analizar Datos"):
    if not archivo_ventas:
        st.sidebar.error("Falta subir el archivo de Ventas (Mariana).")
    elif capsula_seleccionada == "Todas las Cápsulas (Impacto Global)":
        with st.spinner("Procesando análisis global (Multi-cápsula)..."):
            try:
                master_multi, df_ventas, col_region, col_sup = process_all_capsules(archivos_forms, archivo_ventas)
                st.session_state['mode'] = 'multi'
                st.session_state['multi_master'] = master_multi
                st.session_state['df_ventas'] = df_ventas
                st.session_state['col_region'] = col_region
                st.session_state['col_sup'] = col_sup
                st.success("✅ Análisis multi-cápsula generado con éxito.")
            except Exception as e:
                st.error(f"Error procesando los datos: {e}")
    else:
        if not archivo_forms_seleccionado:
            st.sidebar.error("Falta seleccionar un archivo de cápsula.")
        else:
            with st.spinner(f"Cruzando {capsula_seleccionada} con la base de ventas..."):
                try:
                    df_forms, df_ventas = load_uploaded_data(archivo_forms_seleccionado, archivo_ventas)
                    master_full, col_region, col_canal, col_trainer, df_ventas_reno = process_data(df_forms, df_ventas, target_date, window_days)
                
                    st.session_state['mode'] = 'single'
                    st.session_state['master_data'] = master_full
                    st.session_state['col_region'] = col_region
                    st.session_state['col_canal'] = col_canal
                    st.session_state['col_trainer'] = col_trainer
                    st.session_state['df_forms'] = df_forms
                    st.session_state['df_ventas'] = df_ventas
                    st.session_state['df_ventas_reno'] = df_ventas_reno
                    st.success("✅ Datos procesados con éxito.")
                    
                except Exception as e:
                    st.error(f"Error procesando los datos: {e}")

mode = st.session_state.get('mode')

if mode == 'multi' and 'multi_master' in st.session_state:
    master_multi = st.session_state['multi_master'].copy()
    col_region = st.session_state['col_region']
    c_nom_sup = st.session_state['col_sup']
    
    st.header("🌍 Impacto Global Acumulado (Multi-Cápsula)")
    st.markdown("Visualiza el volumen de ventas totales de Reno 16 dependiendo del número de capacitaciones tomadas por los promotores.")
    
    totales = len(master_multi)
    promotores_cap = len(master_multi[master_multi['Capsulas_Tomadas'] > 0])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Promotores Totales", f"{totales}")
    c2.metric("Promotores Capacitados (Al menos 1)", f"{promotores_cap}")
    c3.metric("Promotores Sin Capacitación", f"{totales - promotores_cap}")
    
    st.markdown("---")
    avg_sales = master_multi.groupby('Nivel_Capacitacion')['Ventas_Totales_Reno16'].mean().reset_index()
    # Sort logically so Ninguna is first
    def sort_key(val):
        if "Ninguna" in val: return 0
        try: return int(val.split()[0])
        except: return 99
    
    avg_sales['SortKey'] = avg_sales['Nivel_Capacitacion'].apply(sort_key)
    avg_sales = avg_sales.sort_values(by='SortKey').drop(columns=['SortKey'])
    
    fig_multi = px.bar(avg_sales, x='Nivel_Capacitacion', y='Ventas_Totales_Reno16', 
                       title="Promedio de Ventas Totales (Reno 16) por Nivel de Capacitación",
                       text_auto='.1f', color='Nivel_Capacitacion')
    st.plotly_chart(fig_multi, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Adopción de Capacitación por Región")
    if col_region in master_multi.columns:
        region_mix = master_multi.groupby([col_region, 'Nivel_Capacitacion']).size().reset_index(name='Promotores')
        region_mix['Porcentaje'] = region_mix.groupby(col_region)['Promotores'].transform(lambda x: x / x.sum() * 100)
        region_mix['SortKey'] = region_mix['Nivel_Capacitacion'].apply(sort_key)
        region_mix = region_mix.sort_values(by=['SortKey'])
        
        fig_mix = px.bar(region_mix, x=col_region, y='Porcentaje', color='Nivel_Capacitacion',
                         title="Distribución de Niveles de Capacitación por Región (%)",
                         text=region_mix['Porcentaje'].apply(lambda x: f'{x:.1f}%'))
        fig_mix.update_layout(barmode='stack')
        st.plotly_chart(fig_mix, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📋 Tabla de Asistencia y Ventas por Supervisor")
    
    if c_nom_sup and c_nom_sup in master_multi.columns and col_region in master_multi.columns:
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            lista_reg = ["Todas"] + sorted([str(x) for x in master_multi[col_region].dropna().unique()])
            reg_sel = st.selectbox("Filtrar por Región:", lista_reg)
            
        df_show = master_multi.copy()
        if reg_sel != "Todas":
            df_show = df_show[df_show[col_region] == reg_sel]
            
        with col_f2:
            lista_supervisores = ["Todos"] + sorted([str(x) for x in df_show[c_nom_sup].dropna().unique()])
            sup_sel = st.selectbox("Filtrar por Supervisor:", lista_supervisores)
            
        if sup_sel != "Todos":
            df_show = df_show[df_show[c_nom_sup] == sup_sel]
            
        c_id = master_multi.columns[0]
        c_nombre = master_multi.columns[1]
        
        df_display = df_show[[c_id, c_nombre, col_region, c_nom_sup, 'Nivel_Capacitacion', 'Nombres_Capsulas', 'Ventas_Totales_Reno16']]
        df_display.columns = ['ID Promotor', 'Nombre', 'Región', 'Supervisor', 'Nivel (Cantidad)', 'Cápsulas Tomadas', 'Ventas Totales (Reno 16)']
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)

elif mode == 'single' and 'master_data' in st.session_state:
    master = st.session_state['master_data'].copy()
    col_region = st.session_state['col_region']
    col_canal = st.session_state['col_canal']
    col_trainer = st.session_state['col_trainer']
    df_ventas = st.session_state['df_ventas']
    df_ventas_reno = st.session_state['df_ventas_reno']
    
    # Extraemos el nombre de la columna Sub-Región / Distrito (Índice 3 es 'District' en el reporte de Mariana)
    c_distrito = df_ventas.columns[3] if len(df_ventas.columns) > 3 else "Distrito"
    
    distritos_map = df_ventas[[df_ventas.columns[30], c_distrito]].drop_duplicates(subset=[df_ventas.columns[30]])
    master = master.merge(distritos_map, on=df_ventas.columns[30], how='left')
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtros de Segmentación")
    
    lista_regiones = ["Todas"] + list(master[col_region].dropna().unique())
    region_sel = st.sidebar.selectbox("Filtrar por Región (Province)", lista_regiones)
    if region_sel != "Todas":
        master = master[master[col_region] == region_sel]
        
    lista_distritos = ["Todos"] + list(master[c_distrito].dropna().unique())
    distrito_sel = st.sidebar.selectbox("Filtrar por Zona/Sub-Región (District)", lista_distritos)
    if distrito_sel != "Todos":
        master = master[master[c_distrito] == distrito_sel]
        
    if col_canal and col_canal in master.columns:
        lista_canales = ["Todos"] + list(master[col_canal].dropna().unique())
        canal_sel = st.sidebar.selectbox("Filtrar por Canal (Store Type)", lista_canales)
        if canal_sel != "Todos":
            master = master[master[col_canal] == canal_sel]
            
    col_id = master.columns[0]
    lista_promotores = ["Todos"] + sorted([str(x) for x in master[col_id].dropna().unique()])
    promotor_sel = st.sidebar.selectbox("Filtrar por Promotor (ID - Visitas F2F)", lista_promotores)
    if promotor_sel != "Todos":
        master = master[master[col_id] == promotor_sel]

    # --- KPIS ---
    total_activos = len(master)
    capacitados = len(master[master['Estatus_Capacitacion'] == 'Capacitado'])
    cobertura = (capacitados / total_activos) * 100 if total_activos > 0 else 0
    
    cap_pre = master[master['Estatus_Capacitacion'] == 'Capacitado']['Ventas_Pre_Reno16'].mean()
    cap_post = master[master['Estatus_Capacitacion'] == 'Capacitado']['Ventas_Post_Reno16'].mean()
    
    nocap_pre = master[master['Estatus_Capacitacion'] == 'No Capacitado']['Ventas_Pre_Reno16'].mean()
    nocap_post = master[master['Estatus_Capacitacion'] == 'No Capacitado']['Ventas_Post_Reno16'].mean()
    
    cap_pre = 0 if pd.isna(cap_pre) else cap_pre
    cap_post = 0 if pd.isna(cap_post) else cap_post
    nocap_pre = 0 if pd.isna(nocap_pre) else nocap_pre
    nocap_post = 0 if pd.isna(nocap_post) else nocap_post
    
    lift_neto = (cap_post - cap_pre) - (nocap_post - nocap_pre)
    
    # Delta % Capacitados vs Control
    delta_pct_cap = ((cap_post - cap_pre) / cap_pre * 100) if cap_pre > 0 else 0
    delta_pct_nocap = ((nocap_post - nocap_pre) / nocap_pre * 100) if nocap_pre > 0 else 0
    
    ventas_reno_cap_post = master[master['Estatus_Capacitacion'] == 'Capacitado']['Ventas_Post_Reno16'].sum()
    ventas_tot_cap_post = master[master['Estatus_Capacitacion'] == 'Capacitado']['Ventas_Post_Total'].sum()
    mix_reno = (ventas_reno_cap_post / ventas_tot_cap_post * 100) if ventas_tot_cap_post > 0 else 0
    
    # Mostrar KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Promotores Activos", f"{total_activos}")
    col2.metric("Cobertura Capacitación", f"{cobertura:.1f}%")
    col3.metric("Lift Neto (Unidades)", f"{lift_neto:.2f}", f"Δ% Cap: {delta_pct_cap:.1f}% | Δ% Ctrl: {delta_pct_nocap:.1f}%")
    col4.metric("Mix Reno 16 (Post-Curso)", f"{mix_reno:.1f}%")
    
    # --- DEPURACIÓN ---
    with st.expander("🛠️ Depuración de Datos (Ver si algo falla)"):
        st.write(f"Filas en Formulario (Capsula 3): {len(st.session_state['df_forms'])}")
        st.write(f"Filas en Reporte Ventas (Mariana): {len(df_ventas)}")
    
    # --- GRÁFICOS PRINCIPALES ---
    st.markdown("---")
    st.markdown("### 📈 Impacto de Capacitación: Grupo Capacitado vs Control")
    
    # Preparar datos para gráfica de barras
    plot_data = pd.DataFrame({
        "Grupo": ["Capacitados", "Capacitados", "No Capacitados", "No Capacitados"],
        "Periodo": ["Pre", "Post", "Pre", "Post"],
        "Unidades Promedio": [round(cap_pre,2), round(cap_post,2), round(nocap_pre,2), round(nocap_post,2)]
    })
    fig_bar = px.bar(plot_data, x="Grupo", y="Unidades Promedio", color="Periodo", barmode="group",
                  title="Comparativa Pre vs Post por Grupo (Reno 16)", text_auto=True)
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # --- GRÁFICO: IMPACTO HERRAMIENTA ---
    has_tool = 'Tiene_Herramienta' in master.columns and (master['Tiene_Herramienta'] != 'N/A').any()
    
    if has_tool:
        st.markdown("### 🛠️ Impacto de Herramienta (Con Equipo vs Sin Equipo)")
        
        tool_pre = master[master['Tiene_Herramienta'] == 'Con Herramienta (Equipo)']['Ventas_Pre_Reno16'].mean()
        tool_post = master[master['Tiene_Herramienta'] == 'Con Herramienta (Equipo)']['Ventas_Post_Reno16'].mean()
        notool_pre = master[master['Tiene_Herramienta'] == 'Sin Herramienta']['Ventas_Pre_Reno16'].mean()
        notool_post = master[master['Tiene_Herramienta'] == 'Sin Herramienta']['Ventas_Post_Reno16'].mean()
        
        tool_pre = 0 if pd.isna(tool_pre) else tool_pre
        tool_post = 0 if pd.isna(tool_post) else tool_post
        notool_pre = 0 if pd.isna(notool_pre) else notool_pre
        notool_post = 0 if pd.isna(notool_post) else notool_post
        
        plot_data_tool = pd.DataFrame({
            "Grupo": ["Con Herramienta", "Con Herramienta", "Sin Herramienta", "Sin Herramienta"],
            "Periodo": ["Pre", "Post", "Pre", "Post"],
            "Unidades Promedio": [round(tool_pre,2), round(tool_post,2), round(notool_pre,2), round(notool_post,2)]
        })
        fig_tool = px.bar(plot_data_tool, x="Grupo", y="Unidades Promedio", color="Periodo", barmode="group",
                      title="Comparativa Pre vs Post por Tenencia de Herramienta", text_auto=True, color_discrete_sequence=['#FF7F0E', '#FFB347'])
        st.plotly_chart(fig_tool, use_container_width=True)
    # --- RENDIMIENTO POR PROMOCIÓN (TOP / BOTTOM) Y REGIÓN ---
    st.markdown("---")
    st.markdown("### 🚀 Ranking de Desempeño y Regiones (Grupo Capacitado)")
    
    master_cap = master[master['Estatus_Capacitacion'] == 'Capacitado'].copy()
    master_cap['Lift_Individual'] = master_cap['Ventas_Post_Reno16'] - master_cap['Ventas_Pre_Reno16']
    
    c_rank1, c_rank2 = st.columns(2)
    
    with c_rank1:
        # Gráfica Divergente Top 5 / Bottom 5
        if len(master_cap) > 0:
            top5 = master_cap.nlargest(5, 'Lift_Individual')
            bottom5 = master_cap.nsmallest(5, 'Lift_Individual')
            df_diverge = pd.concat([bottom5, top5]).drop_duplicates()
            df_diverge['Color'] = np.where(df_diverge['Lift_Individual'] >= 0, 'green', 'red')
            
            fig_div = px.bar(df_diverge, x='Lift_Individual', y=master.columns[1], orientation='h',
                             title="Top 5 vs Bottom 5 Promotores (Lift Individual)",
                             color='Color', color_discrete_map={'green':'#00cc66', 'red':'#ff4b4b'}, text_auto=True)
            fig_div.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_div, use_container_width=True)
    
    with c_rank2:
        if region_sel != "Todas":
            lift_sub = master_cap.groupby(c_distrito)['Lift_Individual'].mean().reset_index()
            fig2 = px.bar(lift_sub, x='Lift_Individual', y=c_distrito, orientation='h', title="Lift Promedio por Sub-Región", text_auto='.2f')
        else:
            lift_region = master_cap.groupby(col_region)['Lift_Individual'].mean().reset_index()
            fig2 = px.bar(lift_region, x='Lift_Individual', y=col_region, orientation='h', title="Lift Promedio por Región", text_auto='.2f')
        st.plotly_chart(fig2, use_container_width=True)
    
    # --- TABLA MAESTRA Y EXPORTACIÓN ---
    st.markdown("---")
    st.markdown("### 📋 Tabla Maestra de Resultados")
    
    # Botón de exportación a CSV
    csv = master.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Tabla Maestra (CSV)",
        data=csv,
        file_name='medicion_impacto_reno16.csv',
        mime='text/csv',
    )
    
    # Columnas reorganizadas y mostradas
    cols_to_show = [master.columns[0], master.columns[1], col_region, c_distrito, col_canal, 'Estatus_Capacitacion', 'Ventas_Pre_Reno16', 'Ventas_Post_Reno16']
    cols_to_show = [c for c in cols_to_show if c in master.columns]
    
    # Dataframe con sticky headers/cols soportados nativamente por st.dataframe
    st.dataframe(master[cols_to_show], use_container_width=True, hide_index=True)
    
    # --- CONTROL DE ASISTENCIA POR SUPERVISOR ---
    st.markdown("---")
    st.markdown("### 👥 Control de Asistencia por Supervisor")
    st.markdown("Selecciona un supervisor para ver qué promotores tomaron la capacitación y quiénes faltaron.")
    
    # Obtener el nombre de la columna del supervisor de manera segura
    c_nom_sup_name = st.session_state['df_ventas'].columns[28] if len(st.session_state['df_ventas'].columns) > 28 else None
    
    if c_nom_sup_name and c_nom_sup_name in master.columns:
        lista_supervisores = ["Selecciona un supervisor..."] + sorted([str(x) for x in master[c_nom_sup_name].dropna().unique()])
        
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            sup_sel = st.selectbox("Buscar Supervisor:", lista_supervisores)
        with col_s2:
            st.write("")
            st.write("")
            mostrar_ventas = st.checkbox("Mostrar Ventas (Pre/Post)", value=True)
            
        if sup_sel != "Selecciona un supervisor...":
            df_sup = master[master[c_nom_sup_name] == sup_sel].copy()
            
            c_id_prom = master.columns[0]
            c_nombre = master.columns[1]
            
            if mostrar_ventas:
                df_display = df_sup[[c_id_prom, c_nombre, 'Estatus_Capacitacion', 'Ventas_Pre_Reno16', 'Ventas_Post_Reno16']].copy()
                df_display.columns = ['ID Promotor', 'Nombre', 'Asistencia', 'Ventas Pre (Reno 16)', 'Ventas Post (Reno 16)']
                df_display['Ventas Pre (Reno 16)'] = df_display['Ventas Pre (Reno 16)'].astype(int)
                df_display['Ventas Post (Reno 16)'] = df_display['Ventas Post (Reno 16)'].astype(int)
            else:
                df_display = df_sup[[c_id_prom, c_nombre, 'Estatus_Capacitacion']].copy()
                df_display.columns = ['ID Promotor', 'Nombre', 'Asistencia']
            
            def highlight_status(val):
                if val == 'No Capacitado':
                    return 'color: #ff4b4b; font-weight: bold;'
                elif val == 'Capacitado':
                    return 'color: #00cc66; font-weight: bold;'
                return ''
            
            try:
                styled_df = df_display.style.map(highlight_status, subset=['Asistencia'])
            except AttributeError:
                styled_df = df_display.style.applymap(highlight_status, subset=['Asistencia'])
                
            st.dataframe(styled_df, use_container_width=True)
            
            # --- EXPLORADOR DE PROMOTOR ---
            st.markdown("#### 🔍 Explorador de Promotor (Detalles y Examen)")
            
            # Create a combined ID - Nombre list for easier selection
            df_sup['ID_Nombre'] = df_sup[c_id_prom].astype(str) + " - " + df_sup[c_nombre].astype(str)
            lista_prom_sup = ["Selecciona un promotor..."] + list(df_sup['ID_Nombre'].dropna().unique())
            
            prom_selec = st.selectbox("Ver detalles individuales:", lista_prom_sup)
            
            if prom_selec != "Selecciona un promotor...":
                # Obtener el ID del promotor seleccionado
                id_seleccionado = prom_selec.split(" - ")[0].strip()
                
                c_perfil, c_examen = st.columns(2)
                
                with c_perfil:
                    st.info(f"**Perfil: {prom_selec}**")
                    # Buscar fecha de ingreso dinámica
                    col_ingreso = None
                    for c in df_ventas.columns:
                        if 'ingreso' in str(c).lower() or 'hire' in str(c).lower() or 'alta' in str(c).lower():
                            col_ingreso = c
                            break
                    if col_ingreso:
                        val_ingreso = df_ventas[df_ventas[df_ventas.columns[30]] == id_seleccionado][col_ingreso].values
                        if len(val_ingreso) > 0 and pd.notna(val_ingreso[0]):
                            # Tratar de formatear si es fecha
                            try:
                                fecha_fmt = pd.to_datetime(val_ingreso[0]).strftime('%Y-%m-%d')
                                st.write(f"**Fecha de Ingreso:** {fecha_fmt}")
                            except:
                                st.write(f"**Fecha de Ingreso:** {val_ingreso[0]}")
                        else:
                            st.write("**Fecha de Ingreso:** Sin dato registrado")
                    else:
                        st.write("**Fecha de Ingreso:** Columna no detectada en el reporte.")
                        
                    # Ventas Históricas
                    ventas_tot = master[master[master.columns[0]] == id_seleccionado]['Ventas_Totales_Reno16'].values
                    if len(ventas_tot) > 0:
                        st.write(f"**Ventas Históricas (Reno 16):** {int(ventas_tot[0])} unidades")
                    
                    estatus_cap = master[master[master.columns[0]] == id_seleccionado]['Estatus_Capacitacion'].values[0]
                    color_estatus = "green" if estatus_cap == "Capacitado" else "red"
                    st.markdown(f"**Estatus:** :{color_estatus}[{estatus_cap}]")

                with c_examen:
                    st.info("**Respuestas de Capacitación (Cápsula 3)**")
                    c_forms_id = st.session_state['df_forms'].columns[2]
                    respuestas = st.session_state['df_forms'][st.session_state['df_forms'][c_forms_id].astype(str).str.strip().str.upper() == id_seleccionado.upper()]
                    
                    if not respuestas.empty:
                        with st.container(height=300):
                            dict_resp = respuestas.iloc[0].dropna().to_dict()
                            for k, v in dict_resp.items():
                                if 'Timestamp' in str(k) or 'Puntuación' in str(k) or 'Nombre' in str(k) or 'NOMBRE' in str(k) or 'ID' in str(k) or 'Correo' in str(k) or 'temporal' in str(k).lower():
                                    continue
                                
                                if 'drive.google.com' in str(v):
                                    st.markdown(f"**{k}:** [🔗 Abrir en Google Drive]({v})")
                                    nombre_prom = prom_selec.split(" - ")[1].strip() if " - " in prom_selec else id_seleccionado
                                    # Intentar buscar la foto localmente
                                    # La foto suele llamarse "* - Nombre Promotor.jpg"
                                    # Limitamos a un nombre más corto por si hay variaciones
                                    primer_nombre = nombre_prom.split(" ")[0]
                                    search_path = f"G:/Mi unidad/Real y Divertido  (File responses)/**/*{primer_nombre}*.jpg"
                                    found_photos = glob.glob(search_path, recursive=True)
                                    if found_photos:
                                        st.image(found_photos[0], use_container_width=True, caption=f"Foto local: {os.path.basename(found_photos[0])}")
                                else:
                                    st.markdown(f"_{k}_")
                                    st.markdown(f"👉 **{v}**")
                                    st.markdown("---")
                    else:
                        st.warning("Este promotor no tiene registro en el Google Forms (No Capacitado / Falta).")

    # --- DESEMPEÑO GLOBAL RENO 16 ---
    st.markdown("---")
    st.markdown("### 🏆 Desempeño Global (Reno 16)")
    st.markdown("Ventas totales acumuladas de la serie Reno 16 por promotor (histórico completo).")
    
    df_perf = master[[master.columns[0], master.columns[1], 'Estatus_Capacitacion', 'Ventas_Totales_Reno16']].copy()
    df_perf['Ventas_Totales_Reno16'] = df_perf['Ventas_Totales_Reno16'].astype(int)
    df_perf.columns = ['ID Promotor', 'Nombre', 'Asistencia', 'Ventas Totales (Reno 16)']
    df_perf = df_perf.sort_values(by='Ventas Totales (Reno 16)', ascending=False)
    
    def highlight_performance(val):
        if not isinstance(val, (int, float)):
            return ''
        if val == 0:
            color = '#ffcccc' # Soft Red
        elif 1 <= val <= 5:
            color = '#ffe6cc' # Soft Orange
        elif 6 <= val <= 10:
            color = '#ffffcc' # Soft Yellow
        else:
            color = '#ccffcc' # Soft Green
        return f'background-color: {color}; color: black; font-weight: 500;'
    
    try:
        styled_perf = df_perf.style.map(highlight_performance, subset=['Ventas Totales (Reno 16)'])
    except AttributeError:
        styled_perf = df_perf.style.applymap(highlight_performance, subset=['Ventas Totales (Reno 16)'])
        
    st.dataframe(styled_perf, use_container_width=True)

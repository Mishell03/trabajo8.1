import streamlit as st
import pulp
import pandas as pd
import time

st.title("Ejercicios de Programación Lineal")

# Menú para seleccionar el ejercicio
opcion = st.sidebar.selectbox(
    "Selecciona el ejercicio a resolver",
    [
        "Ejercicio 8.1: Método Branch and Bound",
        "Ejercicio 8.2: Comparación de tiempos de cálculo",
        "Ejercicio 8.3: Minimización con cortes de Gomory",
        "Ejercicio 8.4: Maximización con cortes de Gomory",
        "Ejercicio 8.5: Problema de asignación binaria para maximizar el NPV"
    ]
)

if opcion == "Ejercicio 8.1: Método Branch and Bound":
    st.write("""
    Resolver el problema de maximización usando el Método Branch and Bound:
    
    Maximizar: P(x1, x2, x3) = 4x1 + 3x2 + 3x3
    """)
    
    prob = pulp.LpProblem("Maximizar_P", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0, cat="Integer")
    x2 = pulp.LpVariable("x2", lowBound=0, cat="Integer")
    x3 = pulp.LpVariable("x3", lowBound=0, cat="Integer")
    prob += 4 * x1 + 3 * x2 + 3 * x3
    prob += 4 * x1 + 2 * x2 + x3 <= 10
    prob += 3 * x1 + 4 * x2 + 2 * x3 <= 14
    prob += 2 * x1 + x2 + 3 * x3 <= 7
    prob.solve()

    st.subheader("Resultados:")
    st.write(f"Estado del problema: {pulp.LpStatus[prob.status]}")
    st.write(f"Valor óptimo: {pulp.value(prob.objective)}")
    for v in prob.variables():
        st.write(f"{v.name} = {v.varValue}")

elif opcion == "Ejercicio 8.2: Comparación de tiempos de cálculo":
    st.write("Comparación del tiempo de cálculo entre el problema LP continuo y el problema LP con restricciones enteras.")

    prob_lp = pulp.LpProblem("Maximizar_P_LP", pulp.LpMaximize)
    x1, x2, x3 = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(1, 4)]
    prob_lp += 4 * x1 + 3 * x2 + 3 * x3
    prob_lp += 4 * x1 + 2 * x2 + x3 <= 10
    prob_lp += 3 * x1 + 4 * x2 + 2 * x3 <= 14
    prob_lp += 2 * x1 + x2 + 3 * x3 <= 7
    
    start_time = time.time()
    prob_lp.solve()
    tiempo_lp = time.time() - start_time
    
    st.write("Resultados para el problema LP continuo:")
    st.write(f"Tiempo de cálculo: {tiempo_lp:.4f} segundos")
    st.write(f"Valor óptimo LP: {pulp.value(prob_lp.objective)}")

    prob_entero = pulp.LpProblem("Maximizar_P", pulp.LpMaximize)
    x1, x2, x3 = [pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer") for i in range(1, 4)]
    prob_entero += 4 * x1 + 3 * x2 + 3 * x3
    prob_entero += 4 * x1 + 2 * x2 + x3 <= 10
    prob_entero += 3 * x1 + 4 * x2 + 2 * x3 <= 14
    prob_entero += 2 * x1 + x2 + 3 * x3 <= 7
    
    start_time = time.time()
    prob_entero.solve()
    tiempo_entero = time.time() - start_time
    
    st.write("Resultados para el problema LP con restricciones enteras:")
    st.write(f"Tiempo de cálculo: {tiempo_entero:.4f} segundos")
    st.write(f"Valor óptimo Entero: {pulp.value(prob_entero.objective)}")

elif opcion == "Ejercicio 8.3: Minimización con cortes de Gomory":
    st.write("Ejercicio de minimización usando cortes de Gomory.")

    prob_gomory = pulp.LpProblem("Minimizar_C", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, cat="Integer")
    y = pulp.LpVariable("y", lowBound=0, cat="Integer")
    prob_gomory += x - y
    prob_gomory += 3 * x + 4 * y <= 6
    prob_gomory += x - y <= 1
    prob_gomory.solve()
    
    st.write("Resultados iniciales:")
    st.write(f"Estado del problema: {pulp.LpStatus[prob_gomory.status]}")
    st.write(f"Valor óptimo: {pulp.value(prob_gomory.objective)}")
    for v in prob_gomory.variables():
        st.write(f"{v.name} = {v.varValue}")
    
    st.write("Proceso de cortes de Gomory no implementado automáticamente en esta herramienta.")

elif opcion == "Ejercicio 8.4: Maximización con cortes de Gomory":
    st.write("Ejercicio de maximización usando cortes de Gomory.")

    prob_gomory = pulp.LpProblem("Maximizar_P", pulp.LpMaximize)
    x1, x2, x3 = [pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer") for i in range(1, 4)]
    prob_gomory += 4 * x1 + 3 * x2 + 3 * x3
    prob_gomory += 4 * x1 + 2 * x2 + x3 <= 10
    prob_gomory += 3 * x1 + 4 * x2 + 2 * x3 <= 14
    prob_gomory += 2 * x1 + x2 + 3 * x3 <= 7
    prob_gomory.solve()
    
    st.write("Resultados iniciales:")
    st.write(f"Estado del problema: {pulp.LpStatus[prob_gomory.status]}")
    st.write(f"Valor óptimo: {pulp.value(prob_gomory.objective)}")
    for v in prob_gomory.variables():
        st.write(f"{v.name} = {v.varValue}")

elif opcion == "Ejercicio 8.5: Problema de asignación binaria para maximizar el NPV":
    st.write("Problema de asignación binaria para maximizar el NPV.")

    npv_proyectos = {"P1": 141, "P2": 187, "P3": 163, "P4": 153, "P5": 189, "P6": 127}
    costos_por_ano = {
        "P1": [75, 25, 20, 15, 10],
        "P2": [90, 50, 25, 15, 10],
        "P3": [80, 60, 25, 15, 15],
        "P4": [40, 20, 15, 10, 10],
        "P5": [100, 30, 20, 10, 10],
        "P6": [50, 20, 10, 10, 10]
    }
    presupuestos = [250, 75, 50, 50, 50]

    prob = pulp.LpProblem("Maximizar_NPV", pulp.LpMaximize)
    proyectos = {p: pulp.LpVariable(p, cat="Binary") for p in npv_proyectos}
    prob += pulp.lpSum(proyectos[p] * npv_proyectos[p] for p in npv_proyectos)
    
    for i, presupuesto in enumerate(presupuestos):
        prob += pulp.lpSum(proyectos[p] * costos_por_ano[p][i] for p in npv_proyectos) <= presupuesto
    
    prob.solve()

    st.write(f"Estado del problema: {pulp.LpStatus[prob.status]}")
    st.write(f"Valor óptimo de NPV: {pulp.value(prob.objective)}")
    for p in proyectos:
        st.write(f"{p}: {'Seleccionado' if proyectos[p].varValue == 1 else 'No seleccionado'}")

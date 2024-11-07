import streamlit as st
import pulp
import matplotlib.pyplot as plt
import numpy as np
import time

# Título de la aplicación
st.title("Ejercicios de Programación Lineal y Optimización")

# Menú de selección en la barra lateral
st.sidebar.title("UNIVERSIDAD NACIONAL DEL ALTIPLANO ")
st.sidebar.title("FACULTAD DE INGENIERÍA ESTADÍSTICA E INFORMÁTICA")
st.sidebar.header("Métodos De Optimización")
st.sidebar.header("By: Erika Mishelle Arapa Condori")
st.sidebar.header("Seleccione un ejercicio")
opciones = ["Ejercicio 8.1", "Ejercicio 8.2", "Ejercicio 8.3", "Ejercicio 8.4", "Ejercicio 8.5"]
seleccion = st.sidebar.radio("Elija el ejercicio:", opciones)

# Funciones para cada ejercicio
def ejercicio_8_1():
    st.subheader("Ejercicio 8.1: Método Branch and Bound")
    
    st.write("""
    Resolver el problema de maximización usando el Método Branch and Bound:

    Maximizar: P(x1, x2, x3) = 4x1 + 3x2 + 3x3

    Sujeto a:
    - 4x1 + 2x2 + x3 <= 10
    - 3x1 + 4x2 + 2x3 <= 14
    - 2x1 + x2 + 3x3 <= 7
    - x1, x2, x3 son enteros no negativos
    """)

    prob = pulp.LpProblem("Maximizar_P", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0, cat="Integer")
    x2 = pulp.LpVariable("x2", lowBound=0, cat="Integer")
    x3 = pulp.LpVariable("x3", lowBound=0, cat="Integer")
    
    prob += 4 * x1 + 3 * x2 + 3 * x3, "Función Objetivo"
    prob += 4 * x1 + 2 * x2 + x3 <= 10, "Restricción 1"
    prob += 3 * x1 + 4 * x2 + 2 * x3 <= 14, "Restricción 2"
    prob += 2 * x1 + x2 + 3 * x3 <= 7, "Restricción 3"
    prob.solve()

    estado = pulp.LpStatus[prob.status]
    valor_objetivo = pulp.value(prob.objective)
    valores_variables = {variable.name: variable.varValue for variable in prob.variables()}

    st.write(f"Estado del problema: {estado}")
    st.write(f"Valor óptimo de la función objetivo: {valor_objetivo}")
    for variable, valor in valores_variables.items():
        st.write(f"{variable} = {valor}")

    x1_values = np.linspace(0, 10, 100)
    x2_values_1 = (10 - 4 * x1_values) / 2
    x2_values_2 = (14 - 3 * x1_values) / 4
    x2_values_3 = (7 - 2 * x1_values) / 1

    plt.figure(figsize=(8, 6))
    plt.plot(x1_values, x2_values_1, label="4x1 + 2x2 <= 10")
    plt.plot(x1_values, x2_values_2, label="3x1 + 4x2 <= 14")
    plt.plot(x1_values, x2_values_3, label="2x1 + x2 <= 7")
    plt.fill_between(x1_values, np.minimum.reduce([x2_values_1, x2_values_2, x2_values_3]), color="gray", alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Región Factible de x1 y x2 en el Problema de Optimización")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def ejercicio_8_2():
    st.subheader("Ejercicio 8.2 - Observación de Tiempos de Computación y Región Factible")
    st.write("Resolver el problema en su versión continua y observar el tiempo de computación.")

    # Función para resolver el problema y medir el tiempo de ejecución
    def solve_problem_as(lp_type="continuous", tolerance=None):
        # Definir el problema
        problem = pulp.LpProblem("Ejercicio_8.2", pulp.LpMaximize)
        
        # Definir variables
        x1 = pulp.LpVariable("x1", lowBound=0, cat='Continuous' if lp_type == "continuous" else 'Integer')
        x2 = pulp.LpVariable("x2", lowBound=0, cat='Continuous' if lp_type == "continuous" else 'Integer')
        x3 = pulp.LpVariable("x3", lowBound=0, cat='Continuous' if lp_type == "continuous" else 'Integer')

        # Función objetivo
        problem += 4 * x1 + 3 * x2 + 3 * x3, "Función Objetivo"

        # Restricciones
        problem += 4 * x1 + 2 * x2 + x3 <= 10, "Restricción_1"
        problem += 3 * x1 + 4 * x2 + 2 * x3 <= 14, "Restricción_2"
        problem += 2 * x1 + x2 + 3 * x3 <= 7, "Restricción_3"

        # Resolver el problema con configuración de tolerancia (si aplica)
        solver = pulp.PULP_CBC_CMD(msg=False, mip_gap=tolerance) if tolerance is not None else pulp.PULP_CBC_CMD(msg=False)
        start_time = time.time()
        problem.solve(solver)
        end_time = time.time()

        # Mostrar los resultados
        st.write(f"Tipo de solución: {'Continua' if lp_type == 'continuous' else 'Entera'}")
        st.write(f"Tolerancia: {'No especificada' if tolerance is None else f'{tolerance * 100}%'}")
        if problem.status == 1:
            st.write(f"Estado de la solución: Óptima")
            st.write(f"x1 = {x1.varValue}")
            st.write(f"x2 = {x2.varValue}")
            st.write(f"x3 = {x3.varValue}")
            st.write(f"Valor máximo de la función objetivo: {pulp.value(problem.objective)}")
        else:
            st.write("No se encontró una solución óptima.")
        st.write(f"Tiempo de ejecución: {end_time - start_time:.6f} segundos")

        # Graficar la región factible
        x1_vals = np.linspace(0, 5, 400)
        y1 = (10 - 4 * x1_vals) / 2  # 4x1 + 2x2 <= 10
        y2 = (14 - 3 * x1_vals) / 4  # 3x1 + 4x2 <= 14
        y3 = (7 - 2 * x1_vals)       # 2x1 + x2 <= 7

        # Configuración del gráfico
        plt.figure(figsize=(8, 6))
        plt.plot(x1_vals, y1, label=r'$4x_1 + 2x_2 \leq 10$', color='red')
        plt.plot(x1_vals, y2, label=r'$3x_1 + 4x_2 \leq 14$', color='cyan')
        plt.plot(x1_vals, y3, label=r'$2x_1 + x_2 \leq 7$', color='blue')
        plt.xlim((0, 5))
        plt.ylim((0, 5))

        # Sombrear la región factible
        plt.fill_between(x1_vals, 0, np.minimum(np.minimum(y1, y2), y3), color="green", alpha=0.2)

        # Etiquetas y leyenda
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title("Región Factible - Ejercicio 8.2")
        plt.legend(loc="upper right")
        plt.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(plt)

    # Ejecutar la primera versión (continua)
    solve_problem_as(lp_type="continuous")

def ejercicio_8_3():
    st.subheader("Ejercicio 8.3: Minimización usando Cortes de Gomory")

    st.write("""
    Resolver el siguiente problema de minimización usando cortes de Gomory de manera iterativa:

    Minimizar: C(x, y) = x - y

    Sujeto a:
    - 3x + 4y <= 6
    - x - y <= 1
    - x, y son enteros no negativos
    """)

    prob_gomory = pulp.LpProblem("Minimizar_C", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, cat="Integer")
    y = pulp.LpVariable("y", lowBound=0, cat="Integer")
    
    prob_gomory += x - y, "Función Objetivo"
    prob_gomory += 3 * x + 4 * y <= 6, "Restricción 1"
    prob_gomory += x - y <= 1, "Restricción 2"
    prob_gomory.solve()

    estado = pulp.LpStatus[prob_gomory.status]
    valor_objetivo = pulp.value(prob_gomory.objective)
    
    st.write(f"Estado del problema: {estado}")
    st.write(f"Valor óptimo de la función objetivo: {valor_objetivo}")
    for variable in prob_gomory.variables():
        st.write(f"{variable.name} = {variable.varValue}")

    x_vals = np.linspace(0, 5, 100)
    y1 = (6 - 3 * x_vals) / 4
    y2 = x_vals - 1
    fig, ax = plt.subplots()
    ax.plot(x_vals, y1, label="3x + 4y <= 6", linestyle="--")
    ax.plot(x_vals, y2, label="x - y <= 1", linestyle="--")
    ax.fill_between(x_vals, 0, np.minimum(y1, y2), where=(y1 >= 0) & (y2 >= 0), color="lightblue", alpha=0.3)
    ax.plot([variable.varValue for variable in prob_gomory.variables() if variable.name == "x"],
            [variable.varValue for variable in prob_gomory.variables() if variable.name == "y"],
            'ro', label="Solución Inicial (Entera)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Región Factible y Solución Inicial")
    ax.legend()
    st.pyplot(fig)

def ejercicio_8_4():
    st.subheader("Ejercicio 8.4 - Maximización Continua con `pulp`")
    st.write("Resolver el problema de maximización de forma continua, sin restricciones enteras.")

    # Paso 1: Definir el problema como Programación Lineal Continua
    def solve_continuous_problem():
        # Crear el problema de maximización
        problem = pulp.LpProblem("Ejercicio_8.4", pulp.LpMaximize)

        # Definir variables continuas
        x1 = pulp.LpVariable("x1", lowBound=0)
        x2 = pulp.LpVariable("x2", lowBound=0)
        x3 = pulp.LpVariable("x3", lowBound=0)

        # Función objetivo
        problem += 4 * x1 + 3 * x2 + 3 * x3, "Función Objetivo"

        # Restricciones
        problem += 4 * x1 + 2 * x2 + x3 <= 10
        problem += 3 * x1 + 4 * x2 + 2 * x3 <= 14
        problem += 2 * x1 + x2 + 3 * x3 <= 7

        # Resolver el problema
        problem.solve()

        return x1.varValue, x2.varValue, x3.varValue, pulp.value(problem.objective)

    # Resolver el problema continuo
    x1_val, x2_val, x3_val, max_value = solve_continuous_problem()

    # Mostrar resultados en Streamlit
    st.write(f"Solución Continua:")
    st.write(f"x1 = {x1_val:.2f}")
    st.write(f"x2 = {x2_val:.2f}")
    st.write(f"x3 = {x3_val:.2f}")
    st.write(f"Valor máximo de la función objetivo: {max_value:.2f}")

    # Paso 2: Graficar la región factible en el plano x1-x2
    st.subheader("Región Factible - Ejercicio 8.4")

    # Rango de valores para x1 y cálculo de restricciones en función de x1 y x2
    x1_vals = np.linspace(0, 5, 400)
    y1 = (10 - 4 * x1_vals) / 2  # 4x1 + 2x2 <= 10
    y2 = (14 - 3 * x1_vals) / 4  # 3x1 + 4x2 <= 14
    y3 = (7 - 2 * x1_vals) / 3   # 2x1 + x2 <= 7

    # Configuración del gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(x1_vals, y1, label=r'$4x_1 + 2x_2 + x_3 \leq 10$', color='red')
    plt.plot(x1_vals, y2, label=r'$3x_1 + 4x_2 + 2x_3 \leq 14$', color='blue')
    plt.plot(x1_vals, y3, label=r'$2x_1 + x_2 + 3x_3 \leq 7$', color='green')
    plt.xlim((0, 5))
    plt.ylim((0, 5))

    # Sombrear la región factible
    plt.fill_between(x1_vals, 0, np.minimum(np.minimum(y1, y2), y3), color="gray", alpha=0.3)

    # Etiquetas y leyenda
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title("Región Factible - Ejercicio 8.4")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)

def ejercicio_8_5():
    st.subheader("Ejercicio 8.5 - Selección de Proyectos para Maximizar el NPV")

    problem = pulp.LpProblem("Maximización_del_NPV", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", cat="Binary")
    x2 = pulp.LpVariable("x2", cat="Binary")
    x3 = pulp.LpVariable("x3", cat="Binary")
    x4 = pulp.LpVariable("x4", cat="Binary")
    x5 = pulp.LpVariable("x5", cat="Binary")
    x6 = pulp.LpVariable("x6", cat="Binary")
    
    problem += 141 * x1 + 187 * x2 + 162 * x3 + 83 * x4 + 262 * x5 + 153 * x6, "NPV_Total"
    problem += 75 * x1 + 90 * x2 + 85 * x3 + 30 * x4 + 50 * x5 + 50 * x6 <= 250, "Año_1"
    problem += 25 * x1 + 35 * x2 + 30 * x3 + 20 * x4 + 30 * x5 + 20 * x6 <= 75, "Año_2"
    problem += 25 * x1 + 55 * x2 + 40 * x3 + 20 * x4 + 30 * x5 + 30 * x6 <= 50, "Año_3"
    problem += 20 * x1 + 25 * x2 + 35 * x3 + 15 * x4 + 25 * x5 + 35 * x6 <= 50, "Año_4"
    problem += 15 * x1 + 15 * x2 + 15 * x3 + 10 * x4 + 20 * x5 + 40 * x6 <= 50, "Año_5"
    problem.solve()

    st.write("**Estado de la solución:**", pulp.LpStatus[problem.status])
    st.write("**Valor máximo de NPV =**", pulp.value(problem.objective))
    proyectos = []
    for i, var in enumerate([x1, x2, x3, x4, x5, x6], start=1):
        seleccion = "Seleccionado" if var.varValue == 1 else "No Seleccionado"
        st.write(f"Proyecto {i}: {seleccion}")
        proyectos.append(var.varValue)

    presupuestos_ano = [
        [75, 90, 85, 30, 50, 50],  # Año 1
        [25, 35, 30, 20, 30, 20],  # Año 2
        [25, 55, 40, 20, 30, 30],  # Año 3
        [20, 25, 35, 15, 25, 35],  # Año 4
        [15, 15, 15, 10, 20, 40]   # Año 5
    ]
    presupuestos_disponibles = [250, 75, 50, 50, 50]
    presupuestos_utilizados = [0] * 5
    for i in range(5):
        for j in range(6):
            presupuestos_utilizados[i] += presupuestos_ano[i][j] * proyectos[j]

    st.subheader("Selección de Proyectos")
    fig1, ax1 = plt.subplots()
    ax1.bar(['P1', 'P2', 'P3', 'P4', 'P5', 'P6'], proyectos, color='lightblue')
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel("Selección (1 = Sí, 0 = No)")
    ax1.set_title("Selección de Proyectos")
    st.pyplot(fig1)

    st.subheader("Presupuestos Disponibles vs Utilizados por Año")
    fig2, ax2 = plt.subplots()
    anios = np.arange(1, 6)
    bar_width = 0.35
    ax2.bar(anios, presupuestos_disponibles, width=bar_width, label='Presupuesto Disponible', color='lightblue')
    ax2.bar(anios + bar_width, presupuestos_utilizados, width=bar_width, label='Presupuesto Utilizado', color='salmon')
    ax2.set_xlabel("Año")
    ax2.set_ylabel("Presupuesto (en miles de dólares)")
    ax2.set_title("Presupuesto Disponible vs Utilizado")
    ax2.set_xticks(anios + bar_width / 2)
    ax2.set_xticklabels(['Año 1', 'Año 2', 'Año 3', 'Año 4', 'Año 5'])
    ax2.legend()
    st.pyplot(fig2)

# Lógica para mostrar el ejercicio seleccionado
if seleccion == "Ejercicio 8.1":
    ejercicio_8_1()
elif seleccion == "Ejercicio 8.2":
    ejercicio_8_2()
elif seleccion == "Ejercicio 8.3":
    ejercicio_8_3()
elif seleccion == "Ejercicio 8.4":
    ejercicio_8_4()
elif seleccion == "Ejercicio 8.5":
    ejercicio_8_5()

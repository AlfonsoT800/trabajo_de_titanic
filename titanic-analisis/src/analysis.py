"""
Análisis del dataset Titanic.

Este script carga train.csv, realiza una exploración inicial, limpia los
datos, crea variables nuevas, responde varias preguntas sobre los pasajeros
y guarda las visualizaciones y el dataset limpio en outputs/resultados/.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = os.path.join("data", "train.csv")
OUTPUT_DIR = os.path.join("outputs", "resultados")

sns.set_theme(style="whitegrid")


def linea(titulo):
    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)


def cargar_datos():
    return pd.read_csv(DATA_PATH)


def exploracion_inicial(df):
    linea("EXPLORACIÓN INICIAL")

    print(f"Número de pasajeros: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")

    print("\nVariables disponibles:")
    print(list(df.columns))

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores faltantes por columna:")
    print(df.isnull().sum())

    print(f"\nRegistros duplicados: {df.duplicated().sum()}")

    print("\nEstadísticas descriptivas:")
    print(df.describe(include="all"))


def tratar_valores_faltantes(df):
    linea("TRATAMIENTO DE VALORES FALTANTES")

    # Age: ~20% de valores faltantes, es numérica y tiene outliers,
    # así que se rellena con la mediana en vez del promedio.
    edad_mediana = df["Age"].median()
    df["Age"] = df["Age"].fillna(edad_mediana)
    print(f"Age -> se rellenó con la mediana ({edad_mediana})")

    # Cabin: más del 75% de los datos faltan, así que no tiene sentido
    # rellenarla. En vez de eso se crea un indicador de si el pasajero
    # tenía cabina registrada o no, y se elimina la columna original.
    df["CabinKnown"] = df["Cabin"].notnull().astype(int)
    df = df.drop(columns=["Cabin"])
    print("Cabin -> se reemplazó por la variable binaria CabinKnown y se eliminó la columna original")

    # Embarked: solo 2 valores faltantes, se rellenan con el puerto más común.
    puerto_moda = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(puerto_moda)
    print(f"Embarked -> se rellenó con el valor más frecuente ({puerto_moda})")

    return df


def transformar(df):
    linea("TRANSFORMACIONES")

    # PassengerId, Name y Ticket no aportan directamente al análisis
    # exploratorio, así que se eliminan.
    df = df.drop(columns=["PassengerId", "Name", "Ticket"])
    print("Se eliminaron las columnas PassengerId, Name y Ticket")

    return df


def crear_variables(df):
    linea("NUEVAS VARIABLES")

    # FamilySize: número total de familiares a bordo, incluyendo al pasajero.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print("Se creó FamilySize = SibSp + Parch + 1")

    # AgeGroup: categoría de edad. Criterios definidos para esta práctica:
    #   Niño:         0  - 12 años
    #   Joven:        13 - 25 años
    #   Adulto:       26 - 60 años
    #   Adulto mayor: 61 años en adelante
    bins = [0, 12, 25, 60, 100]
    etiquetas = ["Niño", "Joven", "Adulto", "Adulto mayor"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=etiquetas, include_lowest=True)
    print("Se creó AgeGroup con las categorías: Niño (0-12), Joven (13-25), Adulto (26-60), Adulto mayor (61+)")

    return df


def analisis(df):
    linea("ANÁLISIS")

    tasa_general = df["Survived"].mean() * 100
    print(f"1) Porcentaje de pasajeros que sobrevivió: {tasa_general:.1f}%")

    por_sexo = df.groupby("Sex")["Survived"].mean() * 100
    print("\n2) Supervivencia por sexo (%):")
    print(por_sexo.round(1))

    por_clase = df.groupby("Pclass")["Survived"].mean() * 100
    print("\n3) Supervivencia por clase del pasajero (%):")
    print(por_clase.round(1))

    por_edad = df.groupby("AgeGroup", observed=True)["Survived"].mean() * 100
    print("\n4) Supervivencia por grupo de edad (%):")
    print(por_edad.round(1))

    df["ViajaSolo"] = df["FamilySize"] == 1
    por_compania = df.groupby("ViajaSolo")["Survived"].mean() * 100
    print("\n5) Supervivencia según si viaja solo o acompañado (%):")
    print(por_compania.round(1).rename({True: "Solo", False: "Acompañado"}))

    return {
        "tasa_general": tasa_general,
        "por_sexo": por_sexo,
        "por_clase": por_clase,
        "por_edad": por_edad,
        "por_compania": por_compania,
    }


def visualizaciones(df):
    linea("VISUALIZACIONES")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Sex", y="Survived", errorbar=None)
    plt.title("Tasa de supervivencia por sexo")
    plt.ylabel("Tasa de supervivencia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "supervivencia_por_sexo.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Pclass", y="Survived", errorbar=None)
    plt.title("Tasa de supervivencia por clase")
    plt.xlabel("Clase")
    plt.ylabel("Tasa de supervivencia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "supervivencia_por_clase.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="AgeGroup", y="Survived", errorbar=None, order=["Niño", "Joven", "Adulto", "Adulto mayor"])
    plt.title("Tasa de supervivencia por grupo de edad")
    plt.xlabel("Grupo de edad")
    plt.ylabel("Tasa de supervivencia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "supervivencia_por_edad.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="FamilySize", y="Survived", errorbar=None)
    plt.title("Tasa de supervivencia por tamaño de familia")
    plt.xlabel("Tamaño de familia")
    plt.ylabel("Tasa de supervivencia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "supervivencia_por_familia.png"))
    plt.close()

    print(f"Se guardaron 4 gráficas en {OUTPUT_DIR}/")


def conclusiones(resultados):
    linea("CONCLUSIONES")

    texto = f"""
- Del total de pasajeros, sobrevivió aproximadamente el {resultados['tasa_general']:.1f}%.
- Las mujeres tuvieron una tasa de supervivencia mucho más alta que los hombres
  ({resultados['por_sexo'].get('female', 0):.1f}% vs {resultados['por_sexo'].get('male', 0):.1f}%).
- La clase del pasajero influyó claramente: los pasajeros de primera clase
  sobrevivieron en mayor proporción que los de segunda y tercera.
- Los niños tuvieron una tasa de supervivencia más alta que los adultos,
  lo que coincide con la idea de que se priorizó a los pasajeros más jóvenes.
- Viajar acompañado (en familias pequeñas) se asoció con una mayor
  supervivencia que viajar completamente solo.
"""
    print(texto)

    with open(os.path.join(OUTPUT_DIR, "conclusiones.txt"), "w", encoding="utf-8") as f:
        f.write(texto.strip() + "\n")


def main():
    df = cargar_datos()
    exploracion_inicial(df)
    df = tratar_valores_faltantes(df)
    df = transformar(df)
    df = crear_variables(df)
    resultados = analisis(df)
    visualizaciones(df)
    conclusiones(resultados)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "titanic_clean.csv"), index=False)
    print(f"\nDataset limpio guardado en {OUTPUT_DIR}/titanic_clean.csv")


if __name__ == "__main__":
    main()

# Análisis de pasajeros del Titanic

Proyecto de la práctica de Big Data: *Git, GitHub y reproducibilidad de proyectos de datos*. Analiza la información de los pasajeros del Titanic para identificar qué características se relacionan con la supervivencia.

## Dataset

- **Nombre:** Titanic - Machine Learning from Disaster (archivo `train.csv`)
- **Fuente:** [Kaggle - Titanic](https://www.kaggle.com/c/titanic/data)
- **Descripción:** contiene información de 891 pasajeros del Titanic, incluyendo clase, sexo, edad, tarifa pagada, familiares a bordo y si sobrevivió o no.

## Objetivo

Realizar limpieza, preprocesamiento, análisis exploratorio y visualización de los datos de los pasajeros, para identificar patrones asociados con la supervivencia. **No se entrena ningún modelo de Machine Learning** en esta práctica.

## Requisitos

- Python 3.10 o superior
- Las dependencias listadas en `requirements.txt`

## Instalación

Clonar el repositorio:

```bash
git clone URL_DEL_REPOSITORIO
cd titanic-analisis
```

Crear el entorno virtual:

```bash
python -m venv .venv
```

Activarlo:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python src/analysis.py
```

El script imprime en consola la exploración, el tratamiento de valores faltantes, el análisis y las conclusiones, y guarda en `outputs/resultados/`:

- 4 gráficas (`.png`)
- El dataset limpio (`titanic_clean.csv`)
- Las conclusiones (`conclusiones.txt`)

## Análisis realizados

**Valores faltantes:**
- `Age` (177 nulos) se rellenó con la mediana.
- `Cabin` (687 nulos) se reemplazó por una variable binaria `CabinKnown` (si se conocía o no la cabina) y se eliminó la columna original.
- `Embarked` (2 nulos) se rellenó con el puerto más frecuente.

**Variables nuevas:**
- `FamilySize = SibSp + Parch + 1`
- `AgeGroup`: categoría de edad (Niño 0-12, Joven 13-25, Adulto 26-60, Adulto mayor 61+)

**Preguntas respondidas:**
1. ¿Qué porcentaje de pasajeros sobrevivió?
2. ¿Cómo cambia la supervivencia entre hombres y mujeres?
3. ¿Cómo cambia la supervivencia según la clase del pasajero?
4. ¿Qué grupos de edad presentan mayor supervivencia?
5. ¿Viajar solo o acompañado está relacionado con la supervivencia?

## Resultados y conclusiones

- Sobrevivió aproximadamente el 38.4% de los pasajeros.
- Las mujeres sobrevivieron en mucha mayor proporción que los hombres (74.2% vs 18.9%).
- La clase del pasajero influyó claramente: 1ª clase 63.0%, 2ª clase 47.3%, 3ª clase 24.2%.
- Los niños (0-12 años) tuvieron la tasa de supervivencia más alta entre los grupos de edad (58.0%).
- Viajar acompañado se asoció con una mayor supervivencia (50.6%) que viajar solo (30.4%).

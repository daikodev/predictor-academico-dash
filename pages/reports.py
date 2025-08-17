from dash import dcc, html
from data.data_loader import student
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

X = student.drop("Exam_Result", axis=1)
y = student["Exam_Result"]

model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                      solver='adam', random_state=42, max_iter=500)
model.fit(X, y)

# --- CÁLCULOS DE LOS KPIS ---
total_estudiantes = len(student)
tasa_aprobacion = (student['Exam_Result'].sum() / total_estudiantes) * 100


# Obtener predicciones y contar estudiantes en riesgo
predicciones = model.predict(X)
estudiantes_en_riesgo = (predicciones == 0).sum()

precision_modelo = accuracy_score(y, predicciones) * 100

# Formatear los valores para la visualización
tasa_aprobacion_str = f'{tasa_aprobacion:.1f}%'
precision_modelo_str = f'{precision_modelo:.1f}%'

# Calculamos importancia
r = permutation_importance(model, X, y, n_repeats=30,
                           random_state=42, n_jobs=-1)

# Convertimos a DataFrame
importances_df = pd.DataFrame({
    "feature": X.columns,
    "importance": r.importances_mean
})

# Ordenamos y tomamos las 5 más importantes
top5 = importances_df.sort_values(by="importance", ascending=False).head(5)

top5 = top5.sort_values(by="importance", ascending=True)


fig_importance = px.bar(
    top5,
    x="importance",
    y="feature",
    orientation="h",
    title="Top 5 Variables más influyentes en el resultado del examen",
    labels={"importance": "Importancia", "feature": "Variables"},
    text="importance"
)
fig_importance.update_traces(texttemplate="%{text:.4f}", textposition="outside")

fig = px.pie(
    student,
    names="Exam_Result", 
    title="Distribución de Resultados (Aprobado vs Desaprobado)",
    color="Exam_Result",
    color_discrete_map={1: "green", 0: "red"}  
)

fig_box = px.box(
    student,
    x="Exam_Result",
    y="Hours_Studied",
    color="Exam_Result",
    title="Horas de Estudio según el Resultado",
    labels={"Exam_Result": "Resultado", "Hours_Studied": "Horas de Estudio"},
    color_discrete_map={1: "green", 0: "red"}
)

fig_bar = px.histogram(
    student,
    x="Motivation_Level",
    color="Exam_Result",
    barmode="group",
    histnorm="percent",
    title="Rendimiento por Nivel de Motivación<br>Comparación de tasas de aprobación según motivación del estudiante",
    labels={
        "Motivation_Level": "Nivel de Motivación",
        "Exam_Result": "Resultado del Examen",
        "count": "Porcentaje de Estudiantes"
    },
    color_discrete_map={1: "blue", 0: "gray"}, 
    category_orders={
        "Exam_Result": [1, 0] 
    }
)

cm = confusion_matrix(y, predicciones)
x_labels = ['Desaprobado', 'Aprobado']
y_labels = ['Desaprobado', 'Aprobado']
fig_confusion = ff.create_annotated_heatmap(
    cm, x=x_labels, y=y_labels, colorscale='Blues',
    showscale=True, annotation_text=[[str(cell) for cell in row] for row in cm]
)
fig_confusion.update_layout(title="Matriz de Confusión", xaxis_title="Predicción", yaxis_title="Real")

# Curva ROC
y_score = model.predict_proba(X)[:,1]
fpr, tpr, _ = roc_curve(y, y_score)
roc_auc = auc(fpr, tpr)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatorio', line=dict(dash='dash')))
fig_roc.update_layout(title='Curva ROC', xaxis_title='Tasa de Falsos Positivos', yaxis_title='Tasa de Verdaderos Positivos')

# Mapa de Correlación
fig_corr = px.imshow(
    student.corr(),
    text_auto=True,
    title="Mapa de Correlación entre Variables"
)

# (Opcional) Distribución de Probabilidades de Predicción
probs = model.predict_proba(X)[:,1]
fig_probs = px.histogram(
    probs, nbins=20,
    title="Distribución de Probabilidades de Predicción",
    labels={'value':'Probabilidad de Aprobar', 'count':'Cantidad'}
)
fig_probs.update_layout(showlegend=False)

# Ajusta el eje y para que vaya de 0 a 100
fig_bar.update_yaxes(title="Tasa de Aprobación (%)", range=[0, 100])

# Eliminar las etiquetas de porcentaje sobre las barras si no las quieres
fig_bar.update_traces(texttemplate=None)

kpi_cards = html.Div(
    className='kpi-container',
    children=[
        html.Div(
            className='kpi-card',
            children=[
                html.P('Total Estudiantes', className='kpi-title'),
                html.P(f'{total_estudiantes:,}', className='kpi-value'),
                html.P('Datos totales', className='kpi-subtitle-info')
            ]
        ),
        html.Div(
            className='kpi-card',
            children=[
                html.P('Tasa de Aprobación', className='kpi-title'),
                html.P(tasa_aprobacion_str, className='kpi-value-green'),
                html.P('Porcentaje de aprobados', className='kpi-subtitle-info')
            ]
        ),
        html.Div(
            className='kpi-card',
            children=[
                html.P('Predicciones Exactas', className='kpi-title'),
                html.P(precision_modelo_str, className='kpi-value'),
                html.P('Precisión del modelo', className='kpi-subtitle-info')
            ]
        ),
        html.Div(
            className='kpi-card',
            children=[
                html.P('Estudiantes en Riesgo', className='kpi-title'),
                html.P(f'{estudiantes_en_riesgo}', className='kpi-value-red'),
                html.P('Requieren intervención', className='kpi-subtitle-info')
            ]
        )
    ]
)

layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Reportes y Análisis"),

            html.P("Visualización de datos y tendencias del rendimiento académico",
                  className='subtitle'),
        ], className='container-1'),
        kpi_cards,
        html.Div([
            html.Div([dcc.Graph(figure=fig)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_box)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_bar)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_importance)], className='chart-item'),
        ], className='container-for-report'),
    ], className='container_report')
])

layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Reportes y Análisis"),
            html.P("Visualización de datos y tendencias del rendimiento académico", className='subtitle'),
        ], className='container-1'),
        kpi_cards,
        html.Div([
            html.Div([dcc.Graph(figure=fig)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_box)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_bar)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_importance)], className='chart-item'),
            # --- Agrega aquí los nuevos gráficos ---
            html.Div([dcc.Graph(figure=fig_confusion)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_roc)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_corr)], className='chart-item'),
            html.Div([dcc.Graph(figure=fig_probs)], className='chart-item'),
        ], className='container-for-report'),
    ], className='container_report')
])
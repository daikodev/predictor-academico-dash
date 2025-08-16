import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np


student = pd.read_csv("data/Students.csv", sep=",")

X = student.drop("Exam_Result", axis=1)
y = student["Exam_Result"]


model = MLPClassifier(hidden_layer_sizes=(
    100, 50), activation='relu', solver='adam', random_state=42, max_iter=500)
model.fit(X, y)

app = dash.Dash(__name__)
server = app.server

navbar = html.Nav([
    html.Div([
        html.Div([
            # Contenedor del Logo
            html.Div([
                html.Div([
                    html.Span(
                        className="vaadin--academy-cap logo-icon-container"),
                ], className="icon-wrapper"),
                html.Span("Predictor Académico", className="logo-text")

            ], className="logo"),

            # Contenedor de los Enlaces de Navegación
            html.Div([
                dcc.Link([
                    html.Span(className="lets-icons--form-fill"),
                    html.Span("Formulario")
                ], href="/", className="nav-link"),
                dcc.Link([
                    html.Span(className="mdi--report-line"),
                    html.Span("Reportes")
                ], href="/reports", className="nav-link"),
            ], className="nav-links")
        ], className="nav-menu")
    ], className="navbar")
])

app.layout = html.Div([
    navbar,

    html.Div([
        html.Div([
            html.H1("Predictor de Rendimiento Académico"),

            html.P("Completa todos los campos para predecir si el estudiante aprobará o desaprobará",
                   className='subtitle'),
        ], className='container-1'),

        html.Div([
            html.Div([
                # --- Datos Académicos ---
                html.Div([
                    html.H3("Datos Académicos"),

                    html.Div([
                        html.Div([
                            html.Label("Horas de Estudio (por semana)"),
                            dcc.Input(id='Hours_Studied',
                                      type='number', min=0),
                        ], style={'width': '50%'}),

                        html.Div([
                            html.Label("Asistencia (%)"),
                            dcc.Input(id='Attendance',
                                      type='number', min=0),
                        ], style={'width': '50%'}),
                    ], className='flex-1'),

                    html.Div([
                        html.Div([
                            html.Label("Calificaciones Anteriores"),
                            dcc.Input(id='Previous_Scores',
                                      type='number', min=0),
                        ], style={'width': '50%'}),

                        html.Div([
                            html.Label("Sesiones de Tutoría"),
                            dcc.Dropdown(id='Tutoring_Sessions',
                                         options=[{'label': 'Sí', 'value': 1},
                                                  {'label': 'No', 'value': 0}],
                                         placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),
                    ], className='flex-1'),
                ], className='card'),

                # --- Factores Familiares ---
                html.Div([
                    html.H3("Factores Familiares"),

                    html.Div([
                        html.Div([
                            html.Label("Participación Parental"),
                            dcc.Dropdown(id='Parental_Involvement',
                                         options=[{'label': 'Alta', 'value': 2}, {'label': 'Media', 'value': 1}, {'label': 'Baja', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),

                        html.Div([
                            html.Label("Ingresos Familiares"),
                            dcc.Dropdown(id='Family_Income',
                                         options=[{'label': 'Altos', 'value': 2}, {'label': 'Medios', 'value': 1}, {'label': 'Bajos', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),
                    ], className='flex-1'),

                    html.Div([
                        html.Div([
                            html.Label("Nivel Educativo Parental"),
                            dcc.Dropdown(id='Parental_Education_Level',
                             options=[{'label': 'Universitario', 'value': 2}, {'label': 'Secundaria', 'value': 1}, {'label': 'Primaria', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),

                        html.Div([
                            html.Label("Distancia del Hogar"),
                            dcc.Dropdown(id='Distance_from_Home',
                                         options=[{'label': 'Cerca', 'value': 0}, {'label': 'Media', 'value': 1}, {'label': 'Lejos', 'value': 2}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),
                    ], className='flex-1'),
                ],  className='card'),

                # --- Recursos y Ambiente ---
                html.Div([
                    html.H3("Recursos y Ambiente"),

                    html.Div([
                        html.Div([
                            html.Label("Acceso a Recursos"),
                            dcc.Dropdown(id='Access_to_Resources',
                             options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),

                        html.Div([
                            html.Label("Acceso a Internet"),
                            dcc.Dropdown(id='Internet_Access',
                                         options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),
                    ], className='flex-1'),

                    html.Div([
                        html.Div([
                            html.Label("Calidad del Profesor"),
                            dcc.Dropdown(id='Teacher_Quality',
                             options=[{'label': 'Alta', 'value': 2}, {'label': 'Media', 'value': 1}, {'label': 'Baja', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),

                        html.Div([
                            html.Label("Actividades Extracurriculares"),
                            dcc.Dropdown(id='Extracurricular_Activities',
                                         options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),
                    ], className='flex-1'),

                ],  className='card'),
            ], className='flex-column'),


            html.Div([
                # --- Factores Personales ---
                html.Div([
                    html.H3("Factores Personales"),

                    html.Div([
                        html.Div([
                            html.Label("Nivel de Motivación"),
                            dcc.Dropdown(id='Motivation_Level',
                             options=[{'label': 'Alta', 'value': 2}, {'label': 'Media', 'value': 1}, {'label': 'Baja', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),

                        html.Div([
                            html.Label("Influencia de Compañeros"),
                            dcc.Dropdown(id='Peer_Influence',
                                         options=[{'label': 'Positiva', 'value': 1}, {'label': 'Negativa', 'value': 0}], placeholder="Selecciona una opción"),
                        ], style={'width': '50%'}),
                    ], className='flex-1'),
                ],  className='card'),

                html.Div([
                    html.Button([
                        html.I(className='iconamoon--trend-up',
                               style={'marginRight': '12px'}),
                        "Predecir Resultado"
                    ], id='predict-button', className='button-predict')
                ],  className='card'),

                html.Div([
                    html.Div(id='output_prediction',
                             className='text-container')
                ],  id='prediction_container', className='card', style={'display': 'none'})

            ], className='flex-column')

        ], className='container-2'),
    ], className='container')
])



@app.callback(
    Output('output_prediction', 'children'),
    Output('prediction_container', 'style'),
    Output('prediction_container', 'className'),
    Input('predict-button', 'n_clicks'),
    State('Hours_Studied', 'value'),
    State('Attendance', 'value'),
    State('Parental_Involvement', 'value'),
    State('Access_to_Resources', 'value'),
    State('Extracurricular_Activities', 'value'),
    State('Previous_Scores', 'value'),
    State('Motivation_Level', 'value'),
    State('Internet_Access', 'value'),
    State('Tutoring_Sessions', 'value'),
    State('Family_Income', 'value'),
    State('Teacher_Quality', 'value'),
    State('Peer_Influence', 'value'),
    State('Parental_Education_Level', 'value'),
    State('Distance_from_Home', 'value')
)
def predict_student(n_clicks, Hours_Studied, Attendance, Parental_Involvement,
                    Access_to_Resources, Extracurricular_Activities, Previous_Scores,
                    Motivation_Level, Internet_Access, Tutoring_Sessions, Family_Income,
                    Teacher_Quality, Peer_Influence, Parental_Education_Level, Distance_from_Home):

    if not n_clicks:
        return "",  {'display': 'none'}, 'card'

    inputs = [Hours_Studied, Attendance, Parental_Involvement, Access_to_Resources,
              Extracurricular_Activities, Previous_Scores, Motivation_Level,
              Internet_Access, Tutoring_Sessions, Family_Income, Teacher_Quality,
              Peer_Influence, Parental_Education_Level, Distance_from_Home]

    if None in inputs:
        return [
            html.I(className='mi--warning', style={'marginRight': '12px'}),

            html.Div([
                html.Span("Error"),
                html.Span("Por favor, completa todos los campos.",
                          style={'color': '#c2410c'}),
            ], className='text')
        ], {'display': 'block'}, "card card-warning"

    if Hours_Studied > 85:
        return [
            html.I(className='mi--warning', style={'marginRight': '12px'}),
            html.Div([
                html.Span("Advertencia"),
                html.Span("Las horas de estudio no pueden superar 85 por semana.",
                          style={'color': '#c2410c'}),
            ], className='text')
        ], {'display': 'block'}, "card card-warning"

    if Attendance > 100:
        return [
            html.I(className='mi--warning', style={'marginRight': '12px'}),
            html.Div([
                html.Span("Advertencia"),
                html.Span("La asistencia no puede superar el 100%.",
                          style={'color': '#c2410c'}),
            ], className='text')
        ], {'display': 'block'}, "card card-warning"
    if Previous_Scores > 100:
        return [
            html.I(className='mi--warning', style={'marginRight': '12px'}),
            html.Div([
                html.Span("Advertencia"),
                html.Span("Las calificaciones anteriores no pueden superar 100.",
                          style={'color': '#c2410c'}),
            ], className='text')
        ], {'display': 'block'}, "card card-warning"

    # Calcular automáticamente la variable derivada
    try:
        study_per_attendance = Hours_Studied / Attendance
    except ZeroDivisionError:
        return [
            html.I(className='mi--warning', style={'marginRight': '12px'}),

            html.Div([
                html.Span("Error"),
                html.Span("La asistencia no puede ser 0 para calcular la métrica.",
                          style={'color': '#c2410c'}),
            ], className='text')
        ], {'display': 'block'}, "card card-warning"

    # Crear DataFrame para predecir (en el mismo orden que el entrenamiento)
    new_register = pd.DataFrame([{
        'Hours_Studied': Hours_Studied,
        'Attendance': Attendance,
        'Parental_Involvement': Parental_Involvement,
        'Access_to_Resources': Access_to_Resources,
        'Extracurricular_Activities': Extracurricular_Activities,
        'Previous_Scores': Previous_Scores,
        'Motivation_Level': Motivation_Level,
        'Internet_Access': Internet_Access,
        'Tutoring_Sessions': Tutoring_Sessions,
        'Family_Income': Family_Income,
        'Teacher_Quality': Teacher_Quality,
        'Peer_Influence': Peer_Influence,
        'Parental_Education_Level': Parental_Education_Level,
        'Distance_from_Home': Distance_from_Home,
        'study_per_attendance': study_per_attendance
    }])

    pred = model.predict(new_register)[0]
    if pred == 1:
        return [
            html.I(className='simple-line-icons--check',
                   style={'marginRight': '12px'}),
            html.Div([
                html.Span("Aprobado"),
                html.Span("El estudiante tiene alta probabilidad de aprobar",
                          style={'color': '#008235'})
            ], className='text'),
        ], {'display': 'block'}, 'card card-success'
    else:
        return [
            html.I(className='f7--xmark-circle',
                   style={'marginRight': '12px'}),
            html.Div([
                html.Span("Desaprobado"),
                html.Span("Se recomienda apoyo adicional",
                          style={'color': '#C10006'})
            ], className='text'),
        ], {'display': 'block'}, 'card card-danger'


if __name__ == '__main__':
    app.run(debug=True)
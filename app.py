import dash
from dash import dcc, html, Input, Output, State
from data.data_loader import student
from pages.form import layout as form_layout
from pages.reports import layout as reports_layout
import plotly.express as px
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import os

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Carga de Datos ---
X = student.drop("Exam_Result", axis=1)
y = student["Exam_Result"]
model = MLPClassifier(hidden_layer_sizes=(
    100, 50), activation='relu', solver='adam', random_state=42, max_iter=500)
model.fit(X, y)

# --- Menú de Navegación ---
navbar = html.Nav([
    html.Div([
        html.Div([
            # Logo
            html.Div([
                html.Div([
                    html.Span(className="vaadin--academy-cap logo-icon-container"),
                ], className="icon-wrapper"),
                html.Span("Predictor Académico", className="logo-text")
            ], className="logo"),

            html.Button([
                html.Span(className="hamburger-bar"),
                html.Span(className="hamburger-bar"),
                html.Span(className="hamburger-bar"),
            ], id="hamburger-btn", className="hamburger", n_clicks=0),

            html.Div([
                dcc.Link([
                    html.Span(className="lets-icons--form-fill"),
                    html.Span("Formulario")
                ], href="/", id="link-form", className="nav-link"),
                dcc.Link([
                    html.Span(className="mdi--report-line"),
                    html.Span("Reportes")
                ], href="/reports", id="link-reports", className="nav-link"),
            ], className="nav-links", id="nav-links"),
        ], className="nav-menu")
    ], className="navbar"),
])

# --- Layout Principal ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# --- Callback para Navegar entre Páginas ---
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/reports":
        return reports_layout
    return form_layout

# --- Callback para Clase del Link Activo ---
@app.callback(
    Output("link-form", "className"),
    Output("link-reports", "className"),
    Input("url", "pathname")
)
def update_active_link(pathname):
    active_class = "nav-link active"
    normal_class = "nav-link"

    if pathname == "/reports":
        return normal_class, active_class
    return active_class, normal_class

@app.callback(
    Output("nav-links", "className"),
    Input("hamburger-btn", "n_clicks"),
    State("nav-links", "className"),
)
def toggle_menu(n_clicks, current_class):
    # Solo alterna en movil, en escritorio siempre es visible por CSS
    if n_clicks and n_clicks % 2 == 1:
        return "nav-links nav-links-open"
    return "nav-links"

# --- Callback para Predecir ---
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)

import dash
from dash import dcc, html, Input, Output, State

layout = html.Div([
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
from dash import html

layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Reportes y Análisis"),

            html.P("Visualización de datos y tendencias del rendimiento académico",
                  className='subtitle'),
        ], className='container-1'),

        html.Div([
        ], className='container-2'),
    ], className='container_report')
])

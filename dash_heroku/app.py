import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score,accuracy_score
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import dash_daq as daq
from sklearn import metrics

df = pd.read_csv('https://raw.githubusercontent.com/DavidMairena/Heart_Attack_Possibility/main/heart_disease_uci.csv')


df.drop('dataset', axis=1, inplace=True) #Remove Dataset column
df_not_null = df.dropna()  #Remove NA
df_not_null.num.replace([2,3,4], 1, inplace = True) #Group all levels of Heart Disease into one

#Model and Pipeline
features = df_not_null.drop(['num', 'id', 'oldpeak'], axis=1)
target = df_not_null.num

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model_numeric_features = X_train.select_dtypes(exclude=['object', 'category']).columns
model_categorical_features = X_train.select_dtypes(['object', 'category']).columns

#Logistic Regression

numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[('one_hot', OneHotEncoder(handle_unknown='ignore'))]
)

preprocessor = ColumnTransformer(
    transformers=[
                  ('numeric', numeric_transformer, model_numeric_features),
                  ('categorical', categorical_transformer, model_categorical_features)
    ]
)

clf = Pipeline(
    steps=[
           ('preprocessor', preprocessor),
           ('Logistic_reg', LogisticRegression(solver='liblinear'))
    ]
)

clf.fit(X_train, y_train)

logistic_preds = clf.predict(X_test)



##### Dash App #####


#Figure 1
y_score = clf.predict_proba(X_test)[:, 1]

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
score = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

fig1 = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={score:.4f})',
    labels=dict(
        x='False Positive Rate', 
        y='True Positive Rate'))
fig1.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1)


# Figure 2
y_score2 = clf.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_score2)

# The histogram of scores compared to true labels
fig_hist = px.histogram(df_not_null, x='age', color="num", barmode='group', title="Distribution of Heart Diseases by Age",
    labels={'num': 'Heart Diseases'}
)


# Figure 3

cm = confusion_matrix(y_test, logistic_preds)
cm = cm.astype(int)

z_text = [[str(y) for y in x] for x in cm]
fig3 = ff.create_annotated_heatmap(cm, x=['True', 'False'], y=['True', 'False'], annotation_text=z_text, colorscale='Viridis')
fig3.update_layout(title_text='<i><b>Confusion matrix</b></i>')
    
fig3['data'][0]['showscale'] = True

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1('Heart Diseases Dash'),
        html.Div([
            dcc.Link('See Code', href='#')
        ], className='nav_div')
    ], className='nav'),
    html.Div([
        html.Div([
            html.H2('Predict Heart Diseasse'),
            html.Div([
                html.P('Enter your age:'),
                daq.NumericInput(
                id='age_input',
                min=0,
                max=100,
                value=0
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Select your sex:'),
                dcc.RadioItems(
                    id='sex_input',
                    options=[
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'}
                    ],
                    value='Male'
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Chest pain type:'),
                dcc.Dropdown(
                    id='chest_input',
                    options=[
                        {'label': 'Asymptomatic', 'value': 'asymptomatic'},
                        {'label': 'Non-anginal', 'value': 'non-anginal'},
                        {'label': 'Typical angina', 'value': 'typical angina'},
                        {'label': 'Atypical angina', 'value': 'atypical angina'}
                    ],
                    value='asymptomatic'
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Fasting blood sugar > 120 mg/dl:'),
                dcc.RadioItems(
                    id='blood_sugar_input',
                    options=[
                        {'label': 'Yes', 'value': 'True'},
                        {'label': 'No', 'value': 'False'}
                    ],
                    value='False'
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Max. heart rate:'),
                daq.NumericInput(
                id='heart_rate_input',
                min=0,
                max=1000,
                value=0
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Exercise-induced angina:'),
                dcc.RadioItems(
                    id='exercise_angina_input',
                    options=[
                        {'label': 'Yes', 'value': 'True'},
                        {'label': 'No', 'value': 'False'}
                    ],
                    value='False'
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Slope of the peak exercise:'),
                dcc.Dropdown(
                    id='peak_input',
                    options=[
                        {'label': 'Flat', 'value': 'flat'},
                        {'label': 'Upsloping', 'value': 'upsloping'},
                        {'label': 'Downsloping', 'value': 'downsloping'}
                    ],
                    value='flat'
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Major vessels (0-3) colored by fluoroscopy:'),
                dcc.RadioItems(
                    id='vessels_input',
                    options=[
                        {'label': '0', 'value': 0},
                        {'label': '1', 'value': 1},
                        {'label': '2', 'value': 2},
                        {'label': '3', 'value': 3}
                    ],
                    value=0
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Thal information:'),
                dcc.Dropdown(
                    id='thal_input',
                    options=[
                        {'label': 'Normal', 'value': 'normal'},
                        {'label': 'Fixed defect', 'value': 'fixed defect'},
                        {'label': 'Reversable defect', 'value': 'reversable defect'}
                    ],
                    value='normal'
                )
            ], className='inline_pred'),
            html.Div([
                html.P('Resting blood pressure(mm Hg):', id='blood_press_p'),
                dcc.Slider(
                    id='blood_press_input',
                    min=94,
                    max=200,
                    marks={
                        94: {'label': '94 ', 'style': {'color': 'black'}},125: {'label': '125', 'style': {'color': 'black'}},
                        150: {'label': '150 ', 'style': {'color': 'black'}},175: {'label': '175 ', 'style': {'color': 'black'}},
                        200: {'label': '200 ', 'style': {'color': 'black'}}
                        },
                    value=100,
                )
            ]),
            html.Div([
                html.P('Cholesterol in mg/dl:', id='chol_p'),
                dcc.Slider(
                    id='chol_input',
                    min=100,
                    max=500,
                    marks={
                        100: {'label': '100', 'style': {'color': 'black'}},200: {'label': '200', 'style': {'color': 'black'}},
                        300: {'label': '300', 'style': {'color': 'black'}},400: {'label': '400', 'style': {'color': 'black'}},
                        500: {'label': '500', 'style': {'color': 'black'}}
                        },
                    value=200,
                )
            ]),
            html.Div([
                html.P('Resting electrocardiographic results:'),
                dcc.Dropdown(
                    id='electro_input',
                    options=[
                        {'label': 'Normal', 'value': 'normal'},
                        {'label': 'Abnormality', 'value': 'st-t abnormality'},
                        {'label': 'Hypertrophy', 'value': 'lv hypertrophy'}
                    ],
                    value='normal'
                )
            ]),
            html.Div([
                html.Button(
                    'Run',
                    id='run_button',
                    n_clicks=0
                )
            ], className='button_container')
        ],className='pred_container'),
        html.Div([
            html.Div([
                daq.LEDDisplay(
                    id='records_led',
                    label={'label':'Records', 'style': {'font-size': 19}},
                    value=len(df_not_null),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                )
            ], className='flex_led'),
            html.Div([
                daq.LEDDisplay(
                       id='train_led',
                    label={'label':'Train', 'style': {'font-size': 19}},
                    value=len(X_train),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                ),
                daq.LEDDisplay(
                    id='test_led',
                    label={'label':'Test', 'style': {'font-size': 19}},
                    value=len(X_test),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                )
            ], className='flex_led'),
            html.Div([
                daq.LEDDisplay(
                    id='numeric_led',
                    label={'label':'Numeric', 'style': {'font-size': 19}},
                    value=len(model_numeric_features),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                ),
                daq.LEDDisplay(
                    id='features_led',
                    label={'label':'Features', 'style': {'font-size': 19}},
                    value=X_train.shape[1],
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                ),
                daq.LEDDisplay(
                    id='category_led',
                    label={'label':'Category', 'style': {'font-size': 19}},
                    value=len(model_categorical_features),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                )
            ], className='flex_led'),
            html.Div([
                daq.LEDDisplay(
                    id='precision_led',
                    label={'label':'Precision', 'style': {'font-size': 19}},
                    value=round(precision_score(y_test, logistic_preds), 2),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                ),
                daq.LEDDisplay(
                    id='recall_led',
                    label={'label':'Recall', 'style': {'font-size': 19}},
                    value=round(recall_score(y_test, logistic_preds), 2),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                ),
            ], className='flex_led'),
            html.Div([
                daq.LEDDisplay(
                    id='accuracy_led',
                    label={'label':'Accuracy', 'style': {'font-size': 19}},
                    value=round(accuracy_score(y_test, logistic_preds)*100, 1),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                ),
                daq.LEDDisplay(
                    id='auc_led',
                    label={'label':'AUC', 'style': {'font-size': 19}},
                    value=round(roc_auc_score(y_test, clf.predict_proba(X_test)[::,1]), 2),
                    size=25,
                    color = 'white',
                    backgroundColor= '#1597BB'
                ),
            ], className='flex_led'),
            html.Div([
                html.H3('Your Result: '),
                html.Div(id='result_led', className='default_result_led'),
                html.P('Please run prediction', id='result_p')
            ], className='result_led_container')
        ], className='metrics_container'),
        html.Div([
            dcc.Graph(id='roc_plot', figure=fig1, className='inline_plot margin_plot'),
            dcc.Graph(id='matrix_plot', figure=fig3, className='inline_plot margin_plot'),
            dcc.Graph(id='hist_plot', figure=fig_hist, className='margin_plot')
        ], className='plots_container')
    ], id='page-content'),
    html.Footer([
        html.P('© 2021 Fernando Sirias / David Mairena'),
        html.Div([
            html.P('Data Source: '),
            html.A('UCI Heart Disease Data', href='https://www.kaggle.com/redwankarimsony/heart-disease-data', target='_blank')
        ], className='source')
    ])
], className='content_container')

@app.callback(
    Output('result_led', 'className'),
    Output('result_p', 'children'),
    State('age_input', 'value'),
    State('sex_input', 'value'),
    State('chest_input', 'value'),
    State('blood_press_input', 'value'),
    State('chol_input', 'value'),
    State('blood_sugar_input', 'value'),
    State('electro_input', 'value'),
    State('heart_rate_input', 'value'),
    State('exercise_angina_input', 'value'),
    State('peak_input', 'value'),
    State('vessels_input', 'value'),
    State('thal_input', 'value'),
    Input('run_button', 'n_clicks')
)
def pred_handler(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, clicks):
    if clicks != 0:
        info = pd.DataFrame(np.array([input1, input2, input3, input4, input5, True if input6 == 'True' else False, input7, input8,	True if input9 == 'True' else False,	input10, input11, input12])).T
        info.columns = ['age',	'sex','cp',	'trestbps',	'chol',	'fbs',	'restecg',	'thalch',	'exang',	'slope',	'ca', 'thal']
        print('****************')
        print(info)
        print("///////")
        print(model_categorical_features)
        print("+______+")
        print(model_numeric_features)
        print('****************')
        pred = clf.predict(info)
        print(pred)
        if pred == 0:
            return 'green_result_led', "Low risk of heart diseases"
        elif pred == 1:
            return 'red_result_led', "High risk of heart diseases"
    else:
        return 'default_result_led', 'Please run prediction'

@app.callback(
    Output('chol_p', 'children'),
    Input('chol_input', 'value')
)
def chol_handler(chol_value):
    return "Cholesterol in mg/dl:  {}".format(chol_value)

@app.callback(
    Output('blood_press_p', 'children'),
    Input('blood_press_input', 'value')
)
def blood_press_handler(blood_press_value):
    return "Resting blood pressure(mm Hg):  {}".format(blood_press_value)


if __name__ == '__main__':
    app.run_server(debug=True)

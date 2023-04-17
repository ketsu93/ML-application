import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston

st.set_page_config(page_title='Gamification in Education',
                   layout='wide')

def plot_feature_importances(rf, X):
    plt.figure(figsize=(10, 6))
    n_features = X.shape[1]
    plt.barh(range(n_features), rf.feature_importances_, align='center')
    plt.yticks(range(n_features), X.columns)
    plt.xlabel("Consumption in future from each sector in kW⋅h")
    plt.ylabel("Production from each sector")
    plt.ylim(-1, n_features)

    st.pyplot()

def build_model(df):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100)

    st.markdown('**1.2. Data splits**')
    st.write('Data Set')
    st.info(X_train.shape)
    st.write('Data set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                               random_state=parameter_random_state,
                               max_features=parameter_max_features,
                               criterion=parameter_criterion,
                               n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Consumption**')
    Y_pred_train = rf.predict(X_train)
    st.write('Consumption per hour ($kW⋅h$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('Total Consumption ($kW⋅h$):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Production**')
    Y_pred_test = rf.predict(X_test)
    st.write('Production per hour ($kW⋅h$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('Total Production ($kW⋅h$):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

    plot_feature_importances(rf, X)

# ---------------------------------#
st.write("""
# Gamification in Education 

In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm and study and predict future grades by using gamification.


""")

# ---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('Gamification Data'):
    uploaded_file = st.sidebar.file_uploader("Gamified Student Grades")
    st.sidebar.markdown("")

# Sidebar - Specify parameter settings
with st.sidebar.header('Gamifcation in Education'):
    split_size = st.sidebar.slider('Data split ratio (80)', 10, 90, 80, 5)

with st.sidebar.subheader("2.1. Classification of Students"):
    parameter_n_estimators = st.sidebar.slider('Number of Samples/Studenets', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('1 = Gamified, 2 = Non_gamified', options=[1, 2])


with st.sidebar.subheader('2.2. Additional Features'):
    parameter_random_state = st.sidebar.slider('Number of consumption units (42)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Consumption (criterion)', options=['mse', 'mae'])
    parameter_n_jobs = st.sidebar.select_slider('Increasing-1 Decreasing--1', options=[1, -1])


# ---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for the most recent CSV file to be uploaded.')
    if st.button(''):
        # Diabetes dataset
        # diabetes = load_diabetes()
        # X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        # Y = pd.Series(diabetes.target, name='response')
        # df = pd.concat( [X,Y], axis=1 )

        # st.markdown('The Diabetes dataset is used as the example.')
        # st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat([X, Y], axis=1)

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)



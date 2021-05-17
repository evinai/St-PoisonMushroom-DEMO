import pandas as pd
import numpy as np
import streamlit as st
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn .metrics import precision_score, recall_score

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.sidebar.image('togaylogogri.jpg', width=250)
    st.sidebar.caption('Demo-Uygulama')
    st.title(" Makine Öğretisi ile Zehirli Mantar Tespiti")
    st.text("DEMO-Uygulamadir sonuclar gercegi yansitmayabilir")
    st.markdown('---')
    st.sidebar.title("Klasifikasyon web app")
    st.sidebar.markdown("> ### Model Secimi")
    st.markdown('🍄Mantarlariniz Yenilebilirmi yoksa zehirlimi?')
    st.markdown('---')
# all features are string and categorical. We need to convert them to numeric for scikit learn estimater

    @st.cache(persist=True)
    def load_data():
        data = pd.read_excel('data/mushrooms.xls')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data


    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("> Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    ## SVM
    #lower value of C is a higher regularization strength
    # gamma is kernel coefficient
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader(" > Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient",("scale", "auto"), key='gamma' )

        # Which Evaluation Metrics plotted out
        ## MAKE SURE THE THE SELECTIONS MATCH THE   def plot metrcis if options
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        # with each hyper parameter change we dont want to instantaneously update the web page.
        # thats why its in the if clause
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.markdown("---")
            plot_metrics(metrics)
            st.markdown("---")
        ## LOGISTIC REGRESSION
    if classifier == 'Logistic Regression':
        st.sidebar.subheader(" > Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        # Which Evaluation Metrics plotted out
        ## MAKE SURE THE THE SELECTIONS MATCH THE   def plot metrcis if options
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        # with each hyper parameter change we dont want to instantaneously update the web page.
        # thats why its in the if clause
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)

            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.markdown("---")
            plot_metrics(metrics)
            st.markdown("---")

    if classifier == 'Random Forest':
        st.sidebar.subheader(" > Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap')
        # Which Evaluation Metrics plotted out
        ## MAKE SURE THE THE SELECTIONS MATCH THE   def plot metrcis if options
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # with each hyper parameter change we dont want to instantaneously update the web page.
        # thats why its in the if clause
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            #Remains the same in all 3 ML
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.markdown("---")
            plot_metrics(metrics)
            st.markdown("---")


    df = load_data()
    if st.sidebar.checkbox("(Raw Data) Veri seti", False):
        st.subheader("Mantar veri seti (Classification)")
        st.write(df)
        st.markdown("--- ")











if __name__== '__main__':
    main()

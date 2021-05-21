import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import random
import seaborn as sns
import plotly.graph_objects as go
st.title('Final project Sarvarbek Alimkulov,Bakytzhan Tilektes,Symbat Kozhakhan')
st.sidebar.title('Some statiscs')
st.sidebar.markdown('## Show Z-score')
if st.sidebar.checkbox('Cumulative from mean',key='0'):
    st.markdown('## Cumulative from mean')
    @st.cache(persist=True)
    def load():
        data=pd.read_excel('cfm.xlsx',index_col=0)
        return data
    data=load()
    st.write(data)
    with st.beta_expander("See definition and formula"):
        st.write('The values correspond to the shaded area for given Z This table gives a probability that a statistic is between 0 (the mean) and Z.')
        st.latex(r'''f(z) = \phi(z)-\frac{1}{2}''')
if st.sidebar.checkbox('Cumulative',False, key='1'):
    st.markdown('## Cumulative(less than 0)')
    @st.cache(persist=True)
    def load():
        data=pd.read_excel('z.xlsx',index_col=0)
        return data
    data=load()
    st.write(data)
    st.markdown('## Cumulative(greater than 0)')
    @st.cache(persist=True)
    def load():
        data=pd.read_excel('cp.xlsx',index_col=0)
        return data
    data=load()
    st.write(data)
    with st.beta_expander("See definition and formula"):
        st.write(' This table gives a probability that a statistic is less than Z (i.e. between negative infinity and Z). The values are calculated using the cumulative distribution function of a standard normal distribution with mean of zero and standard deviation of one, usually denoted with the capital Greek letter  𝜙 ')
        st.latex(r'''\phi(z) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z}\exp(\frac{-t^2}{2})dt''')
if st.sidebar.checkbox('Complementary cumulative',False, key='1'):
    st.markdown('## Complementary cumulative ')
    @st.cache(persist=True)
    def load():
        data=pd.read_excel('x.xlsx',index_col=0)
        return data
    data=load()
    st.write(data)
    with st.beta_expander("See definition and formula"):
        st.write('This table gives a probability that a statistic is greater than Z.')
        st.latex(r'''f(z) = 1 - \phi(z)''')
st.sidebar.markdown('## Distributions')
if st.sidebar.checkbox('Show visualizations of distributions?',False,key='3'):
    zscore=st.sidebar.radio('Distributions',('Normal','Poisson','Binomial','Gamma','Exponential'))
    if zscore=='Normal':
        x = np.random.normal(loc=100, scale=20, size=1000).round()
        st.markdown('## Normal distribution')
        st.markdown('The normal distribution is a probability function that describes how the values of a variable are distributed. It is a symmetric distribution where most of the observations cluster around the central peak and the probabilities for values further away from the **mean** taper off equally in both directions.Extreme values in both tails of the distribution are similarly unlikely.')
        st.latex(r'''f(x, \mu, \sigma) = \frac{1}{\sqrt{2\pi * \sigma^2}} e^{-\frac{1}{2} * (\frac{x - \mu}{\sigma})^2}''')
        st.markdown('**Normal random variables with mean=100,standard devation=20,size=100000**')
        fig, ax = plt.subplots()
        ax.hist(x, bins=20)
        st.pyplot(fig)
    elif zscore=='Poisson':
        x = np.random.poisson(lam=100, size=1000).round()
        st.markdown('## Poisson distribution')
        st.markdown(' Poisson distribution is a probability distribution that can be used to show how many times an event is likely to occur within a specified period of time﻿.The Poisson distribution is a discrete function, meaning that the variable can only take specific values in a (potentially infinite) list. ')
        st.latex(r'''P(k) = e^{-\lambda}\frac{\lambda^k}{k!}''')
        st.markdown('**Poisson random variables with lam=100,size=100000**')
        fig, ax = plt.subplots()
        ax.hist(x, bins=20)
        st.pyplot(fig)
    elif zscore=='Binomial':
        x = np.random.binomial(n=10000,p=0.01,size=1000).round()
        st.markdown('## Binomial distribution')
        st.markdown('Binomial distribution is the probability of a success or failure results in an experiment that is repeated a few or many times.')
        st.latex(r''' P(A) = \sum P(\{ (e_1,...,e_N) \})  =  {{N}\choose{k}} \cdot p^kq^{N-k}''')
        st.markdown('**Binomial random variables with n=10000,p=0.01,size=100000**')
        fig, ax = plt.subplots()
        ax.hist(x, bins=20)
        st.pyplot(fig)
    elif zscore=='Gamma':
        x = np.random.gamma(shape=3,scale=20,size=1000).round()
        st.markdown('## Gamma distribution')
        st.markdown('The gamma distribution is a two-parameter family of continuous probability distributions.The gamma distribution is the maximum entropy probability distribution ')
        st.latex(r'''f(x, \alpha, \beta) = \frac{\beta^{\alpha} x^{\alpha-1} e^{-\beta x}}{\Gamma(\alpha)}''')
        st.markdown('**Gamma random variables with shape=3,scale=20,size=100000**')
        fig, ax = plt.subplots()
        ax.hist(x, bins=20)
        st.pyplot(fig)
    elif zscore=='Exponential':
        x = np.random.exponential(scale=20,size=100).round()
        st.markdown('## Exponential distribution')
        st.markdown('The exponential distribution is a continuous probability distribution.The exponential distribution is often concerned with the amount of time until some specific event occurs. ')
        st.latex(r'''f(x, \lambda) = \begin{cases} \lambda e^{-\lambda x} \quad x \geq 0, \\ 0  \quad\quad\quad  x < 0 \end{cases}''')
        st.markdown('**Exponential random variables with scale=20,size=100000**')
        fig, ax = plt.subplots()
        ax.hist(x, bins=20)
        st.pyplot(fig)
st.sidebar.markdown('## Input your numbers')
if st.sidebar.checkbox('Insert your numbers',False,key='4'):
    input=st.sidebar.radio('Situations',('X1 given and total area before X1','X2 given and total area after X2','X1 and X2 given and  total area between X1 and X2'))
    if input=='X1 given and total area before X1':
        loc = st.sidebar.number_input('Insert a mean')
        scale = st.sidebar.number_input('Insert a standard deviation')
        x1 = st.sidebar.number_input('Insert a X1')
        x = np.arange(-6, 10, 0.001)
        fig, ax = plt.subplots()
        less_than=norm.cdf(x=x1, loc=loc, scale=scale)
        ax.plot(x, norm.pdf(x, loc, scale))
        ax.set_title("Normal Distribution")
        ax.set_xlabel('X-Values')
        ax.set_ylabel('PDF(X)')
        ax.grid(True)
        sx=np.arange(-6,x1, 0.01)
        ax.set_ylim(0, 0.25)
        ax.fill_between(sx, norm.pdf(sx, loc, scale), alpha = 0.25, color = 'red')
        ax.text(x1-2, 0.02, round(less_than, 2), fontsize = 15)
        st.pyplot(fig)
    elif input=="X2 given and total area after X2":
        loc = st.sidebar.number_input('Insert a mean')
        scale = st.sidebar.number_input('Insert a standard deviation')
        x2 = st.sidebar.number_input('Insert a X2')
        x = np.arange(-6, 10, 0.001)
        fig, ax = plt.subplots()
        greater_than=norm.sf(x=x2, loc=loc, scale=scale)
        ax.plot(x, norm.pdf(x, loc, scale))
        ax.set_title("Normal Distribution")
        ax.set_xlabel('X-Values')
        ax.set_ylabel('PDF(X)')
        ax.grid(True)
        sx=np.arange(x2,10, 0.01)
        ax.set_ylim(0, 0.25)
        ax.fill_between(sx, norm.pdf(sx, loc, scale), alpha = 0.25, color = 'red')
        ax.text(x2+2, 0.02, round(greater_than, 2), fontsize = 15)
        st.pyplot(fig)
    elif input=="X1 and X2 given and  total area between X1 and X2":
        loc = st.sidebar.number_input('Insert a mean')
        scale = st.sidebar.number_input('Insert a standard deviation')
        x1 = st.sidebar.number_input('Insert a X1')
        x2 = st.sidebar.number_input('Insert a X2')
        x = np.arange(-6, 10, 0.001)
        fig, ax = plt.subplots()
        equals=norm(loc, scale).cdf(x1) - norm(loc, scale).cdf(x2)
        ax.plot(x, norm.pdf(x, loc, scale))
        ax.set_title("Normal Distribution")
        ax.set_xlabel('X-Values')
        ax.set_ylabel('PDF(X)')
        ax.grid(True)
        sx=np.arange(x2,x1, 0.01)
        ax.set_ylim(0, 0.25)
        ax.fill_between(sx, norm.pdf(sx, loc, scale), alpha = 0.25, color = 'c')
        ax.text(x2+2, 0.02, round(equals, 2), fontsize = 15)
        st.pyplot(fig)
st.sidebar.header('Linear regression')
if st.sidebar.checkbox('Show regression?',False,key='5'):
    st.sidebar.header('Simple linear regression')
    @st.cache(allow_output_mutation=True)
    def get_data():
        return []
    X1 = float(st.sidebar.number_input("X"))
    Y1 = float(st.sidebar.number_input("Y"))
    if st.sidebar.button("Add row"):
        get_data().append({"X": X1, "Y": Y1})
    if st.sidebar.button('Show table'):
        st.table(pd.DataFrame(get_data()))
    if st.sidebar.button("Clear dataframe"):
        get_data().clear()
    df = pd.DataFrame(get_data())
    if st.sidebar.button('Show plot'):
        st.table(df)
        X=df.iloc[:, 0].values
        y=df.iloc[:, 1].values
        model=LinearRegression()
        X=X.reshape(-1,1)
        model.fit(X,y)
        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        fig = px.scatter(df, x='X', y='Y', opacity=0.85)
        fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
        st.plotly_chart(fig)
        st.write('Coefficient of determination is: ', model.score(X,y))
        st.write('Intercept is: ',model.intercept_)
        st.write('Slope is:', model.coef_)

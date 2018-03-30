import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression

from ipywidgets import interact, interactive

# Set the seaborn
plt.style.use('seaborn')


# create some data in Pandas dataframes
X, y = make_blobs(n_samples=80, centers=2,
                  random_state=0, cluster_std=0.60)
X = X.round(2)

labeled_data = pd.DataFrame(np.hstack([X[:50], y[:50, None]]),
                            columns=['feature1', 'feature2', 'color'])
unknown_data = pd.DataFrame(X[50:], columns=['feature1', 'feature2'])


def plot_data(df, slope=None, intercept=None):
    """Plot data along with an optional line"""
    style = dict(cmap='Paired', s=50, colorbar=False)
    if 'color' in df:
        ax = df.plot.scatter('feature1', 'feature2', c='color', **style)
    else:
        ax = df.plot.scatter('feature1', 'feature2', c='gray', **style)
    if slope is not None and intercept is not None:
        x = np.linspace(-1, 4)
        y = slope * x + intercept
        ax.plot(x, y, '-', color='black')    
    ax.axis([-1, 4, -1, 6])
    return ax


def plot_fit_model(df, slope, intercept):
    """Plot data, with color determined by model"""
    line = slope * df['feature1'] + intercept
    color = df['feature2'] < line
    plot_data(df.assign(color=color), slope, intercept)


def _plot_data_interactive(slope, intercept):
    plot_data(labeled_data, slope=slope, intercept=intercept)
    
    
interactive_model = interactive(_plot_data_interactive, slope=0.0, intercept=1.0)


def plot_regression_data():
    """Plot regression training data in 2D"""
    # Create some data for the regression
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    y = np.dot(X, [-2, 1]) + 0.1 * rng.randn(X.shape[0])
    
    # plot data points
    fig, ax = plt.subplots()
    points = ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
                        cmap='viridis')
    ax.axis([-4, 4, -3, 3])
    

def plot_3d_regression_data():
    """Plot regression training data in 3D"""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    
    # Create some data for the regression
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    y = np.dot(X, [-2, 1]) + 0.1 * rng.randn(X.shape[0])

    points = np.hstack([X, y[:, None]]).reshape(-1, 1, 3)
    segments = np.hstack([points, points])
    segments[:, 0, 2] = -8

    # plot points in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c=y, s=35,
               cmap='viridis')
    ax.add_collection3d(Line3DCollection(segments, colors='gray', alpha=0.2))
    ax.scatter(X[:, 0], X[:, 1], -8 + np.zeros(X.shape[0]), c=y, s=10,
               cmap='viridis')

    # format plot
    ax.patch.set_facecolor('white')
    ax.view_init(elev=20, azim=-70)
    ax.set_zlim3d(-8, 8)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())
    ax.set(xlabel='feature 1', ylabel='feature 2', zlabel='label')

    # Hide axes (is there a better way?)
    ax.w_xaxis.line.set_visible(False)
    ax.w_yaxis.line.set_visible(False)
    ax.w_zaxis.line.set_visible(False)
    for tick in ax.w_xaxis.get_ticklines():
        tick.set_visible(False)
    for tick in ax.w_yaxis.get_ticklines():
        tick.set_visible(False)
    for tick in ax.w_zaxis.get_ticklines():
        tick.set_visible(False)
        

def plot_linear_fit():
    """Plot regression training data along with linear fit"""
    # Create some data for the regression
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    y = np.dot(X, [-2, 1]) + 0.1 * rng.randn(X.shape[0])

    # fit the regression model
    model = LinearRegression()
    model.fit(X, y)

    # plot data points
    fig, ax = plt.subplots()
    pts = ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
                     cmap='viridis', zorder=2)

    # compute and plot model color mesh
    xx, yy = np.meshgrid(np.linspace(-4, 4),
                         np.linspace(-3, 3))
    Xfit = np.vstack([xx.ravel(), yy.ravel()]).T
    yfit = model.predict(Xfit)
    zz = yfit.reshape(xx.shape)
    ax.pcolorfast([-4, 4], [-3, 3], zz, alpha=0.5,
                  cmap='viridis', norm=pts.norm, zorder=1)
    ax.axis([-4, 4, -3, 3])
    

def plot_model_new_data():
    """plot new data with colors learned from the fit"""
    # Create some data for the regression
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    y = np.dot(X, [-2, 1]) + 0.1 * rng.randn(X.shape[0])

    # fit the regression model
    model = LinearRegression()
    model.fit(X, y)

    # create some new points to predict
    X2 = rng.randn(100, 2)

    # predict the labels
    y2 = model.predict(X2)
    
    # plot the model fit
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

    pts = ax[0].scatter(X2[:, 0], X2[:, 1], c='gray', s=50)
    ax[0].axis([-4, 4, -3, 3])

    ax[1].scatter(X2[:, 0], X2[:, 1], c=y2, s=50,
                  cmap='viridis', norm=pts.norm)
    ax[1].axis([-4, 4, -3, 3])

    # format plots
    ax[0].set_title('Unknown Data')
    ax[1].set_title('Predicted Labels')
    
    
def data_layout_figure():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.axis('equal')

    # Draw features matrix
    ax.vlines(range(6), ymin=0, ymax=9, lw=1)
    ax.hlines(range(10), xmin=0, xmax=5, lw=1)
    font_prop = dict(size=12, family='monospace')
    ax.text(-1, -1, "Feature Matrix ($X$)", size=14)
    ax.text(0.1, -0.3, r'n_features $\longrightarrow$', **font_prop)
    ax.text(-0.1, 0.1, r'$\longleftarrow$ n_samples', rotation=90,
            va='top', ha='right', **font_prop)

    # Draw labels vector
    ax.vlines(range(8, 10), ymin=0, ymax=9, lw=1)
    ax.hlines(range(10), xmin=8, xmax=9, lw=1)
    ax.text(7, -1, "Target Vector ($y$)", size=14)
    ax.text(7.9, 0.1, r'$\longleftarrow$ n_samples', rotation=90,
            va='top', ha='right', **font_prop)

    ax.set_ylim(10, -2)

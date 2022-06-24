import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spatial_interpolation.data import load_world_borders

def plot_interpolation(
    x,y,
    x_test,y_test,z_test,
    interpolator,
    num_points=100,
    df_countries=None,
    bbox=None,
    title=None,
    radius=0.1,
    zmin=None,
    zmax=None,
    ax=None,
    colorbar=True,
    map_args:dict=None,
    latlon=False,
    dim_cols=None,
    **kwargs
    ):
    """
    Plots the interpolation of the data as a heatmap and the data points
    """
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
    else:
        xmin, xmax = min(x.min(),x_test.min()), max(x.max(),x_test.max())
        ymin, ymax = min(y.min(),y_test.min()), max(y.max(),y_test.max())
    if df_countries is None:
        df_countries = load_world_borders()
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,10))
    else:
        fig = ax.get_figure()

    # create grid of points to interpolate
    X = np.linspace(xmin,xmax,num_points)
    Y = np.linspace(ymin,ymax,num_points)
    if latlon:
        X,Y = np.meshgrid(Y,X)
    else:
        X,Y = np.meshgrid(X,Y)
    # interpolate the grid
    if dim_cols:
        df = pd.DataFrame(dict(zip(dim_cols,[x.ravel() for x in [X,Y]])))
        Z = interpolator(df)
    else:
        Z = interpolator(X,Y)
    Z = Z.reshape(X.shape)
    if zmin is None:
        zmin = min(z_test.min(),Z[~np.isnan(Z)].min())
    if zmax is None:
        zmax = max(z_test.max(),Z[~np.isnan(Z)].max())
    zmin = kwargs.get("vmin",zmin)
    zmax = kwargs.get("vmax",zmax)
    # plot a contour of countries to add to the map
    map_args_default = dict(color=None, edgecolor="black")
    map_args_default.update(map_args or {})
    ax = df_countries.plot(ax=ax,zorder=1, **map_args_default)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    # plot the interpolation as a heatmap
    if latlon:
        CS = ax.contourf(Y,X,Z,zorder=0, **kwargs)
    else:
        CS = ax.contourf(X,Y,Z,zorder=0, **kwargs)
    if colorbar:
        fig.colorbar(CS, ax=ax)
    # show the training points
    ax.scatter(x,y,marker="o",s=5,zorder=10, c="black", vmin=zmin, vmax=zmax)
    # show the test points
    ax.scatter(x_test,y_test,c=z_test,marker="o",s=50,zorder=10,cmap=CS.cmap, vmin=zmin, vmax=zmax)
    # customize the plot
    for i,(x,y) in enumerate(zip(x_test,y_test)):
        cir = plt.Circle((x,y),radius=radius,color="red",zorder=10, fill=False)
        ax.add_artist(cir)
    ax.set_aspect("equal", adjustable="box")
    if title is None:
        title = f"{interpolator.__class__.__name__} interpolation"
    # buoys_gdf.plot(ax=ax,color='black',zorder=2)
    ax.set(title=title,xlabel="longitude",ylabel="latitude");
    return ax, CS
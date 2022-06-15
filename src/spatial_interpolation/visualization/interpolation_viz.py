import matplotlib.pyplot as plt
import numpy as np
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
    cmap=plt.cm.viridis,
    zmin=None,
    zmax=None
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
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))

    # create grid of points to interpolate
    X = np.linspace(xmin,xmax,num_points)
    Y = np.linspace(ymin,ymax,num_points)
    X,Y = np.meshgrid(X,Y)
    # interpolate the grid
    Z = interpolator(X,Y)
    Z = Z.reshape(X.shape)
    if zmin is None:
        zmin = min(z_test.min(),Z[~np.isnan(Z)].min())
    if zmax is None:
        zmax = max(z_test.max(),Z[~np.isnan(Z)].max())
    # plot a contour of countries to add to the map
    ax = df_countries.plot(ax=ax,zorder=1)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    # plot the interpolation as a heatmap
    CS = ax.contourf(X,Y,Z,cmap=cmap,zorder=0, vmin=zmin, vmax=zmax)
    fig.colorbar(CS)
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
    return CS
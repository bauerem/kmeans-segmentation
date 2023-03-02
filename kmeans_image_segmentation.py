import numpy as np

def f(x, save_colors_plot=False):
    from sklearn.cluster import KMeans
    shape = x.shape
    assert shape[0] > shape[-1]
    x = x.reshape(-1, shape[-1])
    y = np.empty_like(x)

    kmeans = KMeans(n_clusters=2, random_state=1).fit(x)

    colormap = kmeans.cluster_centers_

    for pixel, cluster in enumerate(kmeans.labels_):
        y[pixel] = colormap[cluster] #.astype(y.dtype)
    y = y.reshape(shape)

    if save_colors_plot == True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x[:,0],x[:,1],x[:,2],c=kmeans.labels_)
        plt.savefig("colorspace.png")

    return y

if __name__ == "__main__":
    package = "plt"
    save_colors_plot = True

    if package == "pillow":
        from PIL import Image
        input = Image.open('input.jpg')
    if package == "opencv":
        import cv2
        input = cv2.imread('input.jpg')
        # BGR -> RGB, TODO: learn why??
        #input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if package == "plt":
        import matplotlib.pyplot as plt
        input = plt.imread('input.jpg', format="jpg")

    # Print info about image
    #print(image.format, image.size, image.mode)


    # Convert image to array
    # TODO: https://stackoverflow.com/questions/14415741/what-is-the-difference-between-np-array-and-np-asarray
    input = np.array(input)

    output = f(input, save_colors_plot=save_colors_plot)

    # Convert array to image
    if package == "pillow":
        output = Image.fromarray(output).convert('RGB')
        output.save("output.jpg")
    if package == "opencv":
        cv2.imwrite("output.jpg", output)
    if package == "plt":
        plt.imsave('output.jpg', output)

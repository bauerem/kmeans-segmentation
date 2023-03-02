import numpy as np

def f(x, save_colors_plot=False):
    from sklearn.cluster import KMeans
    shape = x.shape
    ndims = len(shape)
    # If input is black and white image or video, add single color channel
    if ndims == 2 or ( ndims == 3 and shape[0] < shape[-1] ):
        x = x[...,np.newaxis]
        shape = x.shape
        ndims = len(shape)
    x = x.reshape(-1, shape[-1])
    y = np.empty_like(x)

    kmeans = KMeans(n_clusters=2, random_state=1).fit(x)

    colormap = kmeans.cluster_centers_

    for pixel, cluster in enumerate(kmeans.labels_):
        y[pixel] = colormap[cluster] #.astype(y.dtype)
    y = y.reshape(shape)

    ## Currently only works for images/frames
    if save_colors_plot == True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x[:,0],x[:,1],x[:,2],c=kmeans.labels_)
        plt.savefig("colorspace.png")

    return y

if __name__ == "__main__":
    package = "opencv"
    save_colors_plot = False

    if package == "pillow":
        from PIL import Image
        input = Image.open('input.jpg')
    if package == "opencv":
        import cv2
        #input = cv2.imread('input.jpg')
        # BGR -> RGB, TODO: learn why??
        #input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cap = cv2.VideoCapture('input.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int( cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ) ) )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            #frame = cv2.flip(frame, 0)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #cv2.imshow('frame',frame)
            input = np.array(frame)
            #print(input.shape)
            output = f(input, save_colors_plot=False)
            #frame = output

            out.write(output)
            cv2.imshow('frame', output)
            if cv2.waitKey(1) == ord('q'):
                print("Can't receive frame (stream end?). Exiting ...")
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


    exit()

    if package == "plt":
        import matplotlib.pyplot as plt
        input = plt.imread('input.jpg', format="jpg")

    # Print info about image
    #print(image.format, image.size, image.mode)


    # Convert image to array
    # TODO: https://stackoverflow.com/questions/14415741/what-is-the-difference-between-np-array-and-np-asarray
    input = np.array(input)

    #output = f(input, save_colors_plot=save_colors_plot)

    output = g()

    # Convert array to image
    if package == "pillow":
        output = Image.fromarray(output).convert('RGB')
        output.save("output.jpg")
    if package == "opencv":
        cv2.imwrite("output.jpg", output)
    if package == "plt":
        plt.imsave('output.jpg', output)

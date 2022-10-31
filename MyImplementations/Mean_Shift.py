import numpy as np

class Mean_Shift():
    def __init__(self, bandwidth=0.5, decimal_tolerance=3):
        self.bandwidth = bandwidth
        self.decimal_tolerance = decimal_tolerance
        self.tolerance = 10 ** (-decimal_tolerance)

    def gaussian_kernel(self, distance):
        if np.sum(np.abs(distance)) != np.sum(distance):
            raise ValueError("The input parameter distance, in method gaussian_kernel cannot be negative.")
        return np.exp(-distance/2)

    def flat_kernel(self, distance):
        return (distance < self.bandwidth).astype(int)

    def distance(self, A, B):
        return np.sqrt((A[:, 0] - B[:, 0])**2 + (A[:, 1] - B[:, 1])**2)

    def update_centroids(self, prev_centroids):
        for i, _ in enumerate(self.centroids):
            D = self.distance(self.centroids[i] * np.ones(shape=prev_centroids.shape), prev_centroids)
            #KD = self.gaussian_kernel(D/self.bandwidth)
            KD = self.flat_kernel(D)
            self.centroids[i] = np.sum(KD[:, np.newaxis]*prev_centroids, axis=0)/np.sum(KD)

    def data_normalization(self, data):
        self.normalized_data = data / (np.max(np.abs(data), axis=0) + np.min(np.abs(data), axis=0))


    def fit(self, data):
        # Normalize Data
        self.data_normalization(data)

        self.centroids = np.array(self.normalized_data)

        self.optimized = False

        while not self.optimized:

            prev_centroids = np.array(self.centroids)

            self.update_centroids(prev_centroids=prev_centroids)

            if np.sum(np.linalg.norm(self.centroids - prev_centroids, axis=1)) < self.tolerance:
                # Data denormalization
                self.centroids = self.centroids * (np.max(np.abs(data), axis=0) + np.min(np.abs(data), axis=0))

                # Optimization finished
                self.optimized = True
            else:
                # restrict the number of centroids to the unique ones only
                self.centroids = np.unique(np.round(self.centroids, decimals=self.decimal_tolerance), axis=0)

    def predict(self, data):
        pass
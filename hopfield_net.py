import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as ani

class Hopfield_Net:
    def __init__(self, X, y): #Init and Training
        self.X = X
        self.y = y
        self.n = X.shape[1] #neuron count
        self.num_pattern = self.X.shape[0]
        self.W = np.zeros((self.n, self.n))
        # If we want something similar to normalization
        #μ = np.sum([np.sum(x) for x in X]) / (self.n * self.num_pattern)
        self.all_states = [] #To use if reconstructing (animation)
        print("Starting training.")

        for i in range(self.num_pattern):
            x = self.X[i]# - μ
            self.W += np.outer(x.T, x)
        np.fill_diagonal(self.W, 0)
        self.W /= self.num_pattern

    def inference(self, X_noisy, n_iter, b = 0):
        """
        X_noisy -> noisy patterns
        n_iter -> number of iterations
        b -> threshold value
        """        
        print("Starting inference.")
        self.n_iter = n_iter
        self.b = b
        preds = []

        for i in range(X_noisy.shape[0]):
            preds.append(self.state_update(X_noisy[i]))
        return preds

    def energy(self, x): #Energy function to minimize
        return -0.5 * np.dot(x, np.dot(self.W, x)) + np.sum(x * self.b)

    def state_update(self, x, sync = False, track_states = True):
        """
        State update, either Synchronous or Asynchronous
        """
        x_ = x.copy()
        energy = self.energy(x_)
        self.states = [x_.copy()] #Tracking all states over iters
        
        if sync: #Synchronous update
            for _ in range(self.n_iter):
                x_ = np.sign(np.dot(self.W, x_) - self.b)
                energy_new = self.energy(x_)
                if track_states:
                    self.states.append(x_.copy())
                if energy_new == energy:
                    return x_
                energy = energy_new
            self.all_states.append(self.states)
            return x_
            
        else: #Asynchronous update
            for _ in range(self.n_iter):
                for _ in range(200): #num of neurons to update per iter
                    idx = np.random.randint(self.n)
                    x_[idx] = np.sign(np.dot(self.W[idx], x_) - self.b)
                    if track_states:
                        self.states.append(x_.copy())
                energy_new = self.energy(x_)
                if energy_new == energy:
                    self.all_states.append(self.states)
                    return x_
                energy = energy_new
            self.all_states.append(self.states)
            return x_
        
    
    def weight_plot(self): #Plot weights
        w = plt.imshow(self.W, cmap="hot")
        plt.colorbar(w)
        plt.axis("off")
        plt.show()
        
    def reconstruction(self, dif = 1, save_path = "patterns/HP_ani"):
        """
        Reconstruction of noisy patterns.
        """
        # dif : interval (ms)
        for i in range(len(self.all_states)):
            print(f"Reconstructing from noisy pattern {i}")
            ims = []
            fig, ax = plt.subplots()
            ims.append([ax.imshow(self.all_states[i][0].reshape((28, 28)), cmap="gray", animated = True)])
            ax.imshow(self.all_states[i][0].reshape((28, 28)), cmap="gray", animated = True)
            for j in range(1, len(self.all_states[i])):
                ims.append([ax.imshow(self.all_states[i][j].reshape((28, 28)), cmap="gray", animated = True)])
    
            anim = ani.ArtistAnimation(fig, ims, interval=dif, blit=True, repeat_delay=1000)
            plt.title("Reconstruction")
            plt.axis("off")
        
            if save_path:
                anim.save(f"{save_path}_pat{i}.gif", writer="pillow", fps=1000/dif)
            plt.show()

    def original_patterns(self, save_path = "patterns/pattern"):
        #Save original patterns
        plt.figure(figsize=(4, 4))
        for i in range(self.num_pattern):
            plt.imshow(self.X[i].reshape(28, 28), cmap="gray")
            plt.title(f"Original, label: {self.y[i]}")
            plt.axis("off")
            plt.savefig(f"{save_path}_{i}.png")
            plt.close()
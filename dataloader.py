import numpy as np
from numpy.random import shuffle as random_shuffle
from numpy.random import normal

class DataLoader(object):
    def __init__(self, normal_files, adversarial_files=None, fixed_length = None, do_difference = False, reshape_3d = True, binary_mode = True, autoencoder_mode=False, inject_noise=False, shuffle=True):
        self.final_data = np.array([])
        self.autoencoder_mode = autoencoder_mode
        self.binary_mode = binary_mode
        self.reshape_3d = reshape_3d
        self.do_difference = do_difference
        self.fixed_length = fixed_length
        self.inject_noise = inject_noise
        self.noise_std = 0.1
        self.shuffle = shuffle

        for nf in normal_files:
            d = np.loadtxt(nf, delimiter=",")
            labels = np.empty(d.shape[0])
            labels.fill(0)
            #d[:,0] = labels #set the labels to 0, meaning 'normal'

            #then sort out the length
            if not self.fixed_length is None:
                #extend or truncate
                if len(d[0][1:]) > self.fixed_length:
                    d = d[:,0:fixed_length + 1]
                elif (len(d[0][1:] < self.fixed_length)):
                    d_new = np.zeros((d.shape[0],self.fixed_length + 1))
                    d_new[:, 0:self.fixed_length + 1] = d
                    d = d_new
            
            if (len(self.final_data) == 0):
                self.final_data = d
            else:
                self.final_data = np.vstack((self.final_data, d))


        for idx in range(len(adversarial_files)):
            name = adversarial_files[idx]
            l = None
            if "BIM" in name:
                l = 1
            else:
                l = 2 #FGSM.todo: tag the adversarial files as being BIM or FGSM explicitly

            d = np.loadtxt(adversarial_files[idx], delimiter=",")
            labels = np.empty(d.shape[0])
            labels.fill(l)
            d[:,0] = labels #set the labels to 1..., meaning 'bim, fgsm, whatever'

            #then sort out the length
            if not self.fixed_length is None:
                #extend or truncate
                if len(d[0][1:]) > self.fixed_length:
                    d = d[:,0:fixed_length + 1] #start from 0 to preserve the label information
                elif (len(d[0][1:] < self.fixed_length)):
                    d_new = np.zeros((d.shape[0],self.fixed_length + 1))
                    d_new[:, 0:self.fixed_length + 1] = d
                    d = d_new
           

            if (len(self.final_data) == 0):
                self.final_data = d
            else:
                self.final_data = np.vstack((self.final_data, d))
        
        if self.shuffle:
            random_shuffle(self.final_data) #randomize
        
        self.len = len(self.final_data)

        if (self.autoencoder_mode):
            self.n_classes = -1 #not applicable
        elif (self.binary_mode):
            self.n_classes = 2 #its a binary problem now
        else:
            self.n_classes = len(set(self.final_data[:,0])) #compute it from the anomalous classes
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dataitem = self.final_data[index]
        real_label = dataitem[0]
        real_data = dataitem[1:]
        other_data = real_data #for noisy purposes if desired

        if self.inject_noise:
            noise_ = normal(0,scale=self.noise_std,size=len(real_data))
            other_data = noise_ + real_data

        if self.do_difference:
            real_data = np.diff(real_data)
            other_data = np.diff(other_data)

        if (self.reshape_3d):
            real_data = real_data.reshape((-1,1))
            other_data = other_data.reshape((-1,1))
        
        if (self.autoencoder_mode):
            return (real_data, other_data)
        
        if (self.binary_mode):
            binary_label = 0 if real_label == 0 else 1
            return (real_data, binary_label )
        else:
            return (real_data, real_label)
"""
Weather Generator GAN developed for NOAA Hackathon
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
torch.manual_seed(0) # Set for testing purposes, please do not change!

def parse_arguments():    
    import argparse    
         
    parser = argparse.ArgumentParser()    
    parser.add_argument(    
        "--visualize", action="store_true", help="visualize results at the end of training"    
    )    
    parser.add_argument(    
        "--profile", action="store_true", help="profile with nsight systems"    
    )    
    parser.add_argument(    
        "--profile-start-iter",    
        type=int,    
        default=50,    
        help="start iteration for nsys profile",    
    )    
    parser.add_argument(    
        "--profile-end-iter",    
        type=int,    
        default=70,    
        help="end iteration for nsys profile",    
    )    
    parser.add_argument(    
        "--gen-hidden-dim", type=int, default=8, help="hidden dimension of generator layers"    
    )    
    parser.add_argument(    
        "--disc-hidden-dim", type=int, default=8, help="hidden dimension of discriminator layers"    
    )    
    parser.add_argument(    
        "--max-epochs", type=int, default=1000, help="number of training epochs"    
    )    
    parser.add_argument(    
        "--batch-size", type=int, default=256, help="Batch size"    
    )    
    parser.add_argument(    
        "--noise-dim", type=int, default=2, help="Dimensionality of noise vector"    
    )    
    args = parser.parse_args()    
    print(f"Parameters: {args}")    
    return args

#df = pd.read_csv('data/Monthly_Average_1950_2009_reservoir.csv')
df = pd.read_csv('data/daily_data_1950_2009_reservoir.csv', usecols=range(3, 9))

# GRADED FUNCTION: get_generator_block
def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.Dropout(p=0.25),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=2, im_dim=6, hidden_dim=8):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.ReLU(),
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device 
    # argument to the function you use to generate the noise.
    return torch.randn(n_samples,z_dim,device=device)

def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
         nn.Linear(input_dim, output_dim), #Layer 1
         nn.LeakyReLU(0.2, inplace=True)
    )

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=6, hidden_dim=8):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    
    # Needed for grading
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc


#def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
def get_disc_loss(fake, disc, criterion, real):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        fake: a batch of fake images
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Get the discriminator's prediction of the fake image 
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a 
    #            'ground truth' tensor in order to calculate the loss. 
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       2) Get the discriminator's prediction of the real image and calculate the loss.
    #       3) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    #fake_noise = get_noise(num_images, z_dim, device=device)
    #fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

#def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
def get_gen_loss(fake, disc, criterion):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        fake: a batch of fake images
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Get the discriminator's prediction of the fake image.
    #       2) Calculate the generator's loss. Remember the generator wants
    #          the discriminator to think that its fake images are real

    #fake_noise = get_noise(num_images, z_dim, device=device)
    #fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss

if __name__ == "__main__":
    # Get command line arguments
    args = parse_arguments()
    batch_size = args.batch_size
    n_epochs = args.max_epochs
    z_dim = args.noise_dim

    tensor_x = torch.Tensor(df.values) # transform to torch tensor
    dataset = TensorDataset(tensor_x) # create your datset
    dataloader = DataLoader(dataset,
        batch_size=batch_size,
        shuffle=True) # create your dataloader
    
    # Set your parameters
    criterion = nn.BCEWithLogitsLoss()
    display_step = 25
    lr = 0.00001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    
    gen = Generator(z_dim=z_dim, hidden_dim=args.gen_hidden_dim).to(device)
    disc = Discriminator(hidden_dim=args.disc_hidden_dim).to(device) 

    # git script
    gen = torch.jit.script(gen)
    disc = torch.jit.script(disc)

    # git trace
    #fake_noise = get_noise(batch_size, z_dim, device=device)
    #disc = torch.jit.trace(disc, gen(fake_noise))
    #gen = torch.jit.trace(gen, fake_noise) 

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    
    generator_loss = 0
    discriminator_loss = 0
    test_generator = False # Whether the generator should be tested
    error = False
    history = []
    
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    
    for epoch in range(n_epochs):
        fake__ = []
        real__ = []
        # Dataloader returns the batches
        if torch.cuda.is_available():
            start_event.record()
        else:
            start_time = time.time()
        #for real in tqdm(dataloader):
        for i, real in enumerate(dataloader):
            cur_batch_size = len(real[0])
            #print(cur_batch_size)
            real_ = real[0].to(device)
            # Flatten the batch of real images from the dataset
            #real = real.view(cur_batch_size, -1).to(device)
    
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
    
            ### Update discriminator ###
            # Zero out the gradients before backpropagation
            disc_opt.zero_grad()
    
            # Calculate discriminator loss
            #disc_loss = get_disc_loss(gen, disc, criterion, real_, cur_batch_size, z_dim, device)
            disc_loss = get_disc_loss(fake, disc, criterion, real_)
    
            # Update gradients
            disc_loss.backward(retain_graph=True)
    
            # Update optimizer
            disc_opt.step()
    
            # For testing purposes, to keep track of the generator weights
            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()
    
            ### Update generator ###
            gen_opt.zero_grad()
            #gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss = get_gen_loss(fake, disc, criterion)
            gen_loss.backward()
            gen_opt.step()
    
            # For testing purposes, to check that your code changes the generator weights
            if test_generator:
                try:
                    assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")
    
            real__.append(real_.cpu().detach().numpy())
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake__.append(gen(fake_noise).cpu().detach().numpy())
    
            # Track loss
            generator_loss += gen_loss.item()
            discriminator_loss += disc_loss.item()
    
        generator_loss /= len(dataloader)
        discriminator_loss /= len(dataloader)
    
        ### Visualization code ###
        if epoch % display_step == 0:
            print(f"epoch {epoch}: Generator loss: {generator_loss}, discriminator loss: {discriminator_loss}")
    
        ### History ###
        history.append({'gen_loss': generator_loss, 'disc_loss': discriminator_loss})
    
        if torch.cuda.is_available():
       	    end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
        else:
            elapsed_time = (time.time() - start_time)*1000
    
        print(f'Epoch {epoch}, time = {elapsed_time} ms')
    
    history = pd.DataFrame(history)
    
    # Evaluate the model
    fake_noise = get_noise(720, z_dim, device=device)
    fake = gen(fake_noise).cpu().detach().numpy()
    #np.savetxt('fake.txt', fake, delimiter=',')
    
    #real = pd.read_csv('../data/Monthly_Average_1950_2009_reservoir.csv').values
    real = df.values
    
    # mean error (%)
    mean_error = (fake.mean(axis=0) - real.mean(axis=0)) / real.mean(axis=0) * 100
    
    # std error (%)
    std_error = (fake.std(axis=0) - real.std(axis=0)) / real.std(axis=0) * 100
    
    # Combine
    stats_df = pd.DataFrame([mean_error, std_error],
                            columns=df.columns,
                            index=['Mean error', 'Std error'])
    
    # Mean absolute percent error
    mape_df = stats_df.abs().mean(axis=1)
    
    with pd.option_context('display.float_format', '{:.2f}%'.format, 'display.expand_frame_repr', False):
        print('---Individual watersheds---')
        print(stats_df)
        print('---Summary (MAPE)---')
        print(mape_df)
    
    # Calculate empirical CDFs
    bins = np.linspace(0, 15, 31)   # Maximum 15 mm for now
    binsize = bins[1:] - bins[:-1]
    
    real_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=bins, density=True)[0], 0, real)
    real_ecdf = np.cumsum(real_hist, axis=0) * binsize[:, np.newaxis]
    
    fake_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=bins, density=True)[0], 0, fake)
    fake_ecdf = np.cumsum(fake_hist, axis=0) * binsize[:, np.newaxis]
    
    # Continuous ranked probability score (CRPS)
    crps = np.sum(np.abs((real_ecdf - fake_ecdf) * binsize[:, np.newaxis]), axis=0)
    crps_df = pd.DataFrame([crps], columns=df.columns, index=['CRPS'])
    
    with pd.option_context('display.float_format', '{:.2f}'.format, 'display.expand_frame_repr', False):
        print('---Individual watersheds---')
        print(crps_df)
        print('---Mean CRPS---')
        print(f'{crps.mean():.2f}')
    
    
    if args.visualize:
        # Make plots
        import seaborn as sns
        import matplotlib.pyplot as plt 
        
        dict_ = {'data':real[:,0], 'type':'real', 'station': '1'}
        df = pd.DataFrame(dict_)
        for i in range(2,7):
            dict_ = {'data':real[:,i-1], 'type':'real', 'station': str(i)}
            df = df.append(pd.DataFrame(dict_))
        
        for i in range(1,7):
            dict_ = {'data':fake[:,i-1], 'type':'fake', 'station': str(i)}
            df = df.append(pd.DataFrame(dict_))
        
        sns.boxplot(data=df, x="station", y="data", hue="type")
        plt.savefig('boxplot.png', dpi=500)
        sns.violinplot(data=df, x="station", y="data", hue="type")
        plt.savefig('violinplot.png', dpi=500)

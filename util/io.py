
import os
import tifffile as tiff
import numpy as np
import torch

from util.tools import load_net
from util.preprocessing import normalize

# reads tif formatted file and returns the data in it as a numpy array
def read_tif(file, dtype='uint8'):

    data = tiff.imread(file).astype(dtype)

    return data

# write a 3D data set to a directory (slice by slice)
def imwrite3D(x, dir, prefix='', rescale=False):
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(0,x.shape[0]):
        if rescale:
            tiff.imsave(dir + '/' + prefix + str(i) + '.tif', (x[i,:,:] * 255).astype('uint8'))
        else:
            tiff.imsave(dir + '/' + prefix + str(i) + '.tif', (x[i, :, :]).astype('uint8'))

# write out the activations of a segmentation network for a specific input
def write_activations(model_file, x, write_dir):

    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    xn = normalize(x, np.max(x), np.max(x) - np.min(x))
    tiff.imsave(os.path.join(write_dir, 'input.tif'), (xn * 255).astype('uint8'))

    # transform data to cuda tensor
    x = torch.FloatTensor(x[np.newaxis, np.newaxis, ...]).cuda()

    # load network
    net = load_net(model_file=model_file)
    net.eval()
    net.cuda()

    # apply forward prop and extract network activations
    encoder_outputs, encoded_outputs = net.encoder(x)
    decoder_outputs, final_outputs = net.decoder(encoded_outputs, encoder_outputs)

    # write random activations
    for i, encoder_output in enumerate(encoder_outputs):
        c = np.random.randint(encoder_output.size(1))
        act = encoder_output[0, c, :, :].data.cpu().numpy()
        act = normalize(act, np.max(act), np.max(act) - np.min(act))
        tiff.imsave(os.path.join(write_dir, 'enc_act_'+str(len(encoder_outputs)-i-1)+'.tif'), (act * 255).astype('uint8'))

    c = np.random.randint(encoded_outputs.size(1))
    act = encoded_outputs[0, c, :, :].data.cpu().numpy()
    act = normalize(act, np.max(act), np.max(act) - np.min(act))
    tiff.imsave(os.path.join(write_dir, 'enc_act_'+str(len(encoder_outputs))+'.tif'), (act * 255).astype('uint8'))

    for i, decoder_output in enumerate(decoder_outputs):
        c = np.random.randint(decoder_output.size(1))
        act = decoder_output[0, c, :, :].data.cpu().numpy()
        act = normalize(act, np.max(act), np.max(act) - np.min(act))
        tiff.imsave(os.path.join(write_dir, 'dec_act_'+str(len(decoder_outputs)-i-1)+'.tif'), (act * 255).astype('uint8'))
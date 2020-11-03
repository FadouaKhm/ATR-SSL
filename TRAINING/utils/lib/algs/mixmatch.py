import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def get_augmenter():
    seq = iaa.Sequential([iaa.Fliplr(0.5), # horrizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
    ])
    def augment(images):
        # Only works with list. Convert np to list
        imgs = []
        for i in range(images.shape[0]):
            imgs.append(images[i,:,:])

        images = images

        return seq.augment(images=images)
    return augment




class MixMatch(nn.Module):
    def __init__(self, temperature, n_augment, alpha):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.K = n_augment
        self.xent = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.beta_distirb = torch.distributions.beta.Beta(alpha, alpha)

    def sharpen(self, y):
        y = y.pow(1/self.T)
        return y / y.sum(1,keepdim=True)

    def forward(self, x, y, model, mask, num_cls):

        #augment_fn = get_augmenter()
        # NOTE: this implementaion uses mixup for only unlabeled data
        model.update_batch_stats(False)
             
        xb = x[mask == 0]
        n = torch.randn_like(xb) * 0.15
        xb = n + xb
        ##xb = xb.flip(-1)
        #print(xb.shape)
        u_x = x[mask == 1]
        

        # K augmentation and make prediction labels
        u_x_hat = [u_x.clone().detach().requires_grad_(True) + torch.randn_like(u_x) * 0.15 for _ in range(self.K)]#Ux
        
        ##u_x_hat = [u_x.clone().detach().requires_grad_(True) + torch.randn_like(u_x) * 0.15, u_x.flip(-1)]#Ux
        #u_x_hat = [augment_fn(u_x)  for _ in range(self.K)]#Ux
        y_hat = sum([model(u_x_hat[i])[0].softmax(1) for i in range(len(u_x_hat))]) / self.K
        y_hat = self.sharpen(y_hat)
        y_hat = y_hat.repeat(len(u_x_hat), 1)#Uy
        # mixup
        #u_x_hat = torch.tensor(u_x_hat)
        u_x_hat = torch.cat(u_x_hat,0)
        
        #print(u_x_hat.shape)
        y = torch.nn.functional.one_hot(y[mask == 0], num_classes=num_cls).float()
        #print(y[mask == 0].shape)

        Wx = torch.cat((u_x_hat,xb),0)#Ux
        Wy = torch.cat((y_hat,y.float()),0)#Ux
        index = torch.randperm(Wx.shape[0])
        shuffled_u_x_hat, shuffled_y_hat = Wx[index], Wy[index]
        lam = self.beta_distirb.sample().item()
        lam = max(lam, 1-lam)
        
        U = lam * u_x_hat + (1-lam) * shuffled_u_x_hat[len(xb):]
        q = lam * y_hat + (1-lam) * shuffled_y_hat[len(xb):].softmax(1)

        X = lam * xb + (1-lam) * shuffled_u_x_hat[:len(xb)]
        p = lam * y + (1-lam) * shuffled_y_hat[:len(xb)].softmax(1)

        # mean squared error
        #[out_mix, feat_mix] = model(mixed_x)
        #print(U.shape)
        X_ = torch.cat((X, U),0)
        y_ = torch.cat((p, q),0)

        preds = model(X_)[0]


        p1 = p.max(1)[1]
        q1 = q.max(1)[1]
        
        q11 = torch.nn.functional.one_hot(q1, num_classes=num_cls).float()
        

        #loss = F.mse_loss(preds[len(p):], q) + 
        cls_loss = F.cross_entropy(preds[:len(p)], p1, reduction="none", ignore_index=-1).mean()
        ssl = self.alpha*F.mse_loss(preds[len(p):], q11)

        #print(ssl)
        #print(cls_loss)

        #loss = cls_loss + ssl
        model.update_batch_stats(True)
        return cls_loss, ssl, y_hat

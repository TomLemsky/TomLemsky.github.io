---
layout: post
title:  "Reimplementing Deep Image Prior"
date:   2022-05-04 11:00:00 +0100
---

The paper [Deep Image Prior by Ulyanov et al.](https://dmitryulyanov.github.io/deep_image_prior) is extremely interesting because it shows that even without pretraining, convolutional neural networks are a useful prior in computer vision. For example for image inpainting, they start with a randomly initialized network and train it to produce the input image. The loss is not applied to the area to be inpainted and therefore whatever pixels the network produces there is a byproduct of training it to produce the other parts of image.

Implementation
===========

The Ulyanov et al. use network architectures based on the U-Net segmentation architecture. To implement their architectures, I generalized the U-Net I implemented in PyTorch during my master's thesis, such that it can have arbitrary numbers of layers and number of features per layer. The number of output classes is set to three and the only activation function used is LeakyReLU (especially SoftMax is not used).

For the input of the model, I implemented a small class called *Learned Input* that simply outputs its learned parameters. As in the original paper, the parameters are initialized randomly between 0.0 and 0.1.

{% highlight Python %}
class LearnedInput(nn.Module):
    """Simply outputs the learned Parameters of the specified dimensions"""
    def __init__(self, dimensions):
        super(LearnedInput, self).__init__()
        input = torch.rand(dimensions)*0.1
        self.learned_input = nn.parameter.Parameter(data=input)
    def forward(self,x):
        return self.learned_input
{% endhighlight %}

Using this approach, any segmentation model could be used for inpainting. But one thing to take into account is that the pixels in the LearnedInput that correspond to the mask must be within the receptive field of pixels that aren't masked. Otherwise these pixels wouldn't receive any gradient and wouldn't be trained at all. For this reason, the expanding path (the decoder) of the U-Net uses 5x5 convolutions instead of the normal 3x3, doubling the size of the receptive field.

Training
===========

| ![](/images/deep-image-prior/hase_inpainted_fail.jpg) | ![](/images/deep-image-prior/inpainted_image_kinda_good.jpg) |
|:--:|
| *An image that was trained too long* | *Example for a colorful ellipse appearing in the image* |

As in the original paper, the network was trained using an Adam optimizer with learning rate 0.1 and the standard beta values of 0.9 and 0.999. Training was done for around 6000 iterations. It is important not to train for too long, as otherwise high-contrast artifacts appear in the masked region, while the network is trying to match the rest of the image exactly. For the same reason scheduling and learning rates that are too small should be avoided. Another artifact were colorful ellipses that appeared in the masked regions, perhaps because these positions in the input were not in the receptive field of any unmasked pixels.

Looking at the outputs of the network during training is very interesting. First, large regions of single colors appear. The shape of these regions is iteratively refined during training and details are added. The resulting images look very convincing and the inpainted region does not stand out from the rest. The only problem is that my resulting images look a bit blurry when zoomed in. But it wasn't possible to train the network for longer, since otherwise artifacts would appear in the inpainted region.

Masked image            |  Output during training | Finished output
:-------------------------:|:-------------------------:|:-------------------------:
![](/images/deep-image-prior/eisbaer_small_masked.jpg)  |  ![](/images/deep-image-prior/eisbaer.gif) | ![](/images/deep-image-prior/eisbaer_inpainted.jpg)
![](/images/deep-image-prior/hase2_small_masked.jpg)  |  ![](/images/deep-image-prior/hase.gif) | ![](/images/deep-image-prior/hase_inpainted.jpg)

My code is available here: [https://github.com/TomLemsky/deep-image-prior](https://github.com/TomLemsky/deep-image-prior)

Example images credit
===========

Water image: CC-by-sa, Author: Mbz1, [https://commons.wikimedia.org/wiki/File:Polar_bear_arctic.JPG](https://commons.wikimedia.org/wiki/File:Polar_bear_arctic.JPG)

Grass image: Public domain, Author: U.S. Fish and Wildlife Service, [https://commons.wikimedia.org/wiki/File:New_England_cottontail.jpg](https://commons.wikimedia.org/wiki/File:New_England_cottontail.jpg)

Space Shuttle landing facility: Public domain, Author: NASA, [http://www.collectspace.com/news/news-062215a-shuttle-landing-facility-handover.html](http://www.collectspace.com/news/news-062215a-shuttle-landing-facility-handover.html)

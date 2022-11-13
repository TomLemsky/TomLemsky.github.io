---
layout: post
title:  "Minecraft skin generation using Diffusion (with web demo)"
date:   2022-11-13 20:00:00 +0100
---

[Try my skin generation model here: https://huggingface.co/spaces/TomLemsky/this_skin_does_not_exist](https://huggingface.co/spaces/TomLemsky/this_skin_does_not_exist)

After I collected a data set of about 200 000 Minecraft skins and trained classifiers on it ([See this previous post](/2022/03/30/Classifying-Minecraft-Skins-by-Gender.html)),
the obvious next step was to train a generative model on the same data.
I wanted to show that one can train interesting models using low-level graphics cards like my GTX 1660Ti by collecting your own data set.

## Initial attempts

To confirm skin generation was feasible given my dataset I quickly checked out [this PyTorch Denoising diffusion library](https://github.com/lucidrains/denoising-diffusion-pytorch).
For simplicity of implementation, I just converted to skins from RGBA to RGB, turning transparent regions black.
The results already looked pretty okay after 30 000 iterations of training:

<img src="/images/minecraft-diffusion/diffusion1.png" style="image-rendering: pixelated; display: block; margin-left: auto; margin-right: auto" alt="The RGB model outputs" width="500"/>

But transparency is pretty critical for Minecraft skins,
since the whole second layer of the skin needs transparency to make things like
hair, helmets and outerwear.
So I did another training run with transparency.
The model again found the correct shapes of a Minecraft skin,
but it completely lacked details such as eyes.

<img src="/images/minecraft-diffusion/diffusion2.png" style="image-rendering: pixelated; display: block; margin-left: auto; margin-right: auto" alt="The initial RGBA model outputs" width="500"/>

In hindsight, I should have just trained this model longer at a much lower training rate,
but at the time I thought the error must be because the model was trying to get the alpha channel in the transparent regions as close to zero as possible.
To combat this hypothesized problem, I modified the source code of [denoising_diffusion_pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
to implement a loss function so that it would weight the alpha channel less than the other channels.
This approach did not work out and the model seemed not to be learning much at all, possibly because of an error in my implementation of the cost function:

<img src="/images/minecraft-diffusion/diffusion3.png" style="image-rendering: pixelated; display: block; margin-left: auto; margin-right: auto" alt="The weighted RGBA model outputs" width="500"/>

## Final model and web app

Finally, I realized I was overthinking this and just trained an RGBA model with similar hyperparameters as the original RGB model.
When I finally lowered the learning rate enough, details like eyes started to appear and I was seeing transparency in the correct places.
I gradually lowered the learning rate when the image quality did not seem to improve anymore and left the model train for two nights.
This produced very nice looking outputs after 160 000 iterations of training.

The only problem is that the transparent parts are often noisy, since these regions are mostly empty for many images in the data set.
But for some samples the second layer produces pretty okay looking hair, helmets and jackets.
I confirmed that the model was not just repeating examples from the training set by
generating many samples and finding the training image with the smallest L2 distance.

<img src="/images/minecraft-diffusion/diffusion4.png" style="image-rendering: pixelated; display: block; margin-left: auto; margin-right: auto" alt="The final model outputs" width="500"/>

Now that I had a trained model, I needed to make it accessible somewhere for others to use.
I chose [huggingface spaces](https://huggingface.co/spaces) because it is free and simple to use.
Gradio made it very easy to build an interface for my model, but was also very customizable.

2D Minecraft skins are a bit boring and unintuitive to look at, so I wanted to display them as a 3D Minecraft character.
I found [another huggingface space](https://huggingface.co/blog/spaces_3dmoljs) that displays 3D models and got the idea to use iframes inside of `gradio.HTML()` elements.
Inside of the iframes, I used the [skinview3d JavaScript library](https://github.com/bs-community/skinview3d) to display the skins in 3D.
The skins were injected into the iframe HTML as base64 data using `gradio.processing_utils.encode_array_to_base64(i)`.

Now that my interface was ready and tested locally, I uploaded it to huggingface spaces.
Despite my fiber internet, the 600MB model file took all night to upload.
When I tried the skin generation website the next day, I was happy to see that it worked.
But it took many times longer to generate skins than on my local machine because the free huggingface spaces tier did not include a GPU.
A 5 minute wait is not acceptable for users, so I played around with the number of diffusion steps
and found out that 35 to 50 steps with DDIM sampling worked very well.
So I lowered the step number to this and included some default generations for users to look at while waiting.

You can try out the final skin generation model below or under this link: [https://huggingface.co/spaces/TomLemsky/this_skin_does_not_exist](https://huggingface.co/spaces/TomLemsky/this_skin_does_not_exist)

<script type="module"
src="https://gradio.s3-us-west-2.amazonaws.com/3.9.1/gradio.js">
</script>

<gradio-app space="TomLemsky/this_skin_does_not_exist"></gradio-app>

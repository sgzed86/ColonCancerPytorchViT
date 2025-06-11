# Lesion Detection, Polyp Characterization and fibrosis grading ViT
I think I will try it on the hyperkvasir dataset instead.

I downloaded the hyperkvasir dataset and set up the file format for training a pytorch ViT model. 56gb

The goal of this model will be to build a vision transformer for lesion detection, polyp characterization, and fibrosis grading.

First, a lot of work went into the data loader to get the data into python.  

I ended up going the timm hugging face route.

It took 4 hours to run 1 epoch so, I need to look at how to optimize running this code.

Used automatic mixed precision training which cut the epoch time in half. Added gradient accumulation increase batch size without increasing memory and used persistent workers and pin memory for faster CPU to GPU
I am using a laptop with NVIDIA RTX 3050. It is running the model but I might be able to get it down to 1 hour per epoch which is still way too long. I should have bought the Jetson...
This is a rather ambitous project for this laptop.

I ran just 1 epopch to test it out, took about 50 minutes. Even on 1 epoch the model does not seem too bad.

+ Lesion Accuracy:   0.8009
+ Polyp Accuracy:    0.9744
+ Fibrosis Accuracy: 0.9762



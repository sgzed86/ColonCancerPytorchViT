# Lesion Detection, Polyp Characterization and fibrosis grading ViT
I think I will try it on the hyperkvasir dataset instead.

I downloaded the hyperkvasir dataset and set up the file format for training a pytorch ViT model. 56gb

The goal of this model will be to build a vision transformer for lesion detection, polyp characterization, and fibrosis grading.

First, a lot of work went into the data loader to get the data into python.  

I ended up going the timm hugging face route.

It took 4 hours to run epoch so, I need to look at how to optimize running this code.

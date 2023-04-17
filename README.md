# Work Products
Work products from some freelance jobs that I retained ownership over. The data is rarely my property so the code can't actually be run, but some writeups and graphs are shared along with the code.

# Gate Profile Research

filename: gate_analysis.ipynb

Accelerometers were placed on adolescences' legs as they walked on a treadmille and outside on the ground. This allows us to graph how different parts of their leg move throughout the path of their gate. Analysis was required to smooth and filter out bad data, and then performed to compare signals from right and left legs as well as from indoor to outdoor.

Skills/Tools: Butterworth filter, euclidean distance, cosine similarity, cross-correlation, data analysis, python, data cleaning and exploration, 

# LDA

Started with some ML code written by someone else. Modularized the code to improve testing, readability, and portability. Implemented hyperparameter choice of LDA perplexity from either gensim and sklearn modules. 

# Spectrum Analysis

Written as a proof of concept of the process that would be required to detect the difference between the audio files generated when a fake coin and a real coin are struck. Each file contains audio for only one coin. The difference in the peak of the FFT is clear and easily identifiable between the two, and this analysis was done in < 1 hour.

# [N Template Matching](https://github.com/Alexander-philip-sage/n_template_match_gpu)

OpenCV has a cuda implementation of template matching, but it only parallelizes the convolution of the algorithm. It doesn't provide a function for the common use case of N template-image pairs to be parallelized - thus utilizing only a very small percent of the GPU and possibly slowing down the function compared to the CPU given the data loading onto the GPU mem. 

# [MB Aligner](https://github.com/Alexander-philip-sage/mb_aligner)

Working on speeding up mb_aligner repo to match electron microscopy images of slices of brain tissue. 

# Thesis: Impact estimation as well as band selection for classification of high-resolution hyperspectral data

In order to proceed with this experiment, install the dependencies with the command pip install -r requirements.txt at the terminal, then replace the file paths and band size (here 2402) in all three files mentioned below and (If doesn"t have access to the dataset). After making sure there is a working dataset, run in following order:

1. varianceTest.py
2. band_Select_test.py
3. main.py

varianceTest.py run a variance test on 10 classifier instances of each model (Gaussian based smoothing feature reduced, Band wise Averaging feature reduced, Whole spectral) that generate guided gradhms of each class that'll be saved and use with the other two files. The guided gradhms is average by classifier instances, such that we obtain 10 averaged guided gradhms that combine the results from all labels. A band wise variance is then calculated to measure the fluctuation between classifier instances. 

band_select_test.py run an accuracy test that measure the accuracy of each model (Gaussian based smoothing feature reduced, Band wise Averaging feature reduced, Whole spectral, Gaussian based smoothing band selected, Band wise Averaging band selected, Whole spectral band selected). The band selection algorithm selects the bands based on the bands that contribute the most in the classification by the saliency maps (Guided gradhms) from above. If another amount of bands to select is desired, can adjust the parameter when calling for the band_select function. 

main.py plots out the diagrams that are used in the thesis.

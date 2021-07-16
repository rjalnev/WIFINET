# WIFINet
A neural network, adapted from WaveNet that utilizes dilated convolutions.
Used to classify coexisting Wi-Fi technologies using raw power measurements.
Wi-Fi power measurements were collected across 802.11n, 802.11ac, and 802.11ax wireless technologies both individually and with multiple technologies coexisting across a range of various throughputs.

# Usage
Run test.py to load trained model and test on test dataset.
Model and test dataset can be found [here](https://drive.google.com/drive/folders/1lxk85Lu7JByLs4hyXY3XmtLsCEkX1d7u?usp=sharing).
Place wifinet.h5 and wifinet_history.csv into /model directory and test.npz into /data directory.

# Dataset
The dataset consited of 118,125 samples randomly sampled from 585 tests collecting data across a wide variety of scenarios. Only the in-pahse component was used. Dataset was split into 94,500 training and 23,625 validation sets. See process_data.py for how data was preprocessed and load_data() in main.py for normalization (Norm 1 used).

| Raw               | Normalized        |
| ----------------- | ----------------- |
|<img src=images/samples_raw.png width="400">|<img src=images/samples_normalized1.png width="400">|



# Results
| Loss              | Accuracy          |
| ----------------- | ----------------- |
|<img src=images/wifinet_loss.png width="400">|<img src=images/wifinet_acc.png width="400">|

| ROC / AUC         | Confusion Matrix  |
| ----------------- | ----------------- |
|<img src=images/wifinet_roc.png width="400">|<img src=images/wifinet_cf.png width="400">|

| Comparison        |
| ----------------- |
|<img src=images/val_accuracy.png width="400">|

# References
1. W. Balid, M. O. Al Kalaa, S. Rajab, H. Tafish and H. H. Refai, "Development of measurement techniques and tools for coexistence testing of wireless medical devices," in 2016 IEEE Wireless Communications and Networking Conference Workshops, Doha, 2016. 
2. A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior and K. Kavukcuoglu, "WaveNet: A Generative Model for Raw Audio," arXiv, 2016. 
3. N. Bitar, S. Muhammad and H. H. Refai, "Wireless technology identification using deep Convolutional Neural Networks," in IEEE International Symposium on Personal, Indoor and Mobile Radio Communications, Montreal, 2017. 
4. S. Ruder, "An overview of gradient descent optimization algorithms," 19 January 2016. [Online]. Available: https://ruder.io/optimizing-gradient-descent/index.html#adam. [Accessed 3 July 2021].
5. J. Zhao, F. Huang, J. Lv, Y. Duan, Z. Qin, G. Li and G. Tian, "Do RNN and LSTM have Long Memory?," arXiv, 2020. 
6. S. Bai, Z. Kolter and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv, 2018. 
7. T. H. Trinh, A. M. Dai, M.-T. Luong and Q. V. Le, "Learning Longer-term Dependencies in RNNs with Auxiliary Losses," arXiv, 2018. 
8. E. Almazrouei, G. Gianini, N. Almoos and E. Damiani, "Robust Computationally-Efficient Wireless Emitter Classification Using Autoencoders and Convolutional Neural Networks," Sensors, vol. 21, no. 7, p. 2414, 2021. 
9. L. Bai, L. Yao, S. S. Kanhere, X. Wang and Z. Yang, "Automatic Device Classification from Network Traffic Streams of Internet of Things," in 43rd Conference on Local Computer Networks (LCN), Chicago, 2018. 
10. S. Rajendran, W. Meert, D. Giustiniano, V. Lenders and S. Pollin, "Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors," IEEE Transactions on Cognitive Communications and Networking, vol. 4, no. 3, pp. 433-445, 2018. 
11. X. Zhang, Y. Gao, Y. Yu and W. Li, "MUSIC ARTIST CLASSIFICATION WITH WAVENET CLASSIFIER FOR RAW WAVEFORM AUDIO DATA," arXiv, 2020. 
12. Cisco, "Cisco Annual Internet Report (2018–2023) White Paper," 9 March 2020. [Online]. Available: https://www.cisco.com/c/en/us/solutions/collateral/executive-perspectives/annual-internet-report/white-paper-c11-741490.html. [Accessed 1 7 2021].
13. J. Singh, "WaveNet: Google Assistant’s Voice Synthesizer.," Towards Data Science Inc., 7 November 2018. [Online]. Available: https://towardsdatascience.com/wavenet-google-assistants-voice-synthesizer-a168e9af13b1. [Accessed 15 June 2020].
14. J. Miller, "When Recurrent Models Don't Need to be Recurrent," The Berkeley Artificial Intelligence Research (BAIR) Lab, 6 August 2018. [Online]. Available: https://bair.berkeley.edu/blog/2018/08/06/recurrent/. [Accessed 15 June 2020].
15. M. Thiel, "An Intuitive Introduction to Deep Autoregressive Networks," Machine Learning at Berkeley, 16 August 2020. [Online]. Available: https://ml.berkeley.edu/blog/posts/AR_intro/. [Accessed 2 July 2021].
16. M. Andrews, "DeepMind's WaveNet : How it works, and how it is evolving - TensorFlow and Deep Learning," Engineers.SG (YouTube Channel), 23 January 2018. [Online]. Available: https://www.youtube.com/watch?v=YyUXG-BfDbE. [Accessed 15 June 2020].
17. J. Boilard, P. Gournay and R. Lefebvre, "A Literature Review of WaveNet: Theory, Application and Optimization," in 146th Convention of the Audio Engineering Society, Dublin, 2019. 


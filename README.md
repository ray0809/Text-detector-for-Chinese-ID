## Text detector for Chinese ID ##
This model is forked from [ppt_detecteor](https://gitlab.com/rex-yue-wu/ISI-PPT-Text-Detector)<br>
I add the training part and use it for detecting text in Chinese ID
***
### Environment ###
- python 2.7
- keras=1.2.2
- theano=0.9.0
- some python libraries which can be included in Anaconda 
***
### Usage ###

#### Training:<br>
**X:\Users\Ray**>python train\_demo.py<br>

***prompt***:you need create a list first and two folders(data and label),the format is similar to pascal_voc<br>
just like<br>
 data/1.jpg label/1.xml<br>
\*\*\*<br>
\*\*\*<br>




#### Testing:<br>
**X:\Users\Ray**>python test\_demo.py<br>


### Example ###

<div align="center">
<img src="https://github.com/ray0809/Text-detector-for-Chinese-ID/blob/master/examples/Figure_1.png" height="330" width="400" >

<img src="https://github.com/ray0809/Text-detector-for-Chinese-ID/blob/master/examples/Figure_2.png" height="330" width="400" >

 </div>

<div align="center">
<img src="https://github.com/ray0809/Text-detector-for-Chinese-ID/blob/master/examples/Figure_3.png" height="330" width="400" >

<img src="https://github.com/ray0809/Text-detector-for-Chinese-ID/blob/master/examples/Figure_4.png" height="330" width="400" >

 </div>




# AlexNet
AlexNet is 8 layer architecture with 5 layer of CNN. 

Video link:
https://www.youtube.com/watch?v=DAOcjicFr1Y

Quick overview:
Here I have created replica of architecture using keras.

It has 5 layer CNN which is detailed as follows

Layer-1<br/>
Conv 1 - filters=96 size=(11,11) stride=4 padding=0
<br/>
Pooling1- size=(3,3) stride=2
<br/>
Normalization- I have used BatchNormalization

Layer-2<br/>
Conv 2 - filters=256,size=(5,5) stride=1, padding=2
<br/>
Pooling2- size=(3,3) stride=2
<br/>
Normalization- I have used BatchNormalization

Layer-3<br/>
Conv 3 - filters=384,size=(3,3) stride=1, padding=1

Layer-4<br/>
Conv 4 - filters=384,size=(3,3) stride=1, padding=1

Layer-5<br/>
Conv 5 - filters=256,size=(3,3) stride=1, padding=1
<br/>
Pooling2- size=(3,3) stride=2
<br/>
Normalization- I have used BatchNormalization

Now it has 2 Dense Layer and one for output layer.
Both dense layer has 4096 units and output layer has 1000 units.



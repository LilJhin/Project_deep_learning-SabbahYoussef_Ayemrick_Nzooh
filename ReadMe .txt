The objective of our project is to detect in an automatic way, Violent Events in Video Streams. So how are we going to do this?

CNN are good when working with image data.
LSTM are great when working with sequences of data.
So, when you combine both of them, you can do things like video classification and solve problems like video action recognition.

We know that videos are made of multiple images knwon as frames. We want to take the temporal nature of our videos.

More precisely, in our project we shall use a many-to-one  LSTM approach. Our inputs shall be different frames of our videos where every frame shall be given to an LSTM cell and predict an output where our output shall be the class of the class of the video that is either violence on non violence.

We shall take a video split it into frames and then use a CNN to get an extract visual feature out of it and then fill those visual features to an LSTM network and then get predictions out of it.

The CNN will be responsible to to learn the spatial information and the LSTM the temporal information.

In our code, all the steps are clearly explained in order for the reader to understand precisely the approach we use.

 
# Toy project to detect numbers written on ping pong balls

## Goal

We write internal tickets on balls, so that we can pass them around, order them and talk about them in a visual and tangible way. I wanted to create a system that would allow us to "scan" a ball using a webcam and open the associated ticket in out bug tracking system.

## Method

- preprocess webcam image
- perform a Hough transformation to detect balls
- crop out the ball, threshold the image
- find contours, filter out the ones in the middle of the image
- when 5 contours are detected, crop them out and feed into the network
- when the same numbers are detected 5 times in a row, process the hit

## Files

- `train-tensorflow.py`: training the network, basically directly copied from the [deep mnist tensorflow example](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html)
- `mnist-model.ckpt`: the trained model, saved as checkpoint
- `deepMnist.py`: small library that contains out model
- `opencvcam.py`: the actual detection script. Performs the steps in the model above and sends a get request to a local endpoint
- `server.js`: a small nodejs server (because I am more familiar with websockets in node than in python). To process the request and load the ticket in an iframe
- `public/index.html`: the webpage handling the above websocket data

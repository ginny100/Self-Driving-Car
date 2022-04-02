# Self-driving Car

### This project is part of my journey into learning about Artificial Intelligence. This simplified self-driving car is implemented using Deep Q-Learning.

## I. Inspiration
[Case Study - Self-driving Car](https://docs.google.com/document/d/1tuqZVDI7hLM5jHx8se2AB26G0qIXZ3WgKPw293D7IcI/edit#s)

## II. Running
In order to run this project, first, make sure that you have installed Pytorch and Kivy.
For Mac users, inside the terminal, copy-paste, and enter each of the following line commands separately:

conda install pytorch torchvision -c pytorch
conda install -c conda-forge kivy

The IDE I use for this project is Spyder via Anaconda.
[Download Anaconda](https://www.anaconda.com/)

To run this project, open the map.py file and the ai.py (or improved_ai.py) file. Before hitting the "Run" button, make sure that you are inside the map.py file.

When running this project, the map window will show up with a little car inside. You can draw some roads or obstacles to your environment to challenge the self-driving car.
The car will learn to find the way to go from the Airport (top-left corner of the map) to the Downtown (bottom-right corner of the map) by trial and error.
Please note that, when you draw things to the map window or move that window around your screen, try not to change its size of it, otherwise, the program will get a bug and crash.

Before you quit the app, you can hit "Save" to see the graph of the learning process of the car as well as save what it has learned.
Next time, when you run the program again, you can just hit "Load" to load the saved brain of the car so that it does not have to re-learn everything and be ready for the next challenge.

The "Clear" button helps you clear whatever you have drawn on the screen.

## III. Improvements
I am still learning and working on the improved_ai.py. This [challenge](https://www.udemy.com/course/artificial-intelligence-az/learn/lecture/7626408#questions/2604982) is my inspiration
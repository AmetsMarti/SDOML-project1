#set par(
  justify: true
)

#align(center)[
   #text(size: 16pt, weight: "bold")[
    Report Software Development Oriented to Machine Learning
   ]

   #text(size: 13pt, weight: "bold")[
    Practice 1 — Project Structure, Training, and Visualization
   ]

   #v(1em)

  Teva Philippe - Alvaro Crespo - Amets Marti
]

#v(1em)

= 1. Introduction

#v(1em)

This first practice constitutes the first step in setting up our machine learning project. The objective is
to integrate a public dataset, train a simple classification model, and analyze its performance.

We chose to work on digit classification. The principle is similar
to *MNIST*, which allows digits to be recognized from images, but here we use audio recordings instead.

= 2. Project Structure and Dataset

== Project Structure

#v(1em)

In order to organize our project in a clear and professional manner, we
used *Cookiecutter Data Science* to set up the project structure.

== Dataset

#v(1em)

We chose the *Audio MNIST* dataset. Similarly to the classic MNIST
dataset, which contains images representing digits from 0 to 9, Audio MNIST contains audio recordings.

Each example is associated with a digit between 0 and 9. For each file, we also have access to metadata such as the speaker's accent, age, and gender.

We have access to 3000 recordings divided into 2 groups:
2400 for the training data group
600 for the test data group

It is worth noting that the data is generally balanced. However, regarding accents, we observed that the German accent was more represented in the training data, but we will not focus on this aspect here.


== 3. Data Exploration

#v(1em)

To begin with, we created 4 graphs, 3 using matplotlib and 1 using seaborn:

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,

   figure(
   image("../assets/figure1.png", width: 100%),
   caption: [Bar Chart]
   ),

   figure(
   image("../assets/figure2.png", width: 100%),
   caption: [Histogram]
   )
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
   figure(
   image("../assets/figure3.png", width: 100%),
   caption: [Bar Chart]
   ),

   figure(
   image("../assets/figure4.png", width: 100%),
   caption: [Boxplot]
   )
)


For this part, we first built our dataset without the audio files, since they were not necessary at this stage.

For the next part, we need the recordings because we are moving on to the ML section. We thought about how to provide the audio files to our model and found librosa, a library that allows us to transform the audio into a sequence of 26 digits.

= Model and Training

== Model Architecture

#v(1em)

We chose to use PyTorch to build and train our
classification model.

The architecture is based on the example from Unit 2.2 of the course.
The 26 features extracted from each audio recording are first passed to a
fully connected layer of 64 neurons. Note that this value was chosen arbitrarily.
We then reduce this to 10 neurons corresponding to the ten possible digits.

It should be noted that initially, we reduced the number of neurons from 64 to 32, and then to 10. However, the results plateaued somewhat, and we hypothesize—though this should be taken with a grain of salt since we are not certain—that the model was memorizing the data. Going directly from 64 to 10 neurons seems to have resolved this issue. A second parameter we varied is...

(I’ll let you explain this one)

The results aren’t significantly better in the end, but we can now observe a trend, which we recorded here over 40 epochs—that is, 40 passes through the training data.

== Confusion matrice

#v(1em)

In any case, in practice, we have managed to produce results that we find interesting.

We created confusion matrices for each epoch to verify the actual results produced by our model. You can see, for each training epoch (i.e., across 2,400 data points), the results produced by our model


#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
   figure(
   image("../assets/confusion1.png", width: 100%),
   caption: [Confusion 1]
   ),

   figure(
   image("../assets/confusion10.png", width: 100%),
   caption: [Confusion 10]
   ),

   figure(
   image("../assets/confusion40.png", width: 100%),
   caption: [Confusion 40]
   ),
)

Above, we show the changes between the first round, the 10th round, and the 40th round. The loss continues to decrease (albeit slowly) and the accuracy continues to increase (albeit slowly) even after 40 rounds, but it is especially in the first rounds that the results seem to suggest that our model is learning.

== Loss / Accuracy

#v(1em)



todo : loss / accuracy


== 4. Conclusions

todo

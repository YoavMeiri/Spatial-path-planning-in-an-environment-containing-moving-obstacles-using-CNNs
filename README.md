# Cognitive Robotics Course Project (Technion 097244) 
We consider the problem of spatial path planning, in an environment containing
moving obstacles. In contrast to the classical solutions, in this work we learn a planner
from the data (in a supervised learning manner) that allows us to leverage statistical
regularities from past data. For that purpose we will use a state of the art machine
learning model. In the setting where the ground truth map (real distance from the
goal from each tile) is not known to the agent, we leverage pre-trained deep learning
models (which in our case will contain convolutional neural networks) in an end-to-end
framework that has the structure of mapper and planner built into it which allows
seamless generalization to new maps and goals.

Our experimental results showed that our model can achieve faster and
close to optimal solutions than the baseline method, while also being able to handle complex
and dynamic scenarios. We attribute the success of our model to its ability to learn highlevel features and spatial relations from the input images, and to generate feasible and
diverse layouts using a decoder network (fully-connected head). Our model can be applied
to various domains that require spatial planning, such as urban design, interior design,
robotics, and game development.


<p align="center">
	<img src="https://user-images.githubusercontent.com/81311717/233570278-2ca55799-f673-48c9-9ea4-3dbe4494caa3.png" width="800">
</p>

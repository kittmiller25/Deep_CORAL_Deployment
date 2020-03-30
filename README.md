# Domain Adaptation and Semi-Supervised Learning With Deep Coral
## Introduction
Machine learning practitioners are frequently hindered and set back in their efforts by an inadequate amount of labeled data in their chosen domains. In these situations gathering additional relevant, labeled data is often prohibitively expensive or time consuming. As a result ML professionals frequently seek methods of incorporating additional readily available data that comes from another domain and/or is unlabelled. Deep CORAL is a cutting edge algorithm for improving model accuracy through the incorporation of such additional data sources. It can be used for both incorporating out-of-domain data (i.e. images taken with a different camera type or setting as in the Amazon vs. webcam images of the Office 31 dataset) or for semi-supervised learning with a small number of labeled images and a larger body of in-domain, unlabeled images. Deep CORAL is able to accomplish this goal by using a nonlinear transformation that aligns the correlations of layer activations in deep neural networks. (Based on Deep CORAL: Correlation Alignment for Deep Domain Adaptation by Baochen Sun and Kate Saenko - https://arxiv.org/abs/1607.01719v1 )

## Deployment
This deployment was created using the Amazon Web Services platform and can be simply utilized via the Deployment notebook that follows, provided that the user has AWS credentials for passing in training data and has been enable for access by the deployment owner. The flow for the deployment is as follows: 
- The user organizes the data in their local directory as described in the notebook
- The notebook proceeds to package the data into a tarfile and submit the data to the user's S3 cloud storage bucket of choice 
- The notebook then sends a training request with the data location to the host's SQS queue 
- The SQS queue automatically triggers a Lambda function
- The Lambda function starts an ECS task and passes in environment variables from the SQS message
- The ECS task spins up a container that downloads the training data from the user's S3 bucket and a train script and a serve script from the host's S3 bucket and then runs the training script
- The training script trains a model and creates performance data graphs from the training process, it then uploads the trained model and performance graphs to the user's S3 bucket along with a zip file with the model's artifacts, finally the training script uses the serve script to create a Sagemaker endpoint for the trained model and returns the endpoint location to the host's listener SQS queue
- The notebook checks the listener queue for messages and once training and endpoint creation are complete, receives the endpoint location
- Using a boto3 client session the user invokes the provided endpoint to receive inferences
 
## Future Improvements
- Implement callbacks for the model during training such that the model from the epoch that produces the best result is what is saved and passed out of the training script
- Implement Central Moment Discrepancy (CMD) loss and compare results to Deep CORAL, CMD paper - https://arxiv.org/abs/1702.08811, CMD implementation - https://github.com/wzell/cmd
- Enable the user to select the model layer from which they would like Deep CORAL to run on the activations, as done in this implementation - https://github.com/jindongwang/transferlearning/tree/master/code/deep/DDC_DeepCoral
- Amend the serve script and endpoint creation to allow for the user to submit a jpeg file directly rather than only accepting an image url how-to here: https://github.com/shashankhalo7/Helpful-Scripts/blob/master/Sagemaker/Deploy.ipynb and https://github.com/shashankhalo7/Helpful-Scripts/blob/master/Sagemaker/serve.py
- Change Auto-scaling cluster in ECS from managed EC2 type to Fargate once Fargate supports GPU instances
- Switch from PyTorch implementation to Fastai
 

   
   
Footnote - The PyTorch deployment of the Deep CORAL algorithm was adapted from here - https://github.com/DenisDsh/PyTorch-Deep-CORAL/. The AWS deployment was adapted from a number of tutorials, most notably - https://aws.amazon.com/blogs/compute/better-together-amazon-ecs-and-aws-lambda/ and https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_lstm_word_language_model/pytorch_rnn.ipynb

#!/bin/bash

# Copyright 2015 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file.
# This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

#
# Adapted from the AWS tutorial here - https://aws.amazon.com/blogs/compute/better-together-amazon-ecs-and-aws-lambda/
#
# ENV variables for the data location are passed in from the ECS Task that triggers the container that runs this script. It then fetches 
# a tar file from the specified S3 bucket and a training script from the host bucket, runs the Deep CORAL implementation in the script on 
# the provided data and then upload the results and trained model to the user specified S3 bucket that the data was located in.

# Set SQS URL for results messages to be sent to

listen_url = *<INSERT YOUR SQS LISTENER URL HERE>*

# Fetch target data information from environment variables that are passed to the container when it is initiated by the ECS Task, which was 
# triggered by the Lambda function, which was triggered by the SQS queue message sent by the deployment notebook

bucket=${BUCKET}
echo "Bucket: ${bucket}"
key=${FNAME}
echo "Key: ${key}"

base=${key/.tar.gz}
echo "Base: ${base}"
ext=${key##*.}
echo "Ext: ${ext}"

# Check that the data location information passed values in
if [ \
    -n "${key}" -a \
    -n "${base}" -a \
    -n "${ext}" -a \
    "${ext}" = "gz" \
]; then
    mkdir -p work
    pushd work

    # Download the training data from the user specified bucket and filename and the training and serve scripts from the deployment owner's 
    # host S3 location

    host_bucket = *<INSERT YOUR HOST BUCKET NAME HERE>*
    echo "Copying ${key} from S3 bucket ${bucket}..."
    aws s3 cp s3://${bucket}/${key} .
    aws s3 cp s3://${host_bucket}/train_demo.py .
    aws s3 cp s3://${host_bucket}/serve_demo.py .

    if ! [[ -f "${key}"   ]]; then
        echo "ERROR: Unable to acquire data from the specified bucket and filename."
        aws sqs send-message --queue-url ${listen_url} \
            --message-body \
            "Unable to acquire data from the specified bucket- ${bucket} and filename- ${key}, check its existence and provided format"           
        exit 3
    fi


    # Unzip data and run the training script

    echo "Unzipping ${key}..."
    tar -xzf ${key}
    params=$(<./data/trainingparameters.txt)
    echo "Training Parameters: ${params}"

    echo "Beginning training..."
    python train_demo.py --naming ${base} --bucket ${bucket} ${params}
    
    endpoint=$(<./endpoint.txt)
    
    # Check if training succeeded in producing a model, if so upload files to user's bucket and send success message with the created 
    # endpoint deployment location, otherwise send failure message

    if [ -f "./${base}_model.pt" ]; then

        echo "Training complete, uploading files..."
        aws s3 cp ${base}_model.pt s3://${bucket}/${base}_model.pt
        aws s3 cp accuracies.pdf s3://${bucket}/${base}_accuracies.pdf
        aws s3 cp losses_adapt.pdf s3://${bucket}/${base}_losses_adapt.pdf

        echo "Cleaning up..."
        popd
        /bin/rm -rf work

        aws sqs send-message \
            --queue-url ${listen_url} \
            --message-body "Training complete and data uploaded. Endpoint created: ${endpoint}" 
        exit 1
    else
        echo "Training failed..."

        aws sqs send-message --queue-url ${listen_url} --message-body "Training failed"
        exit 2
    fi            

else
    echo "ERROR: Could not extract S3 bucket and key from SQS message."
    aws sqs send-message --queue-url ${listen_url} \
        --message-body "Failed to acquire data from S3 bucket"
    exit 3	
fi
exit 4
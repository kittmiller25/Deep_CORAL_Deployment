// Copyright 2015 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// A copy of the License is located at
//
//    http://aws.amazon.com/apache2.0/
//
// or in the "license" file accompanying this file.
// This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

// This AWS Lambda function is prompted by an entry to the DA_API_Q2 SQS queue and starts an Amazon ECS task to
// process that event.

var async = require('async');
var aws = require('aws-sdk');
var ecs = new aws.ECS({apiVersion: '2014-11-13'});



exports.handler = function(event, context) {
    console.log('Received event:');
    console.log(JSON.stringify(event, null, '  '));
    var bucket = event.Records[0].messageAttributes.bucket.stringValue;
    var fname = event.Records[0].messageAttributes.fname.stringValue;
    var receipt_handle = event.Records[0].receiptHandle;
    
    console.log(bucket);
    console.log(fname);
    console.log(receipt_handle);
    
    // Start an Amazon ECS task to process the event.
    async.waterfall([
            
            function (next) {
                // Starts an ECS task to work through the feeds.
                console.log('Setting params');
                var params = {
                    taskDefinition: '*<INSERT NAME OF YOUR ECS TASK HERE>*',
                    count: 1,
                    cluster: 'INSERT NAME OF YOUR ECS CLUSTER HERE',
                    overrides: {
                        containerOverrides: [
                            {
                                name: 'INSERT NAME OF THE CONTAINER IN YOUR ECS TASK DEFINITION HERE',
                                environment: [
                                    {
                                    name: 'BUCKET',
                                    value: bucket
                                    },
                                    {
                                    name: 'FNAME',
                                    value: fname
                                    }
                                ]
                            }
                        ]
                    }
                };
                console.log('Starting ECS task');
                console.log(params);
                ecs.runTask(params, function (err, data) {
                    if (err) { console.warn('error: ', "Error while starting task: " + err); }
                    else { console.info('Task-DA_API_Task started: ' + JSON.stringify(data.tasks))}
                    next(err);
                });
            }
        ], function (err) {
            if (err) {
                context.fail('An error has occurred: ' + err);
            }
            else {
                context.succeed('Successfully processed Amazon S3 URL.');
            }
        }
    );
};
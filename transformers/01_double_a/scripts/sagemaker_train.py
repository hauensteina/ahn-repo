
from sagemaker.pytorch import PyTorch

# You need this under trusted entities for your Sagemaker role on AWS console:
'''
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
'''

# This should be called 'command_line_args_for_your_train_script'
hyperparameters = {
    'sagemaker': 1,
    'num_epochs': 2000,
    }

aws_role = 'arn:aws:iam::147785435127:role/service-role/SageMaker-ahx'


# AWS
pytorch_estimator = PyTorch(
    entry_point='train.py', # My train script 
    role=aws_role,
    train_instance_count=1,
    train_instance_type='ml.m5.large',
    source_dir='scripts',
    framework_version='2.0', # PyTorch version
    py_version='py310',
    hyperparameters=hyperparameters
)
# Fit the model
fit_parms = {
    'train': 's3://ahx-sagemaker/double_a_data', # Folder gets copied to os.environ['SM_CHANNEL_TRAIN'] by magic
}
pytorch_estimator.fit( fit_parms)  


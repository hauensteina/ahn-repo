
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch

# Create a local Session. 
sagemaker_local_session = LocalSession()
sagemaker_local_session.config = {'local': {'local_code': True}}

# Create PyTorch estimator.
hyperparameters = {
    'sagemaker': 1,
    'num_epochs': 2000,
    }

aws_role = 'arn:aws:iam::147785435127:role/service-role/SageMaker-ahx'

# local
'''
pytorch_estimator = PyTorch(
    entry_point='train.py',  # My training script
    role=aws_role,   # IAM role from AWS console -> Sagemaker
    instance_count=1,
    instance_type='local',
    framework_version='2.0',  # PyTorch version
    py_version='py310',
    source_dir='scripts',
    hyperparameters=hyperparameters,
    session=sagemaker_local_session
    )
# Fit the model
fit_parms = {
    'train': 'file://iris_data', # Folder gets copied to os.environ['SM_CHANNEL_TRAIN'] by magic
}
pytorch_estimator.fit( fit_parms)  
'''

# AWS
pytorch_estimator = PyTorch(
    entry_point='train.py',
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
    'train': 's3://ahx-sagemaker/iris_data', # Folder gets copied to os.environ['SM_CHANNEL_TRAIN'] by magic
}
pytorch_estimator.fit( fit_parms)  


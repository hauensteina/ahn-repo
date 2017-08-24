
aws-t2() {
    export instanceId=`aws ec2 describe-instances --filters "Name=instance-state-name,Values=stopped,Name=instance-type,Values=t2.micro" --query "Reservations[0].Instances[0].InstanceId" | sed 's/"//g'` && echo $instanceId
    export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query "Reservations[0].Instances[0].PublicIpAddress" | sed 's/"//g'` && echo $instanceIp
}

aws-p2() {
    export instanceId=`aws ec2 describe-instances --filters "Name=instance-state-name,Values=stopped,Name=instance-type,Values=p2.xlarge" --query "Reservations[0].Instances[0].InstanceId" | sed 's/"//g'` && echo $instanceId
    export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query "Reservations[0].Instances[0].PublicIpAddress" | sed 's/"//g'` && echo $instanceIp
}

aws-ip() {
    export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query "Reservations[0].Instances[0].PublicIpAddress" | sed 's/"//g'` && echo $instanceIp
}

alias aws-start='aws ec2 start-instances --instance-ids $instanceId && aws ec2 wait instance-running --instance-ids $instanceId && export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query "Reservations[0].Instances[0].PublicIpAddress"` && echo $instanceIp'

alias aws-stop='aws ec2 stop-instances --instance-ids $instanceId'

alias aws-ssh='ssh -i ~/.ssh/aws-key.pem ubuntu@$instanceIp'

t2mount() {
    sudo  cd ~ahauenst
    sudo umount /aws-t2
    sudo sshfs  -o IdentityFile=/Users/ahauenst/.ssh/aws-key.pem -o cache_dir_timeout=10 -o reconnect -o allow_other,defer_permissions -o Compression=no ubuntu@34.208.16.115:/home/ubuntu /aws-t2
}
p2mount() {
    sudo  cd ~ahauenst
    sudo umount /aws-p2
    sudo sshfs  -o IdentityFile=/Users/ahauenst/.ssh/aws-key.pem -o cache_dir_timeout=10 -o reconnect -o allow_other,defer_permissions -o Compression=no ubuntu@52.88.21.18:/home/ubuntu /aws-p2
}

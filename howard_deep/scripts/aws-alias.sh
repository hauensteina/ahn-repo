
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

aws-mount() {
    sudo  cd ~ahauenst
    sudo umount /aws
    sudo sshfs -o cache_dir_timeout=10 -o reconnect -o allow_other,defer_permissions -o Compression=no ec2-user@$instanceIp:/home/ec2-user /aws
}

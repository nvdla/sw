#!/usr/bin/env sh

# mount first
#mount -t 9p -o trans=virtio r /mnt

echo "Install NVDLA driver"
insmod /mnt/drm.ko
insmod /mnt/opendla.ko

echo "Config SSH for regression envrionment"
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/'  /etc/ssh/sshd_config
/etc/init.d/*sshd restart

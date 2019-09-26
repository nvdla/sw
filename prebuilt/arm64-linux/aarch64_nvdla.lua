CPU = {
    library = "libqbox-nvdla.so",
    extra_arguments = "-machine virt -cpu cortex-a57 -machine type=virt -nographic -smp 1 -m 1024 -kernel /usr/local/nvdla/Image --append \"root=/dev/vda\" -drive file=/usr/local/nvdla/rootfs.ext4,if=none,format=raw,id=hd0 -device virtio-blk-device,drive=hd0 -fsdev local,id=r,path=.,security_model=none -device virtio-9p-device,fsdev=r,mount_tag=r -netdev user,id=user0,hostfwd=tcp::6666-:6666,hostfwd=tcp::6667-:22, -device virtio-net-device,netdev=user0"
}

ram = {
    size = 1048576,
    target_port = {
        base_addr = 0xc0000000,
        high_addr = 0xffffffff
    }
}

nvdla = {
    irq_number = 176,
    csb_port = {
        base_addr = 0x10200000,
        high_addr = 0x1021ffff
    }
}

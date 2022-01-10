grep -r telnetd . #find all the related file / file content contain telnetd

    #-r for recursive


binwalk -t d606a.bin

DECIMAL       HEXADECIMAL     DESCRIPTION
------------------------------------------------------------------------------------------------------------------------
512           0x200           uImage header, header size: 64 bytes, header CRC: 0xE370AC3E, created: 2016-04-01
                              06:37:54, image size: 1171160 bytes, Data Address: 0x80008000, Entry Point: 0x80008000,
                              data CRC: 0xEE2F6311, OS: Linux, CPU: ARM, image type: OS Kernel Image, compression
                              type: none, image name: "Linux-3.4.35"
576           0x240           Linux kernel ARM boot executable zImage (little-endian)
8354          0x20A2          LZMA compressed data, properties: 0x5D, dictionary size: 67108864 bytes, uncompressed
                              size: -1 bytes
1171744       0x11E120        Squashfs filesystem, little endian, version 4.0, compression:xz, size: 2009690 bytes, 170
                              inodes, blocksize: 131072 bytes, created: 2017-10-19 11:13:23
3182880       0x309120        Squashfs filesystem, little endian, version 4.0, compression:xz, size: 2998322 bytes, 144
                              inodes, blocksize: 65536 bytes, created: 2017-10-19 11:13:25


chmod +x 360camera_unpack.py

./360camera_unpack.py unpack d606a.bin
'''
Wrote uimage_header - 0x40 bytes
Wrote uimage_kernel - 0x11dee0 bytes
Wrote squashfs_1 - 0x1eb000 bytes
Wrote squashfs_2 - 0x2dd000 bytes
'''

unsquashfs -d squashfs_1_out squashfs_1
unsquashfs -d squashfs_2_out squashfs_2

#jefferson -d jffs2_out jffs2
#i dont have jffs2 section for this job, but this commnad can be used to unpack jffs2





grep -r telnetd .

string ./jffs2_out/fs_1/bin/iCamera | grep telentd

ls -l squashfs_1_out/sbin/telnetd


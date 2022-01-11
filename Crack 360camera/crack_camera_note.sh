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
#to find out the hardlink of the telnetd.
'''
lrwxrwxrwx 1 guanxv guanxv 14 Jan 28  2016 squashfs_1_out/sbin/telnetd -> ../bin/busybox
'''
#the telnetd is hard linked with busy box


#I modified the /etc/init.d/rcS with extra line of code

# start telnet daemon
busybox telnetd &

# I have added same code to rc.local

# I have also modified the 360.sh to change the led blink from green to red.
killall -9 ledCtrl
ledCtrl -t red -m flicker &

#start to pack

#check the original squashfs info
guanxv@guanxv-ubuntu:~/360camera/unpack_mod_backdoor$ unsquashfs -s squashfs_1
'''
Found a valid SQUASHFS 4:0 superblock on squashfs_1.
Creation or last append time Thu Oct 19 22:13:23 2017
Filesystem size 2009690 bytes (1962.59 Kbytes / 1.92 Mbytes)
Compression xz
Block size 131072
Filesystem is exportable via NFS
Inodes are compressed
Data is compressed
Uids/Gids (Id table) are compressed
Fragments are compressed
Always-use-fragments option is not specified
Xattrs are compressed
Duplicates are not removed
Number of fragments 16
Number of inodes 170
Number of ids 2
'''

guanxv@guanxv-ubuntu:~/360camera/unpack_mod_backdoor$ mksquashfs squashfs_1_out/ squashfs_1_new -comp xz -b 131072
'''
Parallel mksquashfs: Using 6 processors
Creating 4.0 filesystem on squashfs_1_new, block size 131072.
[==========================================================================================================================================================================/] 86/86 100%

Exportable Squashfs 4.0 filesystem, xz compressed, data block size 131072
	compressed data, compressed metadata, compressed fragments,
	compressed xattrs, compressed ids
	duplicates are removed
Filesystem size 1962.40 Kbytes (1.92 Mbytes)
	32.46% of uncompressed filesystem size (6045.67 Kbytes)
Inode table size 1194 bytes (1.17 Kbytes)
	20.69% of uncompressed inode table size (5772 bytes)
Directory table size 1558 bytes (1.52 Kbytes)
	56.72% of uncompressed directory table size (2747 bytes)
Number of duplicate files found 0
Number of inodes 170
Number of files 57
Number of fragments 16
Number of symbolic links  95
Number of device nodes 0
Number of fifo nodes 0
Number of socket nodes 0
Number of directories 18
Number of ids (unique uids + gids) 1
Number of uids 1
	guanxv (1000)
Number of gids 1
	guanxv (1000)
'''
 #do the same thing for squashfs2

 #overide the original squash file

mv squashfs_1_new squashfs_1
mv squashfs_2_new squashfs_2
 
#the 360camera_unpack.py was modified to do the packing job.

#use 360camera_unpack.py to pack
./360camera_unpack.py pack d606a_backdoored.bin

#use bin walk to check 
'''
DECIMAL       HEXADECIMAL     DESCRIPTION
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0             0x0             uImage header, header size: 64 bytes, header CRC: 0xE370AC3E, created: 2016-04-01 06:37:54, image size: 1171160 bytes, Data Address: 0x80008000, Entry
                              Point: 0x80008000, data CRC: 0xEE2F6311, OS: Linux, CPU: ARM, image type: OS Kernel Image, compression type: none, image name: "Linux-3.4.35"
64            0x40            Linux kernel ARM boot executable zImage (little-endian)
7842          0x1EA2          LZMA compressed data, properties: 0x5D, dictionary size: 67108864 bytes, uncompressed size: -1 bytes
1171232       0x11DF20        Squashfs filesystem, little endian, version 4.0, compression:xz, size: 2009502 bytes, 170 inodes, blocksize: 131072 bytes, created: 2022-01-11 10:15:34
3182368       0x308F20        Squashfs filesystem, little endian, version 4.0, compression:xz, size: 2998238 bytes, 144 inodes, blocksize: 65536 bytes, created: 2022-01-11 10:18:26
'''

sudo apt-get install u-boot-tools
# get the mkimage command by installing above package

#use mkimage to make image
mkimage -A ARM -O linux -T OS Kernel Image -C none -a 0x80008000 -e 0x80008000 -n "Linux-3.4.35" -d d606a_backdoored.bin d606a.bin

# but i got a invalid image type has to change -T to firmware , which is not match witht the original image

mkimage -A ARM -O linux -T firmware -C none -a 0x80008000 -e 0x80008000 -n "Linux-3.4.35" -d d606a_backdoored.bin d606a.bin













# Reference https://www.youtube.com/watch?v=hV8W4o-Mu2o&t=475s



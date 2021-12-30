#how to crack a linux password. 

go to /etc folder. find passwd and shadow file.

when open shadow. you cound find

    root:$1$60a6wQfe$XQEdtkH8HMvQis1IZIwT50:16384:0:99999:7:::
    guanxv:$6$TYEUnI5fP7zjLSTk$fVyPSA19N1IDoDAAdB8orxfjk3gEr20PxG5tfJ3ZobD7a2Up532LXRo1JuSJ7CV0jS6Obdj4LnV3WLIYU/S0:18656:0:99999:7:::

$1$ is MD5
$2a$ is Blowfish
$2y$ is Blowfish
$5$ is SHA-256
$6$ is SHA-512

if the password is simple enough , you can use

john --fork=4 shadow # to crack the password.

but most of the times, the password is complex, you need to use the hackcat to crack the password.

hackcat is able to use GPU, so the speed is very fast.


sample command:

hackcat -a3 -m500 "$1$60a6wQfe$XQEdtkH8HMvQis1IZIwT50" 

    #$1$ indicate it is MD5. 
    #60a6wQfe$ is the salt, 
    #XQEdtkH8HMvQis1IZIwT50 is the actaul hash of the password.

hackcat is able to crack many types of password. see help for more inforamtion 

sample of using mask attack

hashcat -a 3 -m 0 example0.hash ?a?a?a?a?a?a #will try all the 6 keyspaces (uper lower letter + digit + special)


  ? | Charset
 ===+=========
  l | abcdefghijklmnopqrstuvwxyz
  u | ABCDEFGHIJKLMNOPQRSTUVWXYZ
  d | 0123456789
  h | 0123456789abcdef
  H | 0123456789ABCDEF
  s |  !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  a | ?l?u?d?s
  b | 0x00 - 0xff


# by the way to extrac a firmware and modify and packing back to a bin file see this link https://www.youtube.com/watch?v=hV8W4o-Mu2o&t=314s




